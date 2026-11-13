"""The training loop shared by the BiLSTM and the transformers.

A neural model here is an encoder (text -> one vector per question) with an output head on top.
`NeuralModel` trains the two together and follows the model contract of `levellens/models/base.py`.
A subclass only says how to turn text into token ids and how to build its encoder:

    class BiLSTM(NeuralModel):
        def prepare(self, train, config): ...     # build the vocabulary / load the tokeniser
        def encode(self, items): ...              # -> one list of token ids per question
        def build_encoder(self): ...              # -> nn.Module with .out_features
        def extra_state(self) / restore_extra(self, folder, state)   # what save() and load() need on top

What the loop does, in the order of one run:
- class-weighted loss (optional): rarer levels count more, see `levellens.heads.class_weights`
- AdamW, learning rate warms up linearly and then decays linearly to zero, gradient clipping
- dynamic padding: every batch is padded to its own longest question, not to a fixed length,
  and questions of similar length share a batch (`length_buckets`)
- mixed precision on the GPU (bf16), never fp16: DeBERTa-v3 gives NaN losses in fp16
- after every evaluation: QWK on dev. The best weights are kept; training stops when QWK has not
  improved for `patience` evaluations in a row (early stopping), or when `max_minutes` is over
- `on_eval(step, qwk)` is called after every evaluation, so Optuna can prune a weak trial
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from levellens.evaluate import qwk, score_output
from levellens.heads import build_head, class_weights
from levellens.models.base import Model

WEIGHTS_FILE = "weights.pt"
STATE_FILE = "model.json"
BUCKET_BATCHES = 50  # batches per pool in `length_buckets`

# Settings every neural model understands. A model config overrides them under `params:`.
COMMON_DEFAULTS = {
    "epochs": 5,               # passes over the training set, at most
    "batch_size": 32,
    "eval_batch_size": 128,
    "lr": 2e-5,                # peak learning rate
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,       # share of the steps in which the learning rate rises from 0 to `lr`
    "grad_clip": 1.0,          # largest gradient norm; 0 switches clipping off
    "dropout": 0.1,            # on the question vector, before the head
    "class_weights": True,     # weight the loss so that every level counts equally
    "evals_per_epoch": 1,      # how often QWK on dev is measured
    "patience": 2,             # stop after this many evaluations without a better dev QWK; 0 = never
    "max_minutes": 50,         # stop training after this long, to stay under 1 GPU-hour per run; 0 = no limit
    "precision": "auto",       # auto, fp32 or bf16. auto = bf16 on a GPU that has it, fp32 elsewhere
}


class Network(nn.Module):
    """Encoder, dropout and head: token ids in, logits out."""

    def __init__(self, encoder: nn.Module, head: nn.Module, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = head

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.encoder(input_ids, mask)))


def pad_batch(rows: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the token ids of one batch to its longest row. Return the ids and the mask (True = real token)."""
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.as_tensor(row, dtype=torch.long)
    lengths = torch.as_tensor([len(row) for row in rows])
    return input_ids, torch.arange(width)[None, :] < lengths[:, None]


def length_buckets(rows: Sequence[Sequence[int]], batch_size: int, generator: torch.Generator) -> list[list[int]]:
    """Return the batches of one epoch, as lists of row numbers, in random order.

    The rows are shuffled, cut into pools of BUCKET_BATCHES batches, and sorted by length inside each
    pool. So the questions of a batch have similar lengths and little padding is computed, while the
    batches still differ from epoch to epoch.
    """
    order = torch.randperm(len(rows), generator=generator).tolist()
    pool = batch_size * BUCKET_BATCHES
    batches = []
    for start in range(0, len(order), pool):
        chunk = sorted(order[start : start + pool], key=lambda row: len(rows[row]))
        batches += [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
    return [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]


def linear_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """Learning rate: linear rise over `warmup_steps`, then linear fall to zero at `total_steps`."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class NeuralModel(Model):
    """Base class of the BiLSTM and the transformer wrapper. Subclasses fill in the five hooks below."""

    DEFAULTS: dict = {}  # settings of the subclass, on top of COMMON_DEFAULTS
    pad_id = 0

    def __init__(self, num_levels: int, **params):
        unknown = set(params) - set(COMMON_DEFAULTS) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"{type(self).__name__} has no setting {sorted(unknown)}")
        super().__init__(num_levels, **{**COMMON_DEFAULTS, **self.DEFAULTS, **params})
        self.net: Network | None = None
        self.device = "cpu"
        self.head_config: dict = {}
        self.history: list[dict] = []  # one entry per evaluation: step, epoch, train loss, dev QWK
        self.on_eval: Callable[[int, float], None] | None = None

    # ---- hooks for the subclass -------------------------------------------------------------

    def prepare(self, train: Sequence[dict], config: DictConfig) -> None:
        """Get ready to encode text: build the vocabulary or load the tokeniser."""
        raise NotImplementedError

    def encode(self, items: Sequence[dict]) -> list[list[int]]:
        """Return the token ids of each item, already cut to the longest allowed length."""
        raise NotImplementedError

    def build_encoder(self) -> nn.Module:
        """Return the encoder: `forward(input_ids, mask) -> [n, out_features]`, with an `out_features` attribute."""
        raise NotImplementedError

    def extra_state(self, folder: Path) -> dict:
        """Save what `encode` and `build_encoder` need after a reload. Return JSON-ready settings."""
        return {}

    def restore_extra(self, folder: Path, state: dict) -> None:
        """Undo `extra_state`: afterwards `encode` and `build_encoder` work without the training set."""

    # ---- training ---------------------------------------------------------------------------

    def _autocast(self):
        """Return the mixed-precision context for the forward pass."""
        precision = self.params["precision"]
        on_gpu = str(self.device).startswith("cuda")
        if precision == "auto":
            precision = "bf16" if on_gpu and torch.cuda.is_bf16_supported() else "fp32"
        if precision not in ("fp32", "bf16"):
            raise ValueError(f"precision={precision!r}: use auto, fp32 or bf16 (fp16 gives NaN losses with DeBERTa-v3)")
        return torch.autocast(device_type="cuda" if on_gpu else "cpu", dtype=torch.bfloat16, enabled=precision == "bf16")

    def fit(self, train, dev, config) -> None:
        p = self.params
        self.device = str(config.device)
        self.head_config = OmegaConf.to_container(config.head, resolve=True)
        torch.manual_seed(int(config.seed))
        clock = time.perf_counter()

        self.prepare(train, config)
        rows = self.encode(train)
        y = torch.as_tensor([item["level"] for item in train], dtype=torch.long)
        dev_rows = self.encode(dev)
        dev_y = np.array([item["level"] for item in dev])

        weights = class_weights(y, self.num_levels) if p["class_weights"] else None
        encoder = self.build_encoder()
        head = build_head(config.head, encoder.out_features, self.num_levels, weights)
        self.net = Network(encoder, head, p["dropout"]).to(self.device)

        steps_per_epoch = math.ceil(len(rows) / p["batch_size"])
        total_steps = steps_per_epoch * p["epochs"]
        eval_every = max(1, steps_per_epoch // p["evals_per_epoch"])
        optimizer = torch.optim.AdamW(self.parameter_groups(), lr=p["lr"], weight_decay=p["weight_decay"])
        schedule = linear_schedule(optimizer, int(p["warmup_ratio"] * total_steps), total_steps)
        shuffle = torch.Generator().manual_seed(int(config.seed))

        self.history, best_qwk, best_weights, stale, step, running = [], -math.inf, None, 0, 0, []
        stop = ""
        for epoch in range(p["epochs"]):
            for batch in length_buckets(rows, p["batch_size"], shuffle):
                input_ids, mask = pad_batch([rows[i] for i in batch], self.pad_id)
                self.net.train()
                with self._autocast():
                    logits = self.net(input_ids.to(self.device), mask.to(self.device))
                loss = head.loss(logits, y[batch].to(self.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Loss is {loss.item()} at step {step}. Try precision=fp32 or a lower lr.")
                optimizer.zero_grad()
                loss.backward()
                if p["grad_clip"]:
                    nn.utils.clip_grad_norm_(self.net.parameters(), p["grad_clip"])
                optimizer.step()
                schedule.step()
                running.append(loss.item())
                step += 1

                if step % eval_every and step != total_steps:
                    continue
                score = qwk(score_output(dev_y, self._proba(dev_rows), self.num_levels)) or 0.0  # None: one level only
                minutes = (time.perf_counter() - clock) / 60
                self.history.append({"step": step, "epoch": round(step / steps_per_epoch, 2),
                                     "train_loss": round(float(np.mean(running)), 4), "dev_qwk": round(score, 4),
                                     "minutes": round(minutes, 2)})
                print(f"  epoch {step / steps_per_epoch:5.2f}  loss {np.mean(running):.4f}  dev QWK {score:.4f}  {minutes:.1f} min", flush=True)
                running = []
                if score > best_qwk:
                    best_qwk, stale = score, 0
                    best_weights = copy.deepcopy({k: v.detach().cpu() for k, v in self.net.state_dict().items()})
                else:
                    stale += 1
                if self.on_eval:
                    self.on_eval(len(self.history), score)
                if p["patience"] and stale >= p["patience"]:
                    stop = f"no better dev QWK for {stale} evaluations"
                elif p["max_minutes"] and minutes > p["max_minutes"]:
                    stop = f"time limit of {p['max_minutes']} minutes"
                if stop:
                    break
            if stop:
                break
        if best_weights is not None:
            self.net.load_state_dict(best_weights)
        self.best_dev_qwk = best_qwk
        print(f"  stopped: {stop or 'all epochs done'}; best dev QWK {best_qwk:.4f}", flush=True)

    def parameter_groups(self) -> list[dict]:
        """Return the parameter groups for AdamW: no weight decay on biases and normalisation weights."""
        decay, plain = [], []
        for name, parameter in self.net.named_parameters():
            if parameter.requires_grad:
                (plain if parameter.ndim < 2 or "LayerNorm" in name else decay).append(parameter)
        return [{"params": decay}, {"params": plain, "weight_decay": 0.0}]

    # ---- prediction -------------------------------------------------------------------------

    @torch.no_grad()
    def _proba(self, rows: Sequence[Sequence[int]]) -> np.ndarray:
        """Return the probabilities [n, K] for encoded questions. Similar lengths share a batch."""
        self.net.eval()
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        proba = np.empty((len(rows), self.num_levels))
        size = self.params["eval_batch_size"]
        for start in range(0, len(order), size):
            batch = order[start : start + size]
            input_ids, mask = pad_batch([rows[i] for i in batch], self.pad_id)
            with self._autocast():
                logits = self.net(input_ids.to(self.device), mask.to(self.device))
            proba[batch] = self.net.head.proba(logits).float().cpu().numpy()
        proba = np.clip(proba, 0.0, 1.0)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict_proba(self, items) -> np.ndarray:
        return self._proba(self.encode(items))

    @property
    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.net.parameters()) if self.net is not None else 0

    # ---- save and load ----------------------------------------------------------------------

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        state = {
            "num_levels": self.num_levels,
            "params": self.params,
            "head": self.head_config,
            "history": self.history,
            "extra": self.extra_state(folder),
        }
        (folder / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")
        torch.save({k: v.cpu() for k, v in self.net.state_dict().items()}, folder / WEIGHTS_FILE)

    @classmethod
    def load(cls, folder: Path, device: str | None = None) -> "NeuralModel":
        """Return the saved model, on the CPU unless LEVELLENS_DEVICE or `device` says otherwise."""
        import os

        folder = Path(folder)
        state = json.loads((folder / STATE_FILE).read_text(encoding="utf-8"))
        model = cls(num_levels=state["num_levels"], **state["params"])
        model.device = device or os.environ.get("LEVELLENS_DEVICE", "cpu")
        model.head_config, model.history = state["head"], state["history"]
        model.restore_extra(folder, state["extra"])
        encoder = model.build_encoder()
        head = build_head(OmegaConf.create(state["head"]), encoder.out_features, model.num_levels)
        model.net = Network(encoder, head, model.params["dropout"])
        model.net.load_state_dict(torch.load(folder / WEIGHTS_FILE, map_location="cpu"))
        model.net.to(model.device).eval()
        return model
