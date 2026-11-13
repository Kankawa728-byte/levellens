"""Fine-tuning wrapper: any Hugging Face encoder with any output head.

    python -m levellens.run data=arc model=deberta-v3 head=corn seed=0
    python -m levellens.run data=toy model=distilbert head=ce model.params.backbone=<any encoder on the hub>

    text -> tokeniser -> transformer encoder -> vector of the first token -> head

What this file adds to the shared loop in `levellens/models/neural.py`:
- Any backbone: `params.backbone` is a model name on the Hugging Face hub. The head is ours
  (`levellens/heads/`), so cross-entropy and CORN sit on exactly the same encoder.
- Tokenisation cache: the token ids of a split are stored in `<LEVELLENS_CACHE>/tokens/`, keyed by the
  backbone, the length limit and the texts. A second run on the same split (another head, another
  seed, every Optuna trial) does not tokenise again.
- Dynamic padding: questions are tokenised without padding; each batch is padded to its own longest question.
- The 1 GPU-hour limit of specification R6 is `params.max_minutes` (default 50, which leaves room for scoring).
- A saved model holds the weights, the tokeniser and the encoder config, so `Predictor.load` needs no internet.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from levellens.models.base import texts
from levellens.models.neural import NeuralModel
from levellens.paths import cache_dir

BACKBONE_FOLDER = "backbone"  # inside a saved model: tokeniser files and the encoder config
CACHE_FROM = 64  # splits smaller than this are not worth a cache file (single questions typed into the app)


class TransformerEncoder(nn.Module):
    """A Hugging Face encoder. Returns the vector of the first token ([CLS]) or the mean over the tokens."""

    def __init__(self, backbone: nn.Module, pooling: str):
        """Wrap a Hugging Face encoder. `pooling` is `first` or `mean`."""
        super().__init__()
        if pooling not in ("first", "mean"):
            raise ValueError(f"pooling={pooling!r}: use first or mean")
        self.backbone = backbone
        self.pooling = pooling
        self.out_features = backbone.config.hidden_size

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return one vector per question: the state of the first token, or the mean over the real tokens."""
        states = self.backbone(input_ids=input_ids, attention_mask=mask.long()).last_hidden_state  # [n, T, d]
        if self.pooling == "first":
            return states[:, 0]
        weights = mask.unsqueeze(-1).to(states.dtype)
        return (states * weights).sum(dim=1) / weights.sum(dim=1)


class Transformer(NeuralModel):
    """A fine-tuned transformer encoder. Each backbone has a file in `configs/model/`."""

    DEFAULTS = {
        "backbone": "distilbert/distilbert-base-uncased",
        "max_length": 256,       # tokens; longer questions are cut
        "pooling": "first",      # first ([CLS]) or mean
        "token_cache": True,
        "batch_size": 16,
        "eval_batch_size": 64,
    }

    def prepare(self, train, config) -> None:
        """Load the tokeniser of the backbone."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.params["backbone"])
        self.pad_id = self.tokenizer.pad_token_id
        self._pretrained = True  # build_encoder loads the pretrained weights

    def _cache_file(self, strings: list[str]) -> Path:
        """Return the cache file for these texts under this backbone and length limit."""
        digest = hashlib.sha1()
        digest.update(f"{self.params['backbone']}|{self.params['max_length']}|{len(strings)}".encode())
        for string in strings:
            digest.update(hashlib.sha1(string.encode("utf-8")).digest())
        return cache_dir() / "tokens" / f"{digest.hexdigest()}.pt"

    def encode(self, items) -> list[list[int]]:
        """Return the token ids of each item, cut to `max_length`. Large splits are cached on disk."""
        strings = texts(items)
        use_cache = self.params["token_cache"] and len(strings) >= CACHE_FROM
        path = self._cache_file(strings) if use_cache else None
        if path is not None and path.exists():
            return torch.load(path)
        rows = self.tokenizer(strings, truncation=True, max_length=self.params["max_length"], padding=False)["input_ids"]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(rows, path)
        return rows

    def build_encoder(self) -> TransformerEncoder:
        """Return the encoder: with the pretrained weights before training, with the architecture only after `load`."""
        from transformers import AutoConfig, AutoModel

        if getattr(self, "_pretrained", False):
            backbone = AutoModel.from_pretrained(self.params["backbone"])
        else:  # after load(): only the architecture; the fine-tuned weights are loaded right after
            backbone = AutoModel.from_config(AutoConfig.from_pretrained(self._folder / BACKBONE_FOLDER))
        # .float(): the DeBERTa-v3 weights on the hub are stored in half precision, and we train in fp32 or bf16
        return TransformerEncoder(backbone.float(), self.params["pooling"])

    def extra_state(self, folder: Path) -> dict:
        """Write the tokeniser and the encoder config into `folder`."""
        self.tokenizer.save_pretrained(folder / BACKBONE_FOLDER)
        self.net.encoder.backbone.config.save_pretrained(folder / BACKBONE_FOLDER)
        return {"pad_id": self.pad_id}

    def restore_extra(self, folder: Path, state: dict) -> None:
        """Load the tokeniser from `folder` and remember where the encoder config is."""
        from transformers import AutoTokenizer

        self._folder = folder
        self.tokenizer = AutoTokenizer.from_pretrained(folder / BACKBONE_FOLDER)
        self.pad_id = state["pad_id"]
