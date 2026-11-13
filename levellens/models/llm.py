"""D6: score the level labels of Qwen3-4B-Instruct-2507 without fine-tuning.

Install CUDA-enabled PyTorch following https://pytorch.org/get-started/locally/,
then install "transformers>=4.57.1,<5" and "accelerate>=1.10,<2". For the
optional CUDA NF4 mode, also install "bitsandbytes>=0.48,<0.50".

Download and transfer the official snapshot (about 8 GB) once:
    python -m levellens.models.llm --download

Then run offline from the repository root, once per dataset (race, arc, math):
    python -m levellens.run data=arc model=qwen_zero_shot owner=D device=cuda "eval_splits=[test_1k]"

On an 8 GB GPU, append:
    model.params.quantization=nf4 variant=nf4

By default the snapshot is read from LEVELLENS_CACHE/share-d/qwen3-4b-instruct-2507
or the existing Hugging Face cache. Set model.params.model_path to a transferred
snapshot, and lower model.params.batch_size if VRAM is tight. Test with:
    python -m pytest tests/test_llm.py -q

Both prompts are zero-shot. A deterministic stratified sample of dev selects
the prompt by QWK. The only formal evaluation split is test_1k. The shared
runner's train_seconds field measures checkpoint loading and prompt selection.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import re
from string import ascii_uppercase

import numpy as np

from levellens.data import info, stratified_subset
from levellens.evaluate import evaluate
from levellens.models.base import Model, texts
from levellens.paths import cache_dir


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_PARAMS = 4_022_468_096  # the pinned official checkpoint, before quantization
PROMPT_VERSION = 1
PROMPTS = {
    "race": (
        "English reading-comprehension question. Estimate its intended school stage.",
        "Consider vocabulary, sentence complexity, and the inference required by the stem and options. "
        "The source passage may be unavailable.",
    ),
    "arc": (
        "U.S. school science question. Estimate its intended grade.",
        "Consider the science concepts, how much prior knowledge is needed, and the steps required.",
    ),
    "math": (
        "Competition mathematics problem. Estimate its intended difficulty level, from 1 to 5.",
        "Consider the mathematical techniques and number of reasoning steps likely required.",
    ),
    "toy": (
        "Practice question. Estimate its intended difficulty.",
        "Consider vocabulary, concepts, and the steps required.",
    ),
}


def _snapshot_folder() -> Path:
    return cache_dir() / "share-d" / "qwen3-4b-instruct-2507"


def render_prompt(tokenizer, text, dataset, level_names, variant):
    task, rubric = PROMPTS[dataset]
    if variant == "rubric":
        task += " " + rubric
    elif variant != "direct":
        raise ValueError(f"Unknown D6 prompt variant: {variant!r}")
    labels = "\n".join(f"{letter}: {name}" for letter, name in zip(ascii_uppercase, level_names))
    # Prevent a literal Qwen control token in a question from changing chat roles.
    text = re.sub(r"<\|[^>\n]*\|>", lambda match: match.group().replace("<", "‹").replace(">", "›"), text)
    prompt = (
        f"{task}\n\nLevels, ordered from lowest to highest:\n{labels}\n\n"
        f"Question text:\n{text}\n\n"
        "Which listed level best matches this question? Reply with only the corresponding "
        "uppercase label letter."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )


def single_token_labels(tokenizer, prompt, labels):
    """Find distinct one-token continuations in context, rejecting prefix merges."""
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = []
    for label in labels:
        extended = tokenizer.encode(prompt + label, add_special_tokens=False)
        if len(extended) != len(prefix) + 1 or extended[:-1] != prefix:
            raise ValueError(f"Label {label!r} is not a single token after the assistant prefix")
        ids.append(extended[-1])
    if len(ids) != len(set(ids)):
        raise ValueError("Level labels must have distinct token IDs")
    return ids


class QwenZeroShot(Model):
    """Use the next-token probabilities conditional on the allowed level letters."""

    @property
    def n_params(self):
        return MODEL_PARAMS

    def fit(self, train, dev, config):
        if list(config.eval_splits) != ["test_1k"]:
            raise ValueError('D6 must run with "eval_splits=[test_1k]"; prompt selection uses dev internally')
        if config.save_model:
            raise ValueError("D6 uses an existing checkpoint; run with save_model=false")
        self.dataset_ = str(config.data.name)
        if self.dataset_ not in PROMPTS:
            raise ValueError(f"D6 has no prompt for {self.dataset_!r}")
        self.level_names_ = info(self.dataset_).level_names
        if not 2 <= self.num_levels <= len(ascii_uppercase) or len(self.level_names_) != self.num_levels:
            raise ValueError("D6 needs between 2 and 26 named levels")
        sample_size = int(self.params["dev_sample_size"])
        if sample_size <= 0 or not len(dev):
            raise ValueError("D6 prompt selection requires a non-empty validation sample")
        seed = int(config.seed)
        selected_dev = stratified_subset(dev, sample_size, seed=seed)
        self._load_checkpoint(config)
        scores = {}
        for variant in ("direct", "rubric"):
            proba = self._score(texts(selected_dev), variant)
            scores[variant] = evaluate([item["level"] for item in selected_dev], proba, self.num_levels)["qwk"]
        self.prompt_variant_ = max(scores, key=lambda key: scores[key] if scores[key] is not None else -np.inf)
        config.model.prompt_selection = {
            "version": PROMPT_VERSION, "split": "dev", "metric": "qwk", "sample_size": len(selected_dev),
            "seed": seed, "scores": scores, "selected": self.prompt_variant_,
        }
        print(f"{self.dataset_}: selected {self.prompt_variant_!r} on {len(selected_dev)} dev questions", flush=True)

    def _load_checkpoint(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        device = str(config.device)
        quantization = self.params["quantization"]
        if quantization not in {"none", "nf4"}:
            raise ValueError("quantization must be none or nf4")
        if quantization == "nf4" and not device.startswith("cuda"):
            raise ValueError("NF4 requires a CUDA device; use quantization=none for CPU")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in PyTorch; install its CUDA-enabled build")
        dtype = torch.float32
        if device.startswith("cuda"):
            with torch.cuda.device(device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        folder = self.params.get("model_path") or (_snapshot_folder() if _snapshot_folder().is_dir() else None)
        source = str(folder) if folder else MODEL_ID
        if folder and not Path(folder).is_dir():
            raise FileNotFoundError(f"D6 snapshot folder does not exist: {folder}")
        options = {"local_files_only": True, "trust_remote_code": False}
        if not folder:
            options["revision"] = MODEL_REVISION
        self.tokenizer_ = AutoTokenizer.from_pretrained(source, **options)
        self.tokenizer_.padding_side = "left"
        if self.tokenizer_.pad_token_id is None:
            raise ValueError("The Qwen tokenizer must define a padding token")
        options.update(dtype=dtype, device_map={"": device}, use_safetensors=True, attn_implementation="sdpa")
        if quantization == "nf4":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            )
        self.model_ = AutoModelForCausalLM.from_pretrained(source, **options).eval()
        if self.model_.config.model_type != "qwen3":
            raise ValueError("D6 expects a Qwen3 checkpoint")
        config.model.runtime = {
            "checkpoint": MODEL_ID, "revision": MODEL_REVISION, "dtype": str(dtype),
            "quantization": quantization, "torch": str(torch.__version__), "transformers": version("transformers"),
        }
        if quantization == "nf4":
            config.model.runtime.bitsandbytes = version("bitsandbytes")

    def _score(self, inputs, variant):
        import torch

        batch_size = int(self.params["batch_size"])
        max_tokens = int(self.params["max_prompt_tokens"])
        if batch_size < 1 or max_tokens < 1:
            raise ValueError("batch_size and max_prompt_tokens must be positive")
        proba = np.empty((len(inputs), self.num_levels), dtype=np.float32)
        if not len(inputs):
            return proba
        prompts = [
            render_prompt(self.tokenizer_, text, self.dataset_, self.level_names_, variant) for text in inputs
        ]
        label_ids = single_token_labels(self.tokenizer_, prompts[0], ascii_uppercase[: self.num_levels])
        columns = torch.tensor(label_ids, device=self.model_.device)
        order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch = self.tokenizer_(
                    [prompts[index] for index in indices],
                    add_special_tokens=False, padding=True, return_tensors="pt",
                )
                if batch["input_ids"].shape[1] > max_tokens:
                    raise ValueError(f"A D6 prompt exceeds {max_tokens} tokens; increase max_prompt_tokens")
                logits = self.model_(**batch.to(self.model_.device), use_cache=False, logits_to_keep=1).logits
                # The left-padded batch ends in a real assistant prefix for every row.
                chosen = logits[:, -1].index_select(1, columns).float()
                proba[indices] = torch.softmax(chosen, dim=1).cpu().numpy()
        if not np.isfinite(proba).all():
            raise ValueError("Qwen produced non-finite label probabilities")
        return proba

    def predict_proba(self, items):
        return self._score(texts(items), self.prompt_variant_)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true", required=True, help="download the pinned official weights")
    parser.parse_args(argv)
    from huggingface_hub import snapshot_download

    print(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_dir=_snapshot_folder(), token=False,
        allow_patterns=["*.json", "*.txt", "*.safetensors"], max_workers=3,
    ))


if __name__ == "__main__":
    main()
