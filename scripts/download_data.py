"""C2: fetch everything a run needs, so that the GPU jobs can work without internet.

    python scripts/download_data.py                  # splits + the two encoders + GloVe (about 2.5 GB)
    python scripts/download_data.py --all            # the same, plus the 8 GB open LLM
    python scripts/download_data.py encoders glove   # only these targets
    python scripts/download_data.py --check          # offline check: load every split and every model
    python scripts/download_data.py --list           # what each target fetches, and where it goes

Run it on a machine with internet. The GPU jobs then read the two folders below and never
download anything:

    LEVELLENS_DATA   the split files written by `python -m levellens.data.prepare`
    HF_HOME          the Hugging Face cache with the model weights

Targets:

| Target     | What                                                              | Size   |
|------------|-------------------------------------------------------------------|--------|
| `datasets` | RACE++, ARC, MATH and ScienceQA splits, checked against the manifest | 1.2 GB |
| `encoders` | DeBERTa-v3-base and DistilBERT, through `transformers`            | 1.0 GB |
| `glove`    | GloVe 6B 300d, the two files `levellens.features.glove` reads      | 0.5 GB |
| `llm`      | Qwen3-4B-Instruct-2507 at the revision pinned in `levellens.models.llm` | 8 GB |

`encoders` loads each model once after the download, so a target that reports "ok" is a model that
`transformers` can really build. `--check` repeats that with `HF_HUB_OFFLINE=1`, which is what a
GPU job without internet does.

To move the cache to another machine, copy the two folders that `--list` prints.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so that the script runs without `pip install -e .`

from levellens.data import ADAPTERS, split_path  # noqa: E402
from levellens.paths import cache_dir, data_dir  # noqa: E402

ENCODERS = ("microsoft/deberta-v3-base", "distilbert/distilbert-base-uncased")
TARGETS = ("datasets", "encoders", "glove", "llm")
DEFAULT_TARGETS = ("datasets", "encoders", "glove")
CHECKED_SPLITS = ("train", "dev", "test", "test_1k")


def hf_home() -> Path:
    """Return the Hugging Face cache folder that the weights are downloaded into."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()


def fetch_datasets() -> None:
    """Download the four datasets, write the split files and check them against the manifest."""
    from levellens.data.prepare import main as prepare

    prepare([])  # all four datasets; raises SystemExit when a split differs from the manifest


def fetch_encoders() -> None:
    """Download DeBERTa-v3-base and DistilBERT, and build each one once to prove the cache is complete."""
    from transformers import AutoModel, AutoTokenizer

    for name in ENCODERS:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        print(f"  {name}: {sum(p.numel() for p in model.parameters()):,} parameters, "
              f"tokeniser {type(tokenizer).__name__}")
        del model


def fetch_glove() -> None:
    """Download the two files of the Hugging Face copy of GloVe 6B 300d."""
    from huggingface_hub import hf_hub_download

    from levellens.features.glove import HUB_REPO, HUB_VOCAB, HUB_WEIGHTS

    if (local := Path(cache_dir() / "glove" / "glove.6B.300d.txt")).exists():
        print(f"  the local GloVe text file is already there: {local}")
        return
    for filename in (HUB_VOCAB, HUB_WEIGHTS):
        print(f"  {hf_hub_download(HUB_REPO, filename)}")


def fetch_llm() -> None:
    """Download the pinned Qwen3-4B snapshot into the model cache."""
    from levellens.models.llm import main as download_llm

    download_llm(["--download"])


FETCH = {"datasets": fetch_datasets, "encoders": fetch_encoders, "glove": fetch_glove, "llm": fetch_llm}


def check() -> int:
    """Load every split and every model with the hub switched off. Return the number of failures."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    failures = 0

    from levellens.data import load_split

    for name in (dataset for dataset in ADAPTERS if dataset != "toy"):
        for split in CHECKED_SPLITS:
            if not split_path(name, split).exists():
                print(f"MISSING  {name}/{split}: no split file. Run the `datasets` target.")
                failures += 1
                continue
            items = load_split(name, split)
            print(f"ok       {name}/{split}: {len(items):,} items")

    from transformers import AutoModel, AutoTokenizer

    for model_name in ENCODERS:
        try:
            AutoTokenizer.from_pretrained(model_name)
            AutoModel.from_pretrained(model_name)
            print(f"ok       {model_name}: loaded offline")
        except Exception as error:  # noqa: BLE001 - the message is what the user needs
            print(f"MISSING  {model_name}: {type(error).__name__}: {error}")
            failures += 1

    try:
        from levellens.features.glove import glove_vectors

        found = glove_vectors({"whale", "triangle"})
        print(f"ok       GloVe: {len(found)} of 2 test words found")
        failures += len(found) != 2
    except Exception as error:  # noqa: BLE001
        print(f"MISSING  GloVe: {type(error).__name__}: {error}")
        failures += 1

    from levellens.models.llm import _snapshot_folder

    if list(_snapshot_folder().glob("*.safetensors")):
        print(f"ok       open LLM: snapshot in {_snapshot_folder()}")
    else:
        print(f"absent   open LLM: no snapshot in {_snapshot_folder()} (only needed for the `llm` target)")
    return failures


def main(argv: list[str] | None = None) -> int:
    """Run the command line. The exit code is the number of failed targets or checks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", metavar="TARGET", default=[],
                        help=f"any of {' '.join(TARGETS)}; default: {' '.join(DEFAULT_TARGETS)}")
    parser.add_argument("--all", action="store_true", help="every target, the open LLM included")
    parser.add_argument("--check", action="store_true", help="offline check only, download nothing")
    parser.add_argument("--list", action="store_true", help="print the targets and the folders, fetch nothing")
    args = parser.parse_args(argv)
    if unknown := [target for target in args.targets if target not in TARGETS]:
        parser.error(f"unknown targets {unknown}; choose from {' '.join(TARGETS)}")

    print(f"LEVELLENS_DATA   {data_dir()}")
    print(f"LEVELLENS_CACHE  {cache_dir()}")
    print(f"HF_HOME          {hf_home()}")
    if args.list:
        print()
        for line in __doc__.splitlines():
            if line.startswith("|"):
                print(line)
        return 0
    if args.check:
        print()
        failures = check()
        print(f"\n{'FAILED: ' + str(failures) + ' missing' if failures else 'All present. A job can run offline.'}")
        return min(failures, 1)

    targets = TARGETS if args.all else (tuple(args.targets) or DEFAULT_TARGETS)
    failures = []
    for target in targets:
        print(f"\n=== {target} ===", flush=True)
        try:
            FETCH[target]()
            print(f"{target}: ok")
        except Exception as error:  # noqa: BLE001 - one broken target must not stop the others
            print(f"{target}: FAILED: {type(error).__name__}: {error}")
            failures.append(target)
    print(f"\nFetched {[t for t in targets if t not in failures]}" + (f", failed {failures}" if failures else ""))
    if not failures:
        print("Now check that a job can run without internet:  python scripts/download_data.py --check")
    return min(len(failures), 1)


if __name__ == "__main__":
    raise SystemExit(main())
