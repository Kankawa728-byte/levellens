"""Word tokens and GloVe vectors for the BiLSTM.

    words = tokenize("What is $2^3$? A. 6 B. 8")      # ['what', 'is', '$', '2', '^', '3', '$', '?', ...]
    vectors = glove_vectors({"what", "is", "zzzz"})   # {'what': array[300], 'is': array[300]}

GloVe 6B 300d (Pennington, Socher and Manning, 2014; Wikipedia + Gigaword, 400,000 lower-case words,
Public Domain Dedication and License). Two sources, tried in this order:

1. `<LEVELLENS_CACHE>/glove/glove.6B.300d.txt`, the text file from nlp.stanford.edu/projects/glove,
   if someone has put it there (for a machine without internet).
2. The same vectors from the Hugging Face hub, `sentence-transformers/average_word_embeddings_glove.6B.300d`
   (about 480 MB, downloaded once into the Hugging Face cache).

Only the vectors of the words that are asked for are returned, so a model keeps a matrix for the
words of its own training set and not for all 400,000.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from levellens.paths import cache_dir

DIM = 300
HUB_REPO = "sentence-transformers/average_word_embeddings_glove.6B.300d"
HUB_VOCAB = "0_WordEmbeddings/whitespacetokenizer_config.json"
HUB_WEIGHTS = "0_WordEmbeddings/model.safetensors"
TEXT_FILE = "glove.6B.300d.txt"

# a run of letters (with inner apostrophes), a run of digits, or one other visible character
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*|\d+|[^\sa-z\d]")


def tokenize(text: str) -> list[str]:
    """Split a question into lower-case words, numbers and single symbols, as GloVe 6B expects."""
    return TOKEN.findall(text.lower())


def text_file() -> Path:
    """Return where the optional local GloVe text file is looked for."""
    return cache_dir() / "glove" / TEXT_FILE


def _from_text(path: Path, wanted: set[str]) -> dict[str, np.ndarray]:
    """Read the vectors of the wanted words from the GloVe text file: one word and 300 numbers per line."""
    found = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            word, _, rest = line.partition(" ")
            if word in wanted:
                found[word] = np.asarray(rest.split(), dtype=np.float32)
    return found


def _from_hub(wanted: set[str]) -> dict[str, np.ndarray]:
    """Read the vectors of the wanted words from the Hugging Face copy of GloVe 6B 300d."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    vocab = json.loads(Path(hf_hub_download(HUB_REPO, HUB_VOCAB)).read_text(encoding="utf-8"))["vocab"]
    with safe_open(hf_hub_download(HUB_REPO, HUB_WEIGHTS), framework="np") as weights:
        matrix = weights.get_tensor(next(iter(weights.keys())))  # [400000, 300]
    return {word: matrix[row] for row, word in enumerate(vocab) if word in wanted}


def glove_vectors(words: Iterable[str]) -> dict[str, np.ndarray]:
    """Return the 300-d GloVe vector of every word that GloVe knows. Unknown words are left out."""
    wanted = set(words)
    path = text_file()
    found = _from_text(path, wanted) if path.exists() else _from_hub(wanted)
    return {word: np.asarray(vector, dtype=np.float32) for word, vector in found.items()}
