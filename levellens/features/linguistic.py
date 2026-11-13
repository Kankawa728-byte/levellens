"""English surface features for Share D's LightGBM and Share F's demo.

Install the optional feature dependencies with
    python -m pip install "cmudict>=1,<2" "pyphen>=0.17,<1" "wordfreq>=3.1,<4"

Use extract_features(text) for a single question and feature_matrix(texts)
for batches. The text should be the same question-and-options input used by
the shared runner. Both functions use the fixed feature order below.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Sequence

import cmudict
import numpy as np
import pyphen
from wordfreq import zipf_frequency

from levellens.paths import cache_dir


FEATURE_GROUPS = {
    "language": ("flesch_reading_ease", "mean_word_rarity", "word_count", "character_count", "sentence_count"),
    "reasoning": ("number_count", "negation_count", "latex_operator_count"),
}
FEATURE_NAMES = tuple(name for names in FEATURE_GROUPS.values() for name in names)
FEATURE_VERSION = 1  # bump when a feature definition changes

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)*")
_SENTENCE_END = re.compile(r"\.(?!\d)|[!?]+")
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_OPTION_MARKER = re.compile(r"(?<=\s)[A-E]\.\s+")
_LATEX_SPAN = re.compile(
    r"(?<!\\)(?:\$\$.*?(?<!\\)\$\$|\$(?!\$)[^$\n]*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\])",
    re.DOTALL,
)
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_OPERATORS = frozenset(
    "frac dfrac tfrac sqrt cdot times div pm mp neq ne le leq ge geq approx sim equiv "
    "sum prod int iint oint lim log ln sin cos tan mod bmod pmod binom cap cup "
    "in notin subset subseteq supset supseteq land lor".split()
)
_MATH_SYMBOL = re.compile(r"[+\-−=<>^*/×÷≤≥≠±]")
_NEGATIONS = frozenset("no not never neither nor none nothing nobody nowhere cannot without".split())


@lru_cache(maxsize=1)
def _pronunciation_tools():
    return cmudict.dict(), pyphen.Pyphen(lang="en_US")


@lru_cache(maxsize=65_536)
def _word_stats(word: str) -> tuple[int, float]:
    pronunciations, hyphenator = _pronunciation_tools()
    entries = pronunciations.get(word)
    syllables = sum(phone[-1].isdigit() for phone in entries[0]) if entries else 1 + len(hyphenator.positions(word))
    rarity = max(0.0, 7.0 - zipf_frequency(word, "en"))
    return syllables, rarity


def extract_features(text: str) -> dict[str, float]:
    """Return Flesch ease, lexical rarity, prose lengths, and three cue counts.

    Flesch uses CMUdict syllable counts with a hyphenation fallback. Prose
    statistics exclude delimited LaTeX, while numbers are counted in the whole
    input. LaTeX cues count known commands plus arithmetic symbols inside
    math delimiters. Negation counts are lexical and do not resolve scope.
    """
    formulas = _LATEX_SPAN.findall(text)
    prose = _LATEX_SPAN.sub(" ", text)
    prose = _LATEX_COMMAND.sub(" ", prose)
    prose = " ".join(_OPTION_MARKER.sub(" ", prose).split()).replace("’", "'").lower()
    words = _WORD.findall(prose)
    word_count = len(words)
    sentence_count = sum(bool(_WORD.search(part)) for part in _SENTENCE_END.split(prose))
    ease = rarity = 0.0
    if words:
        syllables, rarities = zip(*(_word_stats(word) for word in words))
        ease = 206.835 - 1.015 * word_count / sentence_count - 84.6 * sum(syllables) / word_count
        rarity = sum(rarities) / word_count
    latex_count = sum(command[1:] in _OPERATORS for command in _LATEX_COMMAND.findall(text))
    latex_count += sum(len(_MATH_SYMBOL.findall(formula)) for formula in formulas)
    return {
        "flesch_reading_ease": ease,
        "mean_word_rarity": rarity,
        "word_count": word_count,
        "character_count": len(prose),
        "sentence_count": sentence_count,
        "number_count": len(_NUMBER.findall(text)),
        "negation_count": sum(word in _NEGATIONS or word.endswith("n't") for word in words),
        "latex_operator_count": latex_count,
    }


def _cache_path(texts: Sequence[str]) -> Path:
    dependencies = ",".join(version(name) for name in ("cmudict", "pyphen", "wordfreq"))
    digest = sha256(f"{FEATURE_VERSION}|{FEATURE_NAMES}|{dependencies}".encode("utf-8"))
    for text in texts:
        data = text.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    path = cache_dir() / "share-d" / "linguistic" / f"{digest.hexdigest()}.npy"
    if os.name == "nt":
        location = str(path)
        path = Path(r"\\?\UNC" + location[1:] if location.startswith("\\\\") else "\\\\?\\" + location)
    return path


def feature_matrix(texts: Sequence[str], *, use_cache: bool = False) -> np.ndarray:
    """Return rows in input order, optionally reusing a content-keyed disk cache."""
    shape = (len(texts), len(FEATURE_NAMES))
    if not len(texts):
        return np.empty(shape, dtype=np.float32)
    path = _cache_path(texts) if use_cache else None
    if path is not None and path.is_file():
        try:
            matrix = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            pass
        else:
            if matrix.shape == shape and matrix.dtype == np.float32 and np.isfinite(matrix).all():
                return matrix

    rows = (extract_features(text) for text in texts)
    matrix = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
            pending = Path(handle.name)
        try:
            np.save(pending, matrix, allow_pickle=False)
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    return matrix
