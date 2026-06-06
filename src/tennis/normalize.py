"""Answer normalization helpers for tennis temporal QA."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_RANKING_PATTERNS = (
    re.compile(r"\bworld\s+(?:no\.?|number)\s*#?\s*(\d+)\b"),
    re.compile(r"\b(?:no\.?|number)\s*#?\s*(\d+)\b"),
    re.compile(r"#\s*(\d+)\b"),
)
_NUMERIC_MINUTES = re.compile(r"\b(\d+(?:\.\d+)?)\s+minutes?\b")

_APOSTROPHES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u02bc": "'",
    "\uff07": "'",
}
_DASHES = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\ufe58": "-",
    "\ufe63": "-",
    "\uff0d": "-",
}

_PUNCTUATION_TO_SPACE = str.maketrans(
    {char: " " for char in string.punctuation if char not in {"'"}}
)


def _replace_unicode_variants(text: str) -> str:
    for source, target in _APOSTROPHES.items():
        text = text.replace(source, target)
    for source, target in _DASHES.items():
        text = text.replace(source, target)
    return text


def _normalize_rankings(text: str) -> str:
    for pattern in _RANKING_PATTERNS:
        text = pattern.sub(r"\1", text)
    return text


def normalize_tennis_answer(text: str) -> str:
    """Normalize tennis answer strings for deterministic EM/F1 scoring.

    This intentionally keeps name tokens intact apart from case, Unicode width,
    apostrophe, dash, punctuation, and article cleanup. It does not perform
    player aliasing or fuzzy matching.
    """
    if text is None:
        text = ""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = _replace_unicode_variants(normalized)
    normalized = normalized.casefold()
    normalized = _normalize_rankings(normalized)
    normalized = _NUMERIC_MINUTES.sub(r"\1 minute", normalized)

    # Drop apostrophes without splitting names; turn other punctuation into
    # spaces so hyphenated names and score-like strings remain token comparable.
    normalized = normalized.replace("'", "")
    normalized = normalized.translate(_PUNCTUATION_TO_SPACE)
    normalized = _ARTICLES.sub(" ", normalized)
    normalized = _normalize_rankings(normalized)
    normalized = _NUMERIC_MINUTES.sub(r"\1 minute", normalized)

    return " ".join(normalized.split())


def tennis_exact_match(pred: str, gold: str) -> int:
    """Return 1 when normalized prediction and gold answers are identical."""
    return int(normalize_tennis_answer(pred) == normalize_tennis_answer(gold))


def tennis_token_f1(pred: str, gold: str) -> float:
    """Token-level F1 over tennis-normalized answer strings."""
    pred_tokens = normalize_tennis_answer(pred).split()
    gold_tokens = normalize_tennis_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
