from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation, drop articles, collapse whitespace."""
    s = s.lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        # Both empty -> they agree; exactly one empty -> no overlap.
        return float(pred_tokens == gold_tokens)

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def aggregate(records: list[dict]) -> dict:
    """Group per-example {dataset_name, em, f1} by split; macro = unweighted split mean."""
    per_split: dict[str, dict] = {}
    for name in sorted({r["dataset_name"] for r in records}):
        rows = [r for r in records if r["dataset_name"] == name]
        per_split[name] = {
            "em": sum(r["em"] for r in rows) / len(rows),
            "f1": sum(r["f1"] for r in rows) / len(rows),
            "n": len(rows),
        }

    return {
        "per_split": per_split,
        "macro_em": macro_em(per_split),
        "macro_f1": macro_f1(per_split),
        "n_total": len(records),
    }


def macro_em(per_split: dict) -> float:
    return sum(v["em"] for v in per_split.values()) / len(per_split)


def macro_f1(per_split: dict) -> float:
    return sum(v["f1"] for v in per_split.values()) / len(per_split)
