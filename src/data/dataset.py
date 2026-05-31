from __future__ import annotations

import json

from datasets import Dataset

TRAIN_KEYS = ("dataset_name", "question_id", "question", "answer", "prompt", "output")
TEST_KEYS = ("dataset_name", "question_id", "question", "prompt", "answer")


def _load_records(path: str) -> list[dict]:
    # Released files are a single JSON array; fall back to JSONL if that ever changes.
    with open(path) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]
    if not isinstance(obj, list):
        raise ValueError(f"{path}: expected a JSON array of records, got {type(obj).__name__}")
    return obj


def load_tiser_train(path: str, subset_size: int | None = None) -> Dataset:
    records = _load_records(path)
    if subset_size is not None:
        records = records[:subset_size]
    # TRL prompt-completion format -> SFTTrainer masks the prompt and trains on the
    # completion only. completion = the gold 4-section trace (record["output"]).
    rows = [{"prompt": r["prompt"], "completion": r["output"]} for r in records]
    return Dataset.from_list(rows)


def load_tiser_test(path: str, max_samples_per_split: int | None = None) -> Dataset:
    records = _load_records(path)
    rows = [{k: r[k] for k in TEST_KEYS} for r in records]
    if max_samples_per_split is not None:
        rows = _cap_per_split(rows, max_samples_per_split)
    return Dataset.from_list(rows)


def _cap_per_split(rows: list[dict], n: int) -> list[dict]:
    counts: dict[str, int] = {}
    kept = []
    for row in rows:
        name = row["dataset_name"]
        if counts.get(name, 0) >= n:
            continue
        counts[name] = counts.get(name, 0) + 1
        kept.append(row)
    return kept


def filter_by_length(ds: Dataset, tokenizer, max_seq_len: int) -> tuple[Dataset, int]:
    def within_limit(example):
        n = len(tokenizer(example["prompt"] + example["completion"])["input_ids"])
        return n <= max_seq_len

    filtered = ds.filter(within_limit)
    return filtered, len(ds) - len(filtered)
