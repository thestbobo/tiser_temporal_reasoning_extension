from __future__ import annotations

import os

from src.data.dataset import load_tiser_test
from src.eval.metrics import aggregate, exact_match, token_f1
from src.inference.generate import generate_batch
from src.inference.parser import parse_answer
from src.model.loader import load_adapter_for_inference
from src.utils.io import ensure_dir, write_json, write_run_meta
from src.utils.seeding import set_seed


def _default_adapter_dir(cfg) -> str:
    return os.path.join(cfg.paths.model_dir, cfg.run_name, "adapter")


def run_eval(cfg, adapter_dir: str | None = None) -> dict:
    set_seed(cfg.seed)
    adapter_dir = adapter_dir or _default_adapter_dir(cfg)
    run_dir = os.path.join(cfg.paths.output_dir, cfg.run_name)
    write_run_meta(run_dir, cfg)

    model, tokenizer = load_adapter_for_inference(cfg, adapter_dir)

    test_ds = load_tiser_test(cfg.paths.test_file, cfg.eval.max_samples_per_split)
    _warn_on_unexpected_splits(test_ds["dataset_name"], cfg.splits)

    generations = generate_batch(model, tokenizer, list(test_ds["prompt"]), cfg.eval)

    records = []
    n_malformed = 0
    for row, raw in zip(test_ds, generations):
        parsed = parse_answer(raw)
        n_malformed += int(parsed.malformed)
        records.append(
            {
                "question_id": row["question_id"],
                "dataset_name": row["dataset_name"],
                "gold": row["answer"],
                "raw_generation": raw,
                "pred_answer": parsed.answer,
                "malformed": parsed.malformed,
                "em": exact_match(parsed.answer, row["answer"]),
                "f1": token_f1(parsed.answer, row["answer"]),
            }
        )

    metrics = aggregate(records, cfg.splits)
    metrics["n_malformed"] = n_malformed
    metrics["config_name"] = cfg.run_name
    metrics["adapter_dir"] = adapter_dir

    _write_predictions(os.path.join(run_dir, "predictions.jsonl"), records)
    write_json(os.path.join(run_dir, "metrics.json"), metrics)

    print(f"[eval] macro-EM {metrics['macro_em']:.3f} / macro-F1 {metrics['macro_f1']:.3f}")
    print(f"[eval] malformed outputs: {n_malformed}/{metrics['n_total']}")
    return metrics


def _warn_on_unexpected_splits(names, expected: list[str]) -> None:
    found = set(names)
    unexpected = found - set(expected)
    missing = set(expected) - found
    if unexpected:
        print(f"[eval] WARNING: dataset_name values not in config splits: {sorted(unexpected)}")
    if missing:
        print(f"[eval] WARNING: expected splits absent from test data: {sorted(missing)}")


def _write_predictions(path: str, records: list[dict]) -> None:
    import json

    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
