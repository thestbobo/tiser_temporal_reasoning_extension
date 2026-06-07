from __future__ import annotations

import json
from pathlib import Path

from conftest import load_repo_module


CANONICAL_MIXED_ADAPTER = "model/tiser_tennis_mixed_replay_qwen7b/adapter"


def load_compare_module():
    return load_repo_module("scripts/tennis/compare_adapters.py", "test_compare_module")


def write_metrics(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def fake_metrics(condition: str, *, adapter_dir: str | None = None) -> dict:
    return {
        "condition": condition,
        "prompt_style": "tiser",
        "model_name": "fake-model",
        "adapter_dir": adapter_dir,
        "overall": {
            "n": 2,
            "em": 0.5,
            "f1": 0.75,
            "malformed_count": 1,
            "malformed_rate": 0.5,
        },
        "per_category": {
            "yes_no_before_after": {
                "n": 2,
                "em": 0.5,
                "f1": 0.75,
                "malformed_count": 1,
                "malformed_rate": 0.5,
            }
        },
    }


def test_missing_metrics_are_loaded_as_missing(tmp_path: Path) -> None:
    compare = load_compare_module()

    loaded = compare.load_condition_metrics([tmp_path / "scored" / "tennis_only"])

    assert loaded["tennis_only"]["status"] == "MISSING"
    assert loaded["tennis_only"]["overall"] == {}


def test_compare_main_marks_missing_runs_without_crashing(tmp_path: Path) -> None:
    compare = load_compare_module()
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "comparison_out"
    write_metrics(
        results_dir / "scored" / "base_qwen_standard" / "metrics.json",
        fake_metrics("base_qwen_standard"),
    )

    compare.main(["--results-dir", str(results_dir), "--output-dir", str(output_dir)])

    comparison = json.loads((output_dir / "adapter_comparison.json").read_text())
    statuses = {row["condition"]: row["status"] for row in comparison["conditions"]}
    assert statuses["base_qwen_standard"] == "AVAILABLE"
    assert statuses["original_tiser"] == "MISSING"
    assert "MISSING" in (output_dir / "adapter_comparison.md").read_text()


def test_available_metrics_are_loaded_correctly(tmp_path: Path) -> None:
    compare = load_compare_module()
    condition_dir = tmp_path / "scored" / "base_qwen_tiser"
    write_metrics(condition_dir / "metrics.json", fake_metrics("base_qwen_tiser"))

    loaded = compare.load_condition_metrics([condition_dir])
    comparison = compare.build_comparison(loaded, baseline="original_tiser")
    row = comparison["conditions"][0]

    assert row["condition"] == "base_qwen_tiser"
    assert row["status"] == "AVAILABLE"
    assert row["n"] == 2
    assert row["em"] == 0.5
    assert row["f1"] == 0.75


def test_canonical_mixed_replay_adapter_path_is_preserved(tmp_path: Path) -> None:
    compare = load_compare_module()
    condition_dir = tmp_path / "scored" / "mixed_replay"
    write_metrics(
        condition_dir / "metrics.json",
        fake_metrics("mixed_replay", adapter_dir=CANONICAL_MIXED_ADAPTER),
    )

    loaded = compare.load_condition_metrics([condition_dir])
    comparison = compare.build_comparison(loaded, baseline="original_tiser")

    assert comparison["conditions"][0]["condition"] == "mixed_replay"
    assert comparison["conditions"][0]["adapter_dir"] == CANONICAL_MIXED_ADAPTER
