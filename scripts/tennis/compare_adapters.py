"""Aggregate tennis evaluation metrics across model/adapter conditions."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ORDER = {
    "base_qwen": 0,
    "base_qwen_standard": 0,
    "base_qwen_tiser": 1,
    "original_tiser": 2,
    "tennis_only": 3,
    "mixed_replay": 4,
    "mixed_tennis_tiser_replay": 4,
}

DEFAULT_EXPECTED_CONDITIONS = (
    "base_qwen_standard",
    "base_qwen_tiser",
    "original_tiser",
    "tennis_only",
    "mixed_replay",
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = resolve_repo_path(args.results_dir)
    condition_dirs = [
        resolve_repo_path(path) for path in args.condition_dirs
    ] if args.condition_dirs else discover_condition_dirs(
        results_dir / "scored",
        expected_conditions=args.expected_conditions,
    )

    metrics_by_condition = load_condition_metrics(condition_dirs)
    comparison = build_comparison(metrics_by_condition, baseline=args.baseline)

    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else results_dir / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "adapter_comparison.json"
    md_path = output_dir / "adapter_comparison.md"
    csv_path = output_dir / "per_category_comparison.csv"

    write_json(json_path, comparison)
    write_text(md_path, render_markdown(comparison))
    write_per_category_csv(csv_path, comparison["per_category"])

    print(f"[tennis-compare] compared {len(comparison['conditions'])} conditions")
    print(f"[tennis-compare] wrote {json_path}, {md_path}, {csv_path}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tennis adapter evaluation metrics.")
    parser.add_argument(
        "--results-dir",
        default="results/tennis_domain_adaptation",
        help="Root tennis results directory containing scored/ and comparisons/.",
    )
    parser.add_argument(
        "--condition-dirs",
        nargs="*",
        default=None,
        help="Optional explicit condition directories, each containing metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <results-dir>/comparisons.",
    )
    parser.add_argument(
        "--baseline",
        default="original_tiser",
        help="Condition name used for delta columns when present.",
    )
    parser.add_argument(
        "--expected-conditions",
        nargs="*",
        default=list(DEFAULT_EXPECTED_CONDITIONS),
        help=(
            "Condition directory names expected under scored/. Missing metrics "
            "are reported as MISSING instead of raising."
        ),
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def discover_condition_dirs(
    scored_dir: Path, *, expected_conditions: list[str] | tuple[str, ...] | None = None
) -> list[Path]:
    paths_by_name: dict[str, Path] = {}
    if scored_dir.exists():
        paths_by_name.update(
            {path.name: path for path in scored_dir.iterdir() if path.is_dir()}
        )
    for condition in expected_conditions or []:
        paths_by_name.setdefault(condition, scored_dir / condition)
    return sorted(
        paths_by_name.values(),
        key=lambda path: (DEFAULT_ORDER.get(path.name, 100), path.name),
    )


def load_condition_metrics(condition_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for condition_dir in condition_dirs:
        metrics_path = condition_dir / "metrics.json"
        if not metrics_path.exists():
            condition = condition_dir.name
            metrics = missing_metrics(condition_dir, metrics_path)
        else:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            condition = str(metrics.get("condition") or condition_dir.name)
            metrics["status"] = str(metrics.get("status") or "AVAILABLE")
        metrics["_condition_dir"] = str(condition_dir)
        metrics["_metrics_path"] = str(metrics_path)
        loaded[condition] = metrics
    return dict(
        sorted(loaded.items(), key=lambda item: (DEFAULT_ORDER.get(item[0], 100), item[0]))
    )


def missing_metrics(condition_dir: Path, metrics_path: Path) -> dict[str, Any]:
    return {
        "condition": condition_dir.name,
        "status": "MISSING",
        "overall": {},
        "per_category": {},
        "adapter_dir": None,
        "_condition_dir": str(condition_dir),
        "_metrics_path": str(metrics_path),
    }


def build_comparison(
    metrics_by_condition: dict[str, dict[str, Any]],
    *,
    baseline: str,
) -> dict[str, Any]:
    baseline_metrics = metrics_by_condition.get(baseline)
    conditions = []
    per_category = []

    for condition, metrics in metrics_by_condition.items():
        overall = metrics.get("overall", {})
        baseline_overall = baseline_metrics.get("overall", {}) if baseline_metrics else {}
        conditions.append(
            {
                "condition": condition,
                "status": metrics.get("status", "AVAILABLE"),
                "prompt_style": metrics.get("prompt_style"),
                "model_name": metrics.get("model_name"),
                "adapter_dir": metrics.get("adapter_dir"),
                "n": int(overall.get("n", 0)),
                "em": float(overall.get("em", 0.0)),
                "f1": float(overall.get("f1", 0.0)),
                "malformed_count": int(overall.get("malformed_count", 0)),
                "malformed_rate": float(overall.get("malformed_rate", 0.0)),
                "delta_em_vs_baseline": delta(
                    overall.get("em"), baseline_overall.get("em")
                ),
                "delta_f1_vs_baseline": delta(
                    overall.get("f1"), baseline_overall.get("f1")
                ),
                "metrics_path": metrics.get("_metrics_path"),
            }
        )

        for category, values in sorted(metrics.get("per_category", {}).items()):
            baseline_values = (
                baseline_metrics.get("per_category", {}).get(category, {})
                if baseline_metrics
                else {}
            )
            per_category.append(
                {
                    "condition": condition,
                    "category": category,
                    "n": int(values.get("n", 0)),
                    "em": float(values.get("em", 0.0)),
                    "f1": float(values.get("f1", 0.0)),
                    "malformed_count": int(values.get("malformed_count", 0)),
                    "malformed_rate": float(values.get("malformed_rate", 0.0)),
                    "delta_em_vs_baseline": delta(
                        values.get("em"), baseline_values.get("em")
                    ),
                    "delta_f1_vs_baseline": delta(
                        values.get("f1"), baseline_values.get("f1")
                    ),
                }
            )

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "baseline": baseline if baseline_metrics else None,
        "conditions": conditions,
        "per_category": per_category,
    }


def delta(value: Any, baseline_value: Any) -> float | None:
    if value is None or baseline_value is None:
        return None
    return float(value) - float(baseline_value)


def render_markdown(comparison: dict[str, Any]) -> str:
    baseline = comparison.get("baseline")
    lines = [
        "# Tennis Adapter Comparison",
        "",
        f"- Generated at: `{comparison['generated_at']}`",
        f"- Delta baseline: `{baseline}`" if baseline else "- Delta baseline: not available",
        "",
        "## Overall",
        "",
        "| Condition | Prompt | N | EM | F1 | Malformed | Malformed Rate | Delta EM | Delta F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["conditions"]:
        condition = md_code(row["condition"])
        if row.get("status") == "MISSING":
            condition = f"{condition} MISSING"
        lines.append(
            "| {condition} | {prompt} | {n} | {em:.4f} | {f1:.4f} | {bad} | "
            "{bad_rate:.4f} | {delta_em} | {delta_f1} |".format(
                condition=condition,
                prompt=md_code(row.get("prompt_style") or ""),
                n=row["n"],
                em=row["em"],
                f1=row["f1"],
                bad=row["malformed_count"],
                bad_rate=row["malformed_rate"],
                delta_em=format_optional(row["delta_em_vs_baseline"]),
                delta_f1=format_optional(row["delta_f1_vs_baseline"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per Category",
            "",
            "| Condition | Category | N | EM | F1 | Malformed Rate | Delta EM | Delta F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["per_category"]:
        lines.append(
            "| {condition} | {category} | {n} | {em:.4f} | {f1:.4f} | "
            "{bad_rate:.4f} | {delta_em} | {delta_f1} |".format(
                condition=md_code(row["condition"]),
                category=md_code(row["category"]),
                n=row["n"],
                em=row["em"],
                f1=row["f1"],
                bad_rate=row["malformed_rate"],
                delta_em=format_optional(row["delta_em_vs_baseline"]),
                delta_f1=format_optional(row["delta_f1_vs_baseline"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def md_code(value: str) -> str:
    return f"`{value}`"


def format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:+.4f}"


def write_per_category_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "condition",
        "category",
        "n",
        "em",
        "f1",
        "malformed_count",
        "malformed_rate",
        "delta_em_vs_baseline",
        "delta_f1_vs_baseline",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
