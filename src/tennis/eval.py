"""Tennis-specific prediction scoring utilities."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from src.inference.parser import parse_answer
    from src.tennis.normalize import (
        normalize_tennis_answer,
        tennis_exact_match,
        tennis_token_f1,
    )
except ModuleNotFoundError:  # pragma: no cover - supports scripts that add src/ to sys.path.
    from inference.parser import parse_answer
    from tennis.normalize import (
        normalize_tennis_answer,
        tennis_exact_match,
        tennis_token_f1,
    )


def load_predictions_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON prediction rows."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
            rows.append(row)
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _gold_answer(row: dict[str, Any], row_index: int) -> str:
    if "gold" in row:
        return "" if row["gold"] is None else str(row["gold"])
    if "answer" in row:
        return "" if row["answer"] is None else str(row["answer"])
    raise ValueError(
        f"Prediction row {row_index} must contain either 'gold' or 'answer'."
    )


def is_yes_no_category(category: Any = None, tags: Iterable[Any] | None = None) -> bool:
    """Return True for tennis categories that should collapse answers to Yes/No."""
    values = []
    if category is not None:
        values.append(str(category))
    if tags is not None:
        values.extend(str(tag) for tag in tags if tag is not None)
    return any("yes_no" in value.casefold() for value in values)


def extract_tennis_answer(
    raw_generation: Any,
    *,
    category: Any = None,
    tags: Iterable[Any] | None = None,
    gold: Any = None,
) -> tuple[str, bool]:
    """Extract a usable tennis answer from strict TISER or instruct-style output."""
    text = "" if raw_generation is None else str(raw_generation)
    if not text.strip():
        return "", True

    yes_no = is_yes_no_category(category, tags)
    if not yes_no and normalize_tennis_answer("" if gold is None else str(gold)) in {"yes", "no"}:
        yes_no = True

    parsed = parse_answer(text)
    if parsed.answer:
        answer = _finalize_extracted_answer(parsed.answer, yes_no=yes_no)
        if answer:
            return answer, False

    for candidate in _instruct_answer_candidates(text):
        answer = _finalize_extracted_answer(candidate, yes_no=yes_no)
        if answer:
            return answer, False

    if yes_no:
        answer = _extract_last_yes_no(text)
        if answer:
            return answer, False

    return "", True


def _instruct_answer_candidates(text: str) -> list[str]:
    """Return fallback answer spans from common instruct model phrasings."""
    patterns = (
        r"(?:^|\n)\s*(?:[*_`]+\s*)?final\s+answer\s*(?:[:\-]\s*|\bis\s+)(.+)",
        r"(?:^|\n)\s*(?:[*_`]+\s*)?answer\s*[:\-]\s*(.+)",
        r"(?<!final\s)\b(?:therefore|thus|so|hence|in conclusion|given this)?\s*,?\s*(?:the\s+)?answer\s+is\s+(.+)",
    )
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(
            match.group(1)
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        )
    return list(reversed(candidates))


def _finalize_extracted_answer(candidate: str, *, yes_no: bool) -> str:
    cleaned = _clean_answer_candidate(candidate)
    if not cleaned:
        return ""
    if yes_no:
        match = re.search(r"\b(yes|no)\b", cleaned, flags=re.IGNORECASE)
        if not match:
            return ""
        return "Yes" if match.group(1).casefold() == "yes" else "No"
    return cleaned


def _clean_answer_candidate(candidate: str) -> str:
    value = str(candidate).strip()
    value = re.sub(r"</?answer>", " ", value, flags=re.IGNORECASE)
    value = value.splitlines()[0].strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(
        r"^(?:[*_`]+\s*)?(?:final\s+answer|answer)\s*(?:[:\-]\s*|\bis\s+)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^(?:therefore|thus|so|hence|in conclusion|given this)\s*,?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:the\s+)?answer\s+is\s+", "", value, flags=re.IGNORECASE)
    value = value.strip(" \t\r\n\"'`*_")
    value = re.sub(r"[*_`]+$", "", value).strip()
    value = value.rstrip(".!?:;").strip()
    return value


def _extract_last_yes_no(text: str) -> str:
    tail = text[-1200:]
    matches = list(re.finditer(r"\b(yes|no)\b", tail, flags=re.IGNORECASE))
    if not matches:
        return ""
    value = matches[-1].group(1).casefold()
    return "Yes" if value == "yes" else "No"


def _prediction_answer_and_malformed(row: dict[str, Any]) -> tuple[str, bool]:
    has_pred_answer = "pred_answer" in row and row["pred_answer"] is not None
    has_raw_generation = "raw_generation" in row and row["raw_generation"] is not None
    has_malformed = "malformed" in row and row["malformed"] is not None

    category = row.get("category")
    tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    gold = row.get("gold", row.get("answer"))
    parsed_malformed = True
    parsed_answer = ""
    if has_raw_generation:
        parsed_answer, parsed_malformed = extract_tennis_answer(
            row["raw_generation"],
            category=category,
            tags=tags,
            gold=gold,
        )

    if has_pred_answer:
        pred_answer = str(row["pred_answer"])
        should_reparse = has_raw_generation and (
            not pred_answer.strip() or (has_malformed and bool(row["malformed"]))
        )
        if should_reparse:
            return parsed_answer, parsed_malformed
        cleaned_answer = _finalize_extracted_answer(
            pred_answer,
            yes_no=is_yes_no_category(category, tags),
        )
        if cleaned_answer:
            malformed = bool(row["malformed"]) if has_malformed else False
            return cleaned_answer, malformed
        malformed = bool(row["malformed"]) if has_malformed else parsed_malformed
        return pred_answer, malformed
    if has_raw_generation:
        return parsed_answer, parsed_malformed
    return "", True


def infer_answer_type(answer: str) -> str:
    """Infer a coarse answer type from a normalized answer string."""
    normalized = normalize_tennis_answer(answer)
    if not normalized:
        return "empty"
    if normalized in {"yes", "no"}:
        return "yes_no"
    if normalized == "1":
        return "ranking_1"
    if re.match(r"^\d+(?:\.\d+)? minute$", normalized):
        return "duration_minutes"
    if re.match(r"^\d+$", normalized):
        return "number"
    if normalized in {
        "final",
        "semifinal",
        "semi final",
        "quarterfinal",
        "quarter final",
        "round of 16",
        "first round",
        "second round",
        "third round",
        "fourth round",
    }:
        return "tournament_round"
    return "span"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    em_values = [float(row["em"]) for row in rows]
    f1_values = [float(row["f1"]) for row in rows]
    malformed_values = [1.0 if row["malformed"] else 0.0 for row in rows]
    return {
        "n": len(rows),
        "em": _mean(em_values),
        "f1": _mean(f1_values),
        "malformed_count": int(sum(malformed_values)),
        "malformed_rate": _mean(malformed_values),
    }


def score_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score already-loaded prediction rows with tennis normalization."""
    scored_rows: list[dict[str, Any]] = []
    answer_type_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    category_answer_type_confusion: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for index, row in enumerate(rows, start=1):
        gold = _gold_answer(row, index)
        pred_answer, malformed = _prediction_answer_and_malformed(row)
        category = str(row.get("category") or "unknown")
        question_id = str(row.get("question_id") or index)
        em = tennis_exact_match(pred_answer, gold)
        f1 = tennis_token_f1(pred_answer, gold)
        gold_type = infer_answer_type(gold)
        pred_type = infer_answer_type(pred_answer)

        answer_type_confusion[gold_type][pred_type] += 1
        category_answer_type_confusion[category][gold_type][pred_type] += 1

        scored_rows.append(
            {
                "question_id": question_id,
                "category": category,
                "gold": gold,
                "pred_answer": pred_answer,
                "normalized_gold": normalize_tennis_answer(gold),
                "normalized_pred": normalize_tennis_answer(pred_answer),
                "gold_answer_type": gold_type,
                "pred_answer_type": pred_type,
                "malformed": malformed,
                "em": em,
                "f1": f1,
            }
        )

    per_category = {
        category: _aggregate([row for row in scored_rows if row["category"] == category])
        for category in sorted({row["category"] for row in scored_rows})
    }
    malformed_examples = [
        {
            "question_id": row["question_id"],
            "category": row["category"],
            "gold": row["gold"],
            "pred_answer": row["pred_answer"],
        }
        for row in scored_rows
        if row["malformed"]
    ][:25]

    return {
        "overall": _aggregate(scored_rows),
        "per_category": per_category,
        "answer_type_confusion": _counter_matrix_to_dict(answer_type_confusion),
        "category_answer_type_confusion": _nested_counter_matrix_to_dict(
            category_answer_type_confusion
        ),
        "malformed_examples": malformed_examples,
        "normalization": {
            "function": "src.tennis.normalize.normalize_tennis_answer",
            "notes": [
                "lowercase with NFKC Unicode normalization",
                "normalize curly apostrophes and dash variants",
                "normalize No. 1 / no 1 / #1 / number 1 ranking answers to 1",
                "normalize numeric minute/minutes durations to singular minute",
                "remove punctuation and English articles, then collapse whitespace",
            ],
        },
    }


def _counter_matrix_to_dict(matrix: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        row_key: {col_key: int(counter[col_key]) for col_key in sorted(counter)}
        for row_key, counter in sorted(matrix.items())
    }


def _nested_counter_matrix_to_dict(
    matrix: dict[str, dict[str, Counter[str]]]
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        category: _counter_matrix_to_dict(inner)
        for category, inner in sorted(matrix.items())
    }


def score_predictions_file(predictions_path: str | Path) -> dict[str, Any]:
    return score_prediction_rows(load_predictions_jsonl(predictions_path))


def render_markdown_report(metrics: dict[str, Any], predictions_path: str | Path | None = None) -> str:
    overall = metrics["overall"]
    lines = [
        "# Tennis Evaluation Metrics",
        "",
        "## Summary",
        "",
    ]
    if predictions_path is not None:
        lines.append(f"- Predictions: `{predictions_path}`")
    lines.extend(
        [
            f"- Total examples: {overall['n']}",
            f"- Exact Match: {overall['em']:.4f}",
            f"- Token F1: {overall['f1']:.4f}",
            f"- Malformed outputs: {overall['malformed_count']} ({overall['malformed_rate']:.4f})",
            "",
            "## Per-Category Metrics",
            "",
            "| Category | N | EM | F1 | Malformed | Malformed Rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, values in metrics["per_category"].items():
        lines.append(
            f"| `{category}` | {values['n']} | {values['em']:.4f} | "
            f"{values['f1']:.4f} | {values['malformed_count']} | "
            f"{values['malformed_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Answer-Type Confusion",
            "",
            "| Gold Answer Type | Predicted Answer Type | Count |",
            "| --- | --- | ---: |",
        ]
    )
    for gold_type, pred_counts in metrics["answer_type_confusion"].items():
        for pred_type, count in pred_counts.items():
            lines.append(f"| `{gold_type}` | `{pred_type}` | {count} |")

    lines.extend(
        [
            "",
            "## Category Answer-Type Confusion",
            "",
            "| Category | Gold Answer Type | Predicted Answer Type | Count |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for category, gold_matrix in metrics["category_answer_type_confusion"].items():
        for gold_type, pred_counts in gold_matrix.items():
            for pred_type, count in pred_counts.items():
                lines.append(f"| `{category}` | `{gold_type}` | `{pred_type}` | {count} |")

    lines.extend(["", "## Malformed Examples", ""])
    if metrics["malformed_examples"]:
        lines.extend(
            [
                "| Question ID | Category | Gold | Predicted Answer |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in metrics["malformed_examples"]:
            lines.append(
                f"| `{row['question_id']}` | `{row['category']}` | "
                f"{_md_escape(row['gold'])} | {_md_escape(row['pred_answer'])} |"
            )
    else:
        lines.append("No malformed outputs were detected.")

    lines.extend(
        [
            "",
            "## Normalization",
            "",
            "- Lowercase with Unicode NFKC normalization.",
            "- Normalize curly apostrophes and Unicode dash variants.",
            "- Normalize `No. 1`, `no 1`, `#1`, and `number 1` ranking answers to `1`.",
            "- Normalize numeric `minute`/`minutes` duration answers to singular `minute`.",
            "- Remove punctuation and English articles, then collapse whitespace.",
            "",
        ]
    )
    return "\n".join(lines)


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score tennis prediction JSONL files.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    parser.add_argument("--output-json", required=True, help="Metrics JSON output path.")
    parser.add_argument("--output-md", required=True, help="Markdown report output path.")
    args = parser.parse_args(argv)

    metrics = score_predictions_file(args.predictions)
    write_json(args.output_json, metrics)
    write_text(args.output_md, render_markdown_report(metrics, args.predictions))

    overall = metrics["overall"]
    print(
        "Scored {n} predictions: EM={em:.4f}, F1={f1:.4f}, malformed={malformed}/{n}".format(
            n=overall["n"],
            em=overall["em"],
            f1=overall["f1"],
            malformed=overall["malformed_count"],
        )
    )


if __name__ == "__main__":
    main()
