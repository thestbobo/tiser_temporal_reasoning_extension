"""Export tennis training examples for offline supervised trace generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = Path("data/tennis/tennis_train.json")
DEFAULT_OUTPUT = Path(
    "results/tennis_domain_adaptation/trace_generation/tennis_trace_requests.jsonl"
)


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: record {index} is not an object")
        records.append(item)
    return records


def extract_context(record: dict[str, Any], *, index: int) -> str:
    context = record.get("context")
    if isinstance(context, str) and context.strip():
        return context.strip()

    prompt = record.get("prompt")
    if isinstance(prompt, str):
        start_marker = "Temporal context:\n"
        end_marker = "\n\nQuestion:"
        if start_marker in prompt and end_marker in prompt:
            context_text = prompt.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
            if context_text:
                return context_text

    raise ValueError(f"record {index} has no context and no extractable prompt context")


def required_string(record: dict[str, Any], key: str, *, index: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record {index} is missing non-empty {key}")
    return value.strip()


def build_generation_prompt(record: dict[str, Any], *, index: int) -> str:
    context = extract_context(record, index=index)
    question = required_string(record, "question", index=index)
    answer = required_string(record, "answer", index=index)
    category = str(record.get("category", "unknown")).strip() or "unknown"
    return "\n".join(
        [
            "Generate one supervised TISER-style tennis temporal reasoning trace.",
            "Use only the provided temporal context. Do not add facts from outside it.",
            "The gold answer is provided because this is supervised trace construction.",
            "The final <answer> text must exactly match the gold answer.",
            "Return only the trace. Do not include markdown fences or extra commentary.",
            "",
            "Required output format:",
            "<reasoning>",
            "Concise reasoning over the temporal relation.",
            "<timeline>Ordered events needed to answer the question.</timeline>",
            "<reflection>Brief verification that the answer follows from the timeline.</reflection>",
            "</reasoning>",
            "<answer>GOLD_ANSWER_EXACTLY</answer>",
            "",
            f"Category: {category}",
            "",
            "Temporal context:",
            context,
            "",
            "Question:",
            question,
            "",
            "Gold answer:",
            answer,
        ]
    )


def build_request_rows(records: list[dict[str, Any]], *, limit: int, offset: int) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    if offset < 0:
        raise ValueError("--offset must be non-negative")

    selected = records[offset : offset + limit]
    rows = []
    for index, record in enumerate(selected, start=offset + 1):
        rows.append(
            {
                "dataset_name": required_string(record, "dataset_name", index=index),
                "question_id": required_string(record, "question_id", index=index),
                "category": str(record.get("category", "unknown")).strip() or "unknown",
                "question": required_string(record, "question", index=index),
                "gold_answer": required_string(record, "answer", index=index),
                "prompt": build_generation_prompt(record, index=index),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tennis Trace Generation Batch",
        "",
        "Generate one strict trace per example. Do not call APIs automatically from this file.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['question_id']}",
                "",
                f"- Dataset: `{row['dataset_name']}`",
                f"- Category: `{row['category']}`",
                f"- Gold answer: `{row['gold_answer']}`",
                "",
                "```text",
                row["prompt"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare offline prompts for tennis trace generation."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Base tennis train JSON array.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output batch file.")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples to export.")
    parser.add_argument("--offset", type=int, default=0, help="Number of examples to skip first.")
    parser.add_argument(
        "--format",
        choices=("jsonl", "markdown"),
        default="jsonl",
        help="Batch output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output)
    records = load_json_array(input_path)
    rows = build_request_rows(records, limit=args.limit, offset=args.offset)

    if args.format == "markdown":
        write_markdown(output_path, rows)
    else:
        write_jsonl(output_path, rows)

    print(f"Input: {relative_or_absolute(input_path)}")
    print(f"Exported examples: {len(rows)}")
    print(f"Output: {relative_or_absolute(output_path)}")
    print("No external API was called.")


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
