from __future__ import annotations

import json
from pathlib import Path

from conftest import REPO_ROOT, load_repo_module


def load_validator():
    return load_repo_module(
        "scripts/tennis/validate_tennis_traces.py",
        "test_validate_tennis_traces_module",
    )


def load_prepare_helper():
    return load_repo_module(
        "scripts/tennis/prepare_tennis_trace_generation.py",
        "test_prepare_tennis_trace_generation_module",
    )


def traced_record(output: str, answer: str = "Yes") -> dict:
    return {
        "dataset_name": "tennis_temporal",
        "question_id": "trace_001",
        "question": "Did Sinner break serve before Alcaraz changed rackets?",
        "answer": answer,
        "prompt": (
            "Temporal context:\n"
            "Sinner broke serve. Later, Alcaraz changed rackets.\n\n"
            "Question:\n"
            "Did Sinner break serve before Alcaraz changed rackets?"
        ),
        "category": "yes_no_before_after",
        "tags": ["yes_no_before_after"],
        "output": output,
    }


def valid_output(answer: str = "Yes") -> str:
    return (
        "<reasoning>\n"
        "Sinner's break is stated before Alcaraz's racket change.\n"
        "<timeline>Sinner broke serve. Later, Alcaraz changed rackets.</timeline>\n"
        "<reflection>The ordering supports the requested before relation.</reflection>\n"
        "</reasoning>\n"
        f"<answer>{answer}</answer>"
    )


def placeholder_output(answer: str = "Yes") -> str:
    return (
        "<reasoning>\n"
        "The question asks for a temporal relation in the given tennis context.\n"
        "<timeline>A detailed generated timeline will be added in the trace-generation step.</timeline>\n"
        "<reflection>The final answer is checked against the provided gold answer.</reflection>\n"
        "</reasoning>\n"
        f"<answer>{answer}</answer>"
    )


def test_validator_accepts_valid_traced_sample() -> None:
    validator = load_validator()
    records = json.loads(
        (REPO_ROOT / "data" / "tennis" / "tennis_train_traced_sample.json").read_text(
            encoding="utf-8"
        )
    )

    summary = validator.validate_records(records)

    assert summary["total_records"] == 5
    assert summary["valid_records"] == 5
    assert summary["invalid_records"] == 0


def test_validator_rejects_placeholder_traces() -> None:
    validator = load_validator()

    summary = validator.validate_records([traced_record(placeholder_output())])

    assert summary["invalid_records"] == 1
    assert "placeholder trace" in summary["failed_examples"][0]["errors"]


def test_validator_rejects_answer_mismatch() -> None:
    validator = load_validator()

    summary = validator.validate_records([traced_record(valid_output("No"))])

    assert summary["invalid_records"] == 1
    assert "answer mismatch" in summary["failed_examples"][0]["errors"]


def test_validator_rejects_missing_answer_tag() -> None:
    validator = load_validator()

    summary = validator.validate_records(
        [
            traced_record(
                "<reasoning>\n"
                "The events are ordered clearly.\n"
                "<timeline>Sinner broke serve before Alcaraz changed rackets.</timeline>\n"
                "<reflection>The answer should be affirmative.</reflection>\n"
                "</reasoning>"
            )
        ]
    )

    assert summary["invalid_records"] == 1
    assert "missing <answer>" in summary["failed_examples"][0]["errors"]


def test_generation_helper_prompts_include_gold_answer_and_format_instructions(
    tmp_path: Path,
) -> None:
    helper = load_prepare_helper()
    record = traced_record(valid_output())
    rows = helper.build_request_rows([record], limit=1, offset=0)
    output_path = tmp_path / "requests.jsonl"

    helper.write_jsonl(output_path, rows)
    exported = json.loads(output_path.read_text(encoding="utf-8").strip())
    prompt = exported["prompt"]

    assert exported["gold_answer"] == "Yes"
    assert "Gold answer:\nYes" in prompt
    assert "<reasoning>" in prompt
    assert "<timeline>" in prompt
    assert "<reflection>" in prompt
    assert "<answer>GOLD_ANSWER_EXACTLY</answer>" in prompt
    assert "No external API" not in prompt
