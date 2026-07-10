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


def train_record(
    question_id: str,
    *,
    category: str = "yes_no_before_after",
    context: str | None = None,
    prompt_context: str | None = None,
) -> dict:
    context_text = context if context is not None else prompt_context
    prompt = (
        "Temporal context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"What happened in {question_id}?"
    )
    record = {
        "dataset_name": "tennis_temporal",
        "question_id": question_id,
        "question": f"What happened in {question_id}?",
        "answer": "Yes",
        "prompt": prompt,
        "category": category,
    }
    if context is not None:
        record["context"] = context
    return record


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
    assert "<answer>Yes</answer>" in prompt
    assert "Do not invent events" in prompt
    assert "Do not change the gold answer" in prompt
    assert "Return only one JSONL line with question_id and output" in prompt
    assert "No external API" not in prompt


def test_full_batch_preparation_excludes_traced_ids_and_writes_manifest(
    tmp_path: Path,
) -> None:
    helper = load_prepare_helper()
    records = [
        train_record(
            f"tennis_{index:06d}",
            category="rally" if index <= 60 else "serve",
            prompt_context=f"Event {index} happened before the answer event.",
        )
        for index in range(1, 106)
    ]
    input_path = tmp_path / "tennis_train.json"
    traced_path = tmp_path / "traced.json"
    generated_path = tmp_path / "generated.jsonl"
    output_dir = tmp_path / "batches"

    input_path.write_text(json.dumps(records), encoding="utf-8")
    traced_path.write_text(
        json.dumps([{"question_id": f"tennis_{index:06d}"} for index in range(1, 3)]),
        encoding="utf-8",
    )
    generated_path.write_text(
        "\n".join(
            json.dumps({"question_id": f"tennis_{index:06d}", "output": valid_output()})
            for index in range(3, 6)
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = helper.prepare_batches(
        input_path=input_path,
        output_dir=output_dir,
        exclude_files=[traced_path, generated_path],
        batch_size=50,
        limit=None,
        output_format="jsonl",
    )

    assert manifest["total_train_examples"] == 105
    assert manifest["excluded_ids_count"] == 5
    assert manifest["selected_count"] == 100
    assert manifest["batch_size"] == 50
    assert manifest["number_of_batches"] == 2
    assert manifest["selected_question_ids"][0] == "tennis_000006"
    assert "tennis_000001" not in manifest["selected_question_ids"]
    assert len(manifest["selected_question_ids"]) == len(set(manifest["selected_question_ids"]))
    assert manifest["counts_by_category"] == {"rally": 55, "serve": 45}
    assert (output_dir / "manifest.json").exists()

    first_batch = [
        json.loads(line)
        for line in (output_dir / "batch_001.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    second_batch = [
        json.loads(line)
        for line in (output_dir / "batch_002.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(first_batch) == 50
    assert len(second_batch) == 50
    assert set(first_batch[0]) == {
        "question_id",
        "dataset_name",
        "category",
        "context",
        "question",
        "gold_answer",
        "prompt",
    }
    assert first_batch[0]["context"] == "Event 6 happened before the answer event."
    assert manifest["batches"] == [
        {
            "file": "batch_001.jsonl",
            "question_ids": [f"tennis_{index:06d}" for index in range(6, 56)],
        },
        {
            "file": "batch_002.jsonl",
            "question_ids": [f"tennis_{index:06d}" for index in range(56, 106)],
        },
    ]


def test_selects_all_remaining_when_limit_is_omitted_or_all() -> None:
    helper = load_prepare_helper()
    records = [train_record(f"tennis_{index:06d}", prompt_context="A before B.") for index in range(1, 8)]
    excluded_ids = {"tennis_000001", "tennis_000003"}

    omitted = helper.select_untraced_records(records, excluded_ids=excluded_ids, limit=None)
    explicit_all = helper.select_untraced_records(records, excluded_ids=excluded_ids, limit="all")

    assert [record["question_id"] for record in omitted] == [
        "tennis_000002",
        "tennis_000004",
        "tennis_000005",
        "tennis_000006",
        "tennis_000007",
    ]
    assert explicit_all == omitted


def test_duplicate_question_ids_are_rejected() -> None:
    helper = load_prepare_helper()
    records = [
        train_record("tennis_000001", prompt_context="A before B."),
        train_record("tennis_000001", prompt_context="C before D."),
    ]

    try:
        helper.select_untraced_records(records, excluded_ids=set(), limit=None)
    except ValueError as exc:
        assert "duplicate question_id" in str(exc)
    else:
        raise AssertionError("duplicate question_id was not rejected")


def test_context_extraction_uses_context_field_or_prompt_context() -> None:
    helper = load_prepare_helper()
    rows = helper.build_rows(
        [
            train_record("tennis_000001", context="Explicit context wins."),
            train_record("tennis_000002", prompt_context="Prompt context is extracted."),
        ]
    )

    assert rows[0]["context"] == "Explicit context wins."
    assert rows[1]["context"] == "Prompt context is extracted."
