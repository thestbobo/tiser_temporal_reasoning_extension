from __future__ import annotations

from conftest import load_repo_module


def load_merge_helper():
    return load_repo_module(
        "scripts/tennis/merge_tennis_traces.py",
        "test_merge_tennis_traces_module",
    )


def source_record(question_id: str = "tennis_001", answer: str = "Yes") -> dict:
    return {
        "dataset_name": "tennis_temporal",
        "question_id": question_id,
        "question": "Did Djokovic call the trainer before Medvedev broke serve?",
        "answer": answer,
        "prompt": "Temporal context:\nDjokovic called the trainer. Later Medvedev broke serve.",
        "category": "yes_no_before_after",
        "tags": ["yes_no_before_after"],
        "output": "<answer>placeholder source output</answer>",
    }


def generated_record(question_id: str = "tennis_001", output: str | None = None) -> dict:
    return {
        "question_id": question_id,
        "output": output if output is not None else valid_output(),
    }


def valid_output(answer: str = "Yes") -> str:
    return (
        "<reasoning>\n"
        "The context places Djokovic's trainer call before Medvedev's break.\n"
        "<timeline>Djokovic called the trainer. Later Medvedev broke serve.</timeline>\n"
        "<reflection>The ordered events support the before relation.</reflection>\n"
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


def merge_one(output: str, *, generated_id: str = "tennis_001", answer: str = "Yes"):
    helper = load_merge_helper()
    return helper.merge_records(
        [source_record(answer=answer)],
        [generated_record(generated_id, output)],
    )


def test_successful_merge_replaces_output_and_preserves_source_record() -> None:
    output = valid_output("YES")
    merged, failures, summary = merge_one(output, answer="Yes")

    assert failures == []
    assert summary["merged_records"] == 1
    assert merged[0]["question_id"] == "tennis_001"
    assert merged[0]["question"] == "Did Djokovic call the trainer before Medvedev broke serve?"
    assert merged[0]["output"] == output


def test_missing_id_rejected_when_not_in_source() -> None:
    merged, failures, summary = merge_one(valid_output(), generated_id="missing_001")

    assert merged == []
    assert summary["failed_records"] == 1
    assert "question_id does not exist in source" in failures[0]["errors"]


def test_missing_answer_tag_rejected() -> None:
    output = (
        "<reasoning>\n"
        "The context places Djokovic's trainer call before Medvedev's break.\n"
        "<timeline>Djokovic called the trainer. Later Medvedev broke serve.</timeline>\n"
        "<reflection>The ordered events support the before relation.</reflection>\n"
        "</reasoning>"
    )

    merged, failures, _summary = merge_one(output)

    assert merged == []
    assert "missing <answer>" in failures[0]["errors"]


def test_answer_mismatch_rejected_after_tennis_normalization() -> None:
    merged, failures, _summary = merge_one(valid_output("No"))

    assert merged == []
    assert "answer mismatch" in failures[0]["errors"]


def test_placeholder_trace_rejected() -> None:
    merged, failures, _summary = merge_one(placeholder_output())

    assert merged == []
    assert "placeholder trace" in failures[0]["errors"]
