from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path

import pytest

from conftest import load_repo_module


def load_train_module():
    return load_repo_module("scripts/tennis/train_tennis.py", "test_train_tennis_module")


def valid_record(output: str = "<reasoning>ok</reasoning><answer>Yes</answer>") -> dict:
    return {
        "question_id": "row_1",
        "prompt": "Question?",
        "output": output,
        "answer": "Yes",
    }


def placeholder_output(answer: str = "Yes") -> str:
    return (
        "<reasoning>\n"
        "The question asks for a temporal relation in the given tennis context.\n"
        "</reasoning>\n"
        "<timeline>\n"
        "A detailed generated timeline will be added in the trace-generation step.\n"
        "</timeline>\n"
        "<reflection>\n"
        "The final answer is checked against the provided gold answer.\n"
        "</reflection>\n"
        f"<answer>{answer}</answer>"
    )


def write_records(path: Path, records: list[dict]) -> Path:
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def write_config(path: Path, train_file: Path, test_file: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "run_name: test_run",
                "seed: 42",
                "paths:",
                f"  data_dir: {train_file.parent.as_posix()}",
                f"  train_file: {train_file.as_posix()}",
                f"  test_file: {test_file.as_posix()}",
                f"  model_dir: {(path.parent / 'model').as_posix()}",
                f"  output_dir: {(path.parent / 'outputs').as_posix()}",
                "model:",
                "  name: fake-model",
                "  load_in_4bit: false",
                "  bnb_4bit_quant_type: nf4",
                "  bnb_4bit_use_double_quant: true",
                "lora:",
                "  r: 1",
                "  alpha: 1",
                "  dropout: 0.0",
                "  target_modules: []",
                "  bias: none",
                "train:",
                "  subset_size: null",
                "  max_seq_len: 128",
                "  per_device_batch_size: 1",
                "  gradient_accumulation_steps: 1",
                "  num_epochs: 0",
                "  learning_rate: 1.0e-4",
                "  lr_scheduler_type: constant",
                "  warmup_ratio: 0.0",
                "  weight_decay: 0.0",
                "  optim: adamw_torch",
                "  gradient_checkpointing: false",
                "  logging_steps: 1",
                "  save_strategy: no",
                "eval:",
                "  max_samples_per_split: 1",
                "  batch_size: 1",
                "  max_new_tokens: 8",
                "  do_sample: false",
                "splits: [tennis_temporal]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_placeholder_traces_abort_without_override(tmp_path: Path) -> None:
    train = load_train_module()
    train_path = write_records(tmp_path / "train.json", [valid_record(placeholder_output())])

    validation = train.validate_training_file(train_path, min_match_rate=0.95)

    with pytest.raises(ValueError, match="refusing to train on placeholder-style traces"):
        train.enforce_placeholder_policy(validation, allow_placeholder_traces=False)


def test_placeholder_traces_allowed_with_explicit_override(tmp_path: Path) -> None:
    train = load_train_module()
    train_path = write_records(tmp_path / "train.json", [valid_record(placeholder_output())])

    validation = train.validate_training_file(train_path, min_match_rate=0.95)

    train.enforce_placeholder_policy(validation, allow_placeholder_traces=True)


def test_missing_train_file_is_reported_clearly(tmp_path: Path) -> None:
    train = load_train_module()

    with pytest.raises(FileNotFoundError, match="Training file does not exist"):
        train.select_train_file(tmp_path / "missing.json")


def test_records_without_prompt_or_output_are_rejected(tmp_path: Path) -> None:
    train = load_train_module()
    train_path = write_records(
        tmp_path / "bad.json",
        [{"question_id": "bad_1", "answer": "Yes"}],
    )

    with pytest.raises(ValueError) as excinfo:
        train.validate_training_file(train_path, min_match_rate=0.95)

    message = str(excinfo.value)
    assert "missing non-empty prompt" in message
    assert "missing non-empty output" in message


def test_missing_answer_tag_is_rejected(tmp_path: Path) -> None:
    train = load_train_module()
    train_path = write_records(
        tmp_path / "bad_answer.json",
        [valid_record("The answer is Yes.")],
    )

    with pytest.raises(ValueError) as excinfo:
        train.validate_training_file(train_path, min_match_rate=0.95)

    assert "missing <answer>" in str(excinfo.value)


def test_validation_aborts_before_training_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train = load_train_module()
    train_file = write_records(tmp_path / "train.json", [valid_record(placeholder_output())])
    test_file = write_records(tmp_path / "test.json", [valid_record()])
    config = write_config(tmp_path / "config.yaml", train_file, test_file)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("src.train"):
            raise AssertionError("trainer import should not happen before validation aborts")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["train_tennis.py", "--config", str(config)])

    with pytest.raises(ValueError, match="refusing to train on placeholder-style traces"):
        train.main()


def test_allow_placeholder_reaches_mocked_training_only_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = load_train_module()
    train_file = write_records(tmp_path / "train.json", [valid_record(placeholder_output())])
    test_file = write_records(tmp_path / "test.json", [valid_record()])
    config = write_config(tmp_path / "config.yaml", train_file, test_file)
    calls = []

    fake_train_pkg = types.ModuleType("src.train")
    fake_train_pkg.__path__ = []
    fake_trainer = types.ModuleType("src.train.trainer")
    fake_trainer.run_training = lambda cfg: calls.append(cfg)

    monkeypatch.setitem(sys.modules, "src.train", fake_train_pkg)
    monkeypatch.setitem(sys.modules, "src.train.trainer", fake_trainer)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_tennis.py", "--config", str(config), "--allow-placeholder-traces"],
    )

    train.main()

    assert len(calls) == 1
    assert Path(calls[0].paths.train_file) == train_file
