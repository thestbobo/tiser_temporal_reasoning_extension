from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os

import torch
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

from src.data.dataset import build_train_dataset
from src.model.loader import build_lora_config, load_model_and_tokenizer
from src.utils.io import read_json, write_json, write_run_meta
from src.utils.seeding import set_seed


def _sft_config(cfg) -> SFTConfig:
    use_bf16 = torch.cuda.is_bf16_supported()
    return SFTConfig(
        output_dir=os.path.join(cfg.paths.output_dir, cfg.run_name, "trainer"),
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        optim=cfg.train.optim,
        max_seq_length=cfg.train.max_seq_len,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=cfg.train.logging_steps,
        save_strategy=cfg.train.save_strategy,
        seed=cfg.seed,
        # packing would concatenate examples and break per-example completion-only masking.
        packing=False,
        report_to=[],
    )


def build_trainer(cfg, model, tokenizer, train_ds) -> SFTTrainer:
    # train_ds is already tokenized (input_ids/attention_mask/labels), so SFTTrainer
    # skips its own prep and does NOT apply the chat template a second time. The
    # seq2seq collator pads input_ids and pads our -100 labels (so completion-only
    # masking is preserved) -- the default LM collator would overwrite labels.
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    kwargs = {
        "model": model,
        "args": _sft_config(cfg),
        "train_dataset": train_ds,
        "tokenizer": tokenizer,
        "data_collator": collator,
    }
    # If training starts from an existing PEFT adapter, it is already attached
    # as trainable. Passing a fresh peft_config would create a second adapter.
    if not getattr(model, "peft_config", None):
        kwargs["peft_config"] = build_lora_config(cfg)
    return SFTTrainer(**kwargs)


def _json_safe(obj):
    return json.loads(json.dumps(obj, default=str))


def _trainer_state_dict(trainer: SFTTrainer) -> dict:
    state = trainer.state
    if hasattr(state, "to_json_string"):
        return json.loads(state.to_json_string())
    if hasattr(state, "to_dict"):
        return _json_safe(state.to_dict())
    if is_dataclass(state):
        return _json_safe(asdict(state))
    return _json_safe(vars(state))


def _save_training_artifacts(
    trainer: SFTTrainer,
    train_metrics: dict,
    *,
    run_dir: str,
    adapter_dir: str,
) -> None:
    trainer_state = _trainer_state_dict(trainer)
    log_history = trainer_state.get("log_history", [])
    run_meta = read_json(os.path.join(run_dir, "run_meta.json"))

    for target_dir in (run_dir, adapter_dir):
        write_json(os.path.join(target_dir, "train_metrics.json"), train_metrics)
        write_json(os.path.join(target_dir, "train_log_history.json"), log_history)
        write_json(os.path.join(target_dir, "trainer_state.json"), trainer_state)
        write_json(os.path.join(target_dir, "run_meta.json"), run_meta)


def run_training(cfg) -> str:
    set_seed(cfg.seed)
    run_dir = os.path.join(cfg.paths.output_dir, cfg.run_name)
    write_run_meta(run_dir, cfg)

    model, tokenizer = load_model_and_tokenizer(cfg)

    train_ds, n_dropped = build_train_dataset(
        cfg.paths.train_file, tokenizer, cfg.train.max_seq_len, cfg.train.subset_size
    )
    print(
        f"[train] {len(train_ds)} examples after length filter "
        f"(dropped {n_dropped} > {cfg.train.max_seq_len} tokens)"
    )

    trainer = build_trainer(cfg, model, tokenizer, train_ds)
    train_result = trainer.train()
    train_metrics = dict(train_result.metrics)
    train_metrics["train_examples_after_filter"] = len(train_ds)
    train_metrics["dropped_long_examples"] = n_dropped
    train_metrics["max_seq_len"] = cfg.train.max_seq_len

    adapter_dir = os.path.join(cfg.paths.model_dir, cfg.run_name, "adapter")
    trainer.save_model(adapter_dir)
    _save_training_artifacts(
        trainer,
        train_metrics,
        run_dir=run_dir,
        adapter_dir=adapter_dir,
    )
    print(f"[train] saved LoRA adapter to {adapter_dir}")
    print(f"[train] saved training metrics to {os.path.join(adapter_dir, 'train_metrics.json')}")
    return adapter_dir
