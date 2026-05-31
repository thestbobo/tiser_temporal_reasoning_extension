from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def _compute_dtype() -> torch.dtype:
    # A100 -> bf16, T4 -> fp16.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _bnb_config(cfg) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=cfg.model.load_in_4bit,
        bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.model.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_compute_dtype(),
    )


def _load_tokenizer(name: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(cfg) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = _load_tokenizer(cfg.model.name)
    tokenizer.padding_side = "right"  # training; generation flips this to "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=_bnb_config(cfg),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.train.gradient_checkpointing
    )
    return model, tokenizer


def build_lora_config(cfg) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )


def load_adapter_for_inference(cfg, adapter_dir: str) -> tuple[PeftModel, PreTrainedTokenizer]:
    tokenizer = _load_tokenizer(cfg.model.name)
    tokenizer.padding_side = "left"  # left-pad for batched generation

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=_bnb_config(cfg),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer
