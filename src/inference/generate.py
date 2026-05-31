from __future__ import annotations

import torch
from tqdm import tqdm


def generate_batch(model, tokenizer, prompts: list[str], gen_cfg) -> list[str]:
    outputs: list[str] = []
    for start in tqdm(range(0, len(prompts), gen_cfg.batch_size), desc="generate"):
        batch = prompts[start : start + gen_cfg.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=gen_cfg.max_new_tokens,
                do_sample=gen_cfg.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Left-padded batch -> the prompt occupies a fixed width; new tokens follow it.
        new_tokens = generated[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs
