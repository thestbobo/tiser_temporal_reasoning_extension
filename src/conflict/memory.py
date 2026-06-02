"""M1 - closed-book parametric-memory elicitation (self-consistency).

For each in-scope item we ask the question with NO temporal context (the answer
span lived only in the context, so this is a genuine memory probe) and draw 'k'
samples at temperature 'T'. The item's belief 'm' is the majority answer; its
'agreement' is the majority share. Run once per model entry (the fine-tuned TISER
subject + the off-the-shelf base as a cross-check, per D-MEM).

Closed-book prompt styles (D-CB-PROMPT):
  - 'tiser_no_context'  : the verbatim TISER preamble + question, context removed (default)
  - 'tiser_empty_note'  : same, with an explicit empty-context placeholder

Output: 'outputs/conflict/memory/memory_<tag>.jsonl' (+ 'memory_<tag>_report.json').
The report carries the malformed-rate per split, the D-MEM fallback trigger.

GPU stage. Run on the box that holds the base model + adapter.
"""

from __future__ import annotations

import gc
import os
from collections import Counter

from src.conflict.context_parse import inject_empty_note
from src.conflict.io import read_jsonl, stage_dir, write_jsonl, write_run_meta
from src.inference.generate import generate_batch
from src.inference.parser import parse_answer
from src.eval.metrics import normalize_answer
from src.model.loader import load_adapter_for_inference, load_base_for_inference
from src.utils.config import REPO_ROOT
from src.utils.io import write_json
from src.utils.seeding import set_seed


def build_closed_book_prompts(items: list[dict], style: str) -> list[str]:
    if style == "tiser_no_context":
        return [it["prompt_no_context"] for it in items]
    if style == "tiser_empty_note":
        return [inject_empty_note(it["prompt_no_context"]) for it in items]
    raise ValueError(f"unknown closed_book_style: {style!r}")


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _majority(answers: list[str]) -> tuple[str, float]:
    """Return (representative raw answer, agreement) by normalized-answer vote."""
    if not answers:
        return "", 0.0
    norm = [normalize_answer(a) for a in answers]
    top_key, top_count = Counter(norm).most_common(1)[0]
    # Representative raw answer = first sample whose normalization is the winner.
    rep = next(a for a, n in zip(answers, norm) if n == top_key)
    return rep, top_count / len(answers)


def _load(cfg, entry):
    adapter = entry.get("adapter_dir")
    if adapter:
        return load_adapter_for_inference(cfg, _resolve(adapter))
    return load_base_for_inference(cfg)


def elicit_for_model(cfg, items: list[dict], entry: dict) -> dict:
    tag = entry["tag"]
    mem_cfg = cfg.conflict.memory
    k = mem_cfg.k
    set_seed(cfg.seed)

    model, tokenizer = _load(cfg, entry)
    prompts = build_closed_book_prompts(items, mem_cfg.closed_book_style)

    samples_flat = generate_batch(
        model, tokenizer, prompts, cfg.eval,
        temperature=mem_cfg.temperature, top_p=mem_cfg.top_p, num_return_sequences=k,
    )
    greedy_flat = (
        generate_batch(model, tokenizer, prompts, cfg.eval)
        if mem_cfg.include_greedy else [None] * len(items)
    )

    rows = []
    for i, it in enumerate(items):
        block = samples_flat[i * k : (i + 1) * k]
        parsed = [parse_answer(s) for s in block]
        answers = [p.answer for p in parsed]
        n_malformed = sum(p.malformed for p in parsed)
        m, agreement = _majority(answers)
        greedy = parse_answer(greedy_flat[i]).answer if greedy_flat[i] is not None else None
        rows.append(
            {
                "id": it["id"],
                "split": it["split"],
                "m": m,
                "agreement": agreement,
                "greedy": greedy,
                "n_malformed": n_malformed,
                "samples": answers,
            }
        )

    out_dir = stage_dir(cfg, "memory")
    write_jsonl(os.path.join(out_dir, f"memory_{tag}.jsonl"), rows)
    report = _report(rows, k, tag, mem_cfg.closed_book_style)
    write_json(os.path.join(out_dir, f"memory_{tag}_report.json"), report)
    write_run_meta(out_dir, cfg, extra={"model_tag": tag, "model_entry": dict(entry)})

    # Free the model before the next entry (two 7B loads back-to-back otherwise).
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"[M1:{tag}] {len(rows)} items | malformed-rate {report['malformed_rate']:.3f} "
          f"| mean agreement {report['mean_agreement']:.3f} -> {out_dir}/memory_{tag}.jsonl")
    return report


def elicit_memory(cfg, items: list[dict] | None = None, only_tag: str | None = None) -> dict:
    if items is None:
        items = read_jsonl(os.path.join(stage_dir(cfg, "subset"), "items.jsonl"))
    entries = [e for e in cfg.conflict.memory.models if only_tag is None or e["tag"] == only_tag]
    if not entries:
        raise ValueError(f"no model entry matches tag {only_tag!r}")
    return {e["tag"]: elicit_for_model(cfg, items, e) for e in entries}


def _report(rows: list[dict], k: int, tag: str, style: str) -> dict:
    per_split: dict[str, dict] = {}
    for r in rows:
        s = per_split.setdefault(r["split"], {"n": 0, "malformed": 0, "agree_sum": 0.0})
        s["n"] += 1
        s["malformed"] += r["n_malformed"]
        s["agree_sum"] += r["agreement"]
    per_split_out = {
        name: {
            "n": s["n"],
            "malformed_rate": s["malformed"] / (s["n"] * k),
            "mean_agreement": s["agree_sum"] / s["n"],
        }
        for name, s in sorted(per_split.items())
    }
    n = len(rows)
    return {
        "model_tag": tag,
        "closed_book_style": style,
        "k": k,
        "n_total": n,
        "malformed_rate": sum(r["n_malformed"] for r in rows) / (n * k) if n else 0.0,
        "mean_agreement": sum(r["agreement"] for r in rows) / n if n else 0.0,
        "per_split": per_split_out,
    }
