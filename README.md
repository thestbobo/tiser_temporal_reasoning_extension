# TISER Baseline

A faithful baseline reproduction of **TISER** — "Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models" (Bazaga, Blloshmi, Byrne, de Gispert, ACL 2025; [arXiv:2504.05258](https://arxiv.org/abs/2504.05258)).

TISER is ordinary LoRA supervised fine-tuning where the training target is a structured four-section trace: `<reasoning>` (chain-of-thought) → `<timeline>` (ordered dated events) → `<reflection>` (self-audit of the reasoning against the timeline) → `<answer>` (short final answer). The model architecture, the next-token cross-entropy loss, and the optimizer are all unchanged from vanilla SFT. The contribution is the structured target format, the synthetic data that teaches it, and a matching inference prompt — at inference the model generates the whole trace in a single pass and a regex parser extracts `<answer>` for scoring.

This repo fine-tunes `Qwen2.5-7B-Instruct` with QLoRA (4-bit nf4) on the released `TISER_train.json` and evaluates exact-match / token-F1 on the five in-domain splits of `TISER_test.json`. It uses the released `prompt`/`output` fields verbatim, so the trained model speaks the exact format the data was generated against. No iterative loop, no OOD splits, no demo — just the baseline.

## Layout

| Path | Contents |
|---|---|
| `config/config.yaml` | Single source of truth for all hyperparameters and paths |
| `config/config_smoke.yaml` | Tiny 1.5B / 300-example / 1-epoch config for a fast pipeline smoke test |
| `src/data/` | Dataset loading, prompt/completion mapping, length filtering |
| `src/model/` | QLoRA 4-bit base-model load + PEFT LoRA wiring |
| `src/train/` | TRL `SFTTrainer` setup (completion-only loss) |
| `src/inference/` | Batched generation + defensive 4-section parser |
| `src/eval/` | SQuAD-style normalize + EM/F1 + per-split & macro aggregation |
| `src/utils/` | Config loading, seeding, IO + run metadata |
| `scripts/` | Thin CLI entrypoints: `fetch_data.py`, `train.py`, `evaluate.py` |
| `notebooks/` | `colab_run.ipynb` — install → fetch → train → eval on Colab |
| `data/`, `model/`, `outputs/` | Gitignored: datasets, adapters, predictions/metrics |

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

(On local machines also install torch, e.g. `pip install "torch>=2.3,<2.6"`. On Colab use the bundled torch — do not reinstall it.)

## Run

```bash
python scripts/fetch_data.py --config config/config.yaml
python scripts/train.py --config config/config.yaml
python scripts/evaluate.py --config config/config.yaml
```

`scripts/fetch_data.py` first tries real local files, then falls back to the HuggingFace mirror / upstream git-LFS repo, and validates the schema before writing into `data/`.

## Compute notes

The committed `config.yaml` defaults to the **full** 60k-example training run and full 22k-example evaluation — this realistically needs an **A100** (≈10–15 h training on one A100; the paper used 8×A100). On a **free-tier Colab T4 the full run will not finish**; use the subset knobs for a smoke test:

```bash
python scripts/train.py --config config/config.yaml --subset 2000
python scripts/evaluate.py --config config/config.yaml --max-samples-per-split 200
```

To scale back up on an A100, set `train.subset_size: null`, `eval.max_samples_per_split: null`, and raise `train.per_device_batch_size` / `eval.batch_size` — no code changes.

## Expected results (paper Table 1 — Qwen2.5-7B + TISER)

| Split | EM | F1 |
|---|---|---|
| TGQA | 84.5 | 94.2 |
| TempReason_L2 | 85.5 | 87.5 |
| TempReason_L3 | 91.5 | 94.9 |
| TimeQA_easy | 97.9 | 98.3 |
| TimeQA_hard | 96.1 | 97.2 |
| **Macro** | **91.1** | **94.4** |

## Extensions

### Context-memory conflict

Artifacts are stored under:

```text
results/context_memory_conflict/
docs/extensions/context_memory_conflict/
```

### Tennis domain adaptation

Work-in-progress extension.

Planned code:

```text
src/tennis/
scripts/tennis/
```

Planned data:

```text
data/tennis/
```

Planned results:

```text
results/tennis_domain_adaptation/
```
