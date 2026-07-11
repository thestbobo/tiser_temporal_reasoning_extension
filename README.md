# TISER Baseline

A faithful baseline reproduction of **TISER**, "Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models" (Bazaga, Blloshmi, Byrne, de Gispert, ACL 2025; [arXiv:2504.05258](https://arxiv.org/abs/2504.05258)), plus two extensions: a context-memory conflict probe and a tennis domain adaptation study.

TISER is ordinary LoRA supervised fine-tuning where the training target is a structured four-section trace: `<reasoning>` (chain-of-thought), `<timeline>` (ordered dated events), `<reflection>` (self-audit of the reasoning against the timeline), `<answer>` (short final answer). The model architecture, the next-token cross-entropy loss, and the optimizer are all unchanged from vanilla SFT. The contribution is the structured target format, the synthetic data that teaches it, and a matching inference prompt. At inference the model generates the whole trace in a single pass and a regex parser extracts `<answer>` for scoring.

This repo fine-tunes `Qwen2.5-7B-Instruct` with LoRA on the released `TISER_train.json` (bf16 base weights, no quantization; a 4-bit QLoRA path remains available via `model.load_in_4bit` for smaller GPUs) and evaluates exact-match / token-F1 on `TISER_test.json`: the five in-domain splits form the macro average, and the OOD ToT-semantic split is reported separately. It uses the released `prompt`/`output` fields verbatim, so the trained model speaks the exact format the data was generated against.

## Layout

| Path | Contents |
|---|---|
| `config/config.yaml` | Baseline hyperparameters and paths |
| `config/config_smoke.yaml` | Tiny 1.5B / 300-example / 1-epoch config for a fast pipeline smoke test |
| `config/conflict.yaml` | Context-memory conflict pipeline config |
| `config/config_tennis*.yaml` | Tennis adaptation configs |
| `src/data/` | Dataset loading, prompt/completion mapping, length filtering |
| `src/model/` | Base-model load (bf16 or optional 4-bit) + PEFT LoRA wiring |
| `src/train/` | TRL `SFTTrainer` setup (completion-only loss) |
| `src/inference/` | Batched generation + defensive 4-section parser |
| `src/eval/` | SQuAD-style normalize + EM/F1 + per-split & macro aggregation |
| `src/conflict/` | Context-memory conflict pipeline stages |
| `src/tennis/` | Tennis dataset build, prompts, evaluation |
| `src/utils/` | Config loading, seeding, IO + run metadata |
| `scripts/` | Thin CLI entrypoints: `fetch_data.py`, `train.py`, `evaluate.py`, `conflict/`, `tennis/` |
| `notebooks/` | `colab_run.ipynb` (baseline) and `colab_tennis_experiments.ipynb` |
| `results/` | Committed headline metrics: `baseline/`, `context_memory_conflict/`, tennis comparisons |
| `data/`, `outputs/` | Gitignored datasets and raw run outputs (`data/tennis/` is tracked; it is small) |

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

(On local machines also install torch, e.g. `pip install "torch>=2.3"`. On Colab use the bundled torch; do not reinstall it.)

## Run

```bash
python scripts/fetch_data.py --config config/config.yaml
python scripts/train.py --config config/config.yaml
python scripts/evaluate.py --config config/config.yaml
```

`scripts/fetch_data.py` first tries real local files, then falls back to the HuggingFace mirror / upstream git-LFS repo, and validates the schema before writing into `data/`.

## Compute notes

The committed `config.yaml` defaults to the **full** 54,488-example training run and full 22,014-example evaluation. Our full run trained in about 6.5 h on a single RunPod H200 SXM (bf16 LoRA, roughly 7 samples/s); any 80 GB-class GPU (A100/H100) works, with proportionally longer wall-clock. The full test pass takes about 17 min with the vLLM engine (`eval.engine: vllm`); HF `generate` is much slower. On a free-tier Colab T4 the full run will not finish; use the subset knobs for a smoke test:

```bash
python scripts/train.py --config config/config.yaml --subset 2000
python scripts/evaluate.py --config config/config.yaml --max-samples-per-split 200
```

To scale back up, set `train.subset_size: null` and `eval.max_samples_per_split: null`, and raise `train.per_device_batch_size` / `eval.batch_size`. No code changes needed.

## Results

Our full-test reproduction (see `results/baseline/tiser_qwen7b_full/metrics.json`): **macro-EM 0.878 / macro-F1 0.949** over the five in-domain splits; OOD ToT-semantic EM 0.381, reported separately and excluded from the macro.

Paper reference numbers (Table 1, Qwen2.5-7B + TISER):

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

Inference-only probe: elicit the model's closed-book memory, keep items it confidently knows, deterministically edit the context to contradict that memory, and measure whether the model follows the text or its memory across a 2x2 (model x prompt) run matrix. Headline: faithful-EM rises from 0.380 (base model, standard prompt) to 0.787 (TISER fine-tune, TISER prompt), but an LLM-agent audit finds the `<reflection>` step names the contradiction in only 4.2% of rows.

Pipeline stages are `scripts/conflict/01_build_subset.py` through `07_confidence_vs_reflection.py` plus `08_stats.py` (bootstrap CIs and McNemar), all driven by `config/conflict.yaml`. Artifacts:

```text
results/context_memory_conflict/
docs/extensions/context_memory_conflict/
```

### Tennis domain adaptation

Completed second extension: does TISER-style temporal reasoning transfer to professional-tennis narratives? 1,122 raw QA examples are audited, converted to TISER format, and split 785/113/224 (train/dev/test, seed 42). At 0.5B a tennis-only adapter lifts EM 0.379 to 0.464; at 7B the original TISER adapter transfers at EM 0.580 / F1 0.701 and continued tennis adaptation reaches EM 0.732 / F1 0.856 (see `results/tennis_from_tiser_experiments/comparisons/adapter_comparison.json`).

Code, data, and artifacts:

```text
src/tennis/  scripts/tennis/  data/tennis/
results/tennis_domain_adaptation/  results/tennis_from_tiser_experiments/
docs/extensions/tennis_domain_adaptation/
```
