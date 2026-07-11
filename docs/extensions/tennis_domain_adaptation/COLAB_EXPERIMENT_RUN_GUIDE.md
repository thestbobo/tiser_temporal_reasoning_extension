# Colab Experiment Run Guide  Tennis Domain Adaptation

## 1. Purpose

Use `notebooks/colab_tennis_experiments.ipynb` to run the tennis domain adaptation experiments from Google Colab. Colab is useful here because the base model and adapters are Qwen instruction models, and full 7B evaluation or training should be run on a GPU runtime.

The notebook is only a runner. It should mount or clone the repository, install dependencies, set editable flags and paths, then call scripts under `scripts/tennis/` with shell commands. Model loading, generation, scoring, trace validation, training, comparison, and aggregation all live in repository scripts.

The notebook supports:

- smoke evaluation with a small `--limit`
- full tennis-test evaluation
- optional training after traces are validated
- adapter comparison
- final aggregation
- result zipping for download or Drive copy

Do not run full training or full 7B evaluation until smoke checks pass.

Current database note: several full evaluations have already been run. The
0.5B standalone tennis results live under
`results/tennis_domain_adaptation/scored/`; the 7B original-TISER transfer and
tennis-from-TISER continued-adaptation grid live under
`results/tennis_from_tiser_experiments/scored/`. Use this guide to rerun or
extend those experiments, not as evidence that they are still missing.

## 2. Recommended Colab Runtime

Use `Runtime > Change runtime type > GPU`.

Recommended hardware:

- A100: preferred for full Qwen2.5-7B adapter work.
- L4: usually workable with 4-bit loading and low batch size.
- T4: use only small `BATCH_SIZE`, often `1`, and expect slower runs.

The full config uses `Qwen/Qwen2.5-7B-Instruct` with 4-bit loading. If CUDA memory is tight, keep:

```python
BATCH_SIZE = 1
MAX_NEW_TOKENS = 256
```

Start with the smoke commands using `--limit 5`.

## 3. Repository Setup in Colab

Open the notebook:

```text
notebooks/colab_tennis_experiments.ipynb
```

Run the runtime setup cells at the top.

Option A: use a repository already stored in Google Drive.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/tiser_temporal_reasoning_extension")
print(PROJECT_ROOT)
```

```python
%cd {PROJECT_ROOT}
!pwd
!ls
```

Option B: clone from GitHub.

```python
from pathlib import Path
import subprocess

PROJECT_ROOT = Path("/content/tiser_temporal_reasoning_extension")
GITHUB_REPO_URL = "https://github.com/YOUR_ORG/YOUR_REPO.git"

if not PROJECT_ROOT.exists():
    subprocess.run(f'git clone "{GITHUB_REPO_URL}" "{PROJECT_ROOT}"', shell=True, check=True)
```

```python
%cd {PROJECT_ROOT}
!pwd
!ls
```

After setup, the current directory must be the repository root, the directory that contains `requirements.txt`, `config/`, `scripts/`, `data/`, and `notebooks/`.

## 4. Dependency Installation

Run:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` intentionally does not pin `torch`; it expects Colab's CUDA-compatible torch build.

If the base model or adapters require Hugging Face authentication, log in before running evaluation or training:

```python
from huggingface_hub import login
login()
```

Then verify Python and CUDA:

```python
import platform
import torch

print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
```

## 5. Required Files Before Running

Run the notebook preflight cell before any evaluation. It checks the main files and adapter paths.

| Path | Required for | If missing | How to generate or fix |
| --- | --- | --- | --- |
| `data/tennis/tennis_test.json` | E0, E1, E2, E3 tennis-test evaluation | Full evaluation cannot run. | Rebuild tennis data with the data preparation pipeline, or restore the file from Drive/Git. |
| `data/tennis/tennis_train_traced_full.json` | Meaningful tennis-only and mixed-replay training | Do not train yet. The training wrapper may inspect `tennis_train.json`, but placeholder traces abort training by default. | Use `scripts/tennis/generate_tennis_traces.py` prepare and validate modes. |
| `config/config_tennis.yaml` | Full 7B evaluation and full training | Full commands cannot run. | Restore the config. It should use `Qwen/Qwen2.5-7B-Instruct`. |
| `config/config_tennis_smoke.yaml` | Smoke training or 0.5B plumbing checks | Smoke training cannot run. | Restore the config. It uses `Qwen/Qwen2.5-0.5B-Instruct` and is not compatible with 7B adapters. |
| `model/tiser_qwen7b_full/adapter` | E1 original TISER adapter evaluation and forgetting baseline | E1 cannot run. | Copy the adapter into Colab/Drive, or update `ORIGINAL_TISER_ADAPTER` to the correct existing path. |
| `model/tiser_tennis_only_qwen7b/adapter` | E2 tennis-only evaluation | E2 cannot run. | Train E2 from validated tennis traces, copy the adapter, or update `TENNIS_ONLY_ADAPTER`. |
| `model/tiser_tennis_mixed_replay_qwen7b/adapter` | E3 mixed replay evaluation | E3 cannot run. | Train E3 after validated tennis traces and original TISER replay data exist, copy the adapter, or update `MIXED_REPLAY_ADAPTER`. |

Additional files used by optional steps:

- `data/TISER_train.json`: required to build a mixed tennis plus original TISER replay dataset.
- `data/tennis/original_tiser_eval_sample.json`: required for forgetting evaluation on the original TISER sample.
- `data/tennis/tennis_mixed_replay_train.json`: output of the mixed replay dataset build step.

Current script note: use `results/tennis_domain_adaptation/scored/mixed_replay` for the E3 final aggregation folder.

## 6. Execution Order

Step 0, mandatory: Preflight checks.

Step 1, mandatory first run: Smoke Base Qwen evaluation with standard and TISER-style prompts.

Step 2, optional after smoke passes: Full Base Qwen tennis evaluation, E0.

Step 3, for reruns when `model/tiser_qwen7b_full/adapter` or the equivalent
off-machine adapter is available: Original TISER adapter evaluation, E1. A full
224-example E1 result already exists as `original_tiser_qwen7b_test224`.

Step 4, optional unless training adapters: Generate and validate traces if `tennis_train_traced_full.json` is missing.

Step 5, if extending or rerunning: Train tennis-only adapter if missing. The
current database already contains `tennis_train_traced_full.json`, a 0.5B
tennis-only full600 evaluation, and a completed 7B tennis-from-TISER grid.

Step 6, still unreported in the current database: Build mixed replay dataset if
traced tennis data and `data/TISER_train.json` exist.

Step 7, still unreported in the current database: Train mixed replay adapter if
missing.

Step 8, mandatory for E2 comparison: Evaluate tennis-only adapter.

Step 9, mandatory for E3 comparison: Evaluate mixed replay adapter.

Step 10, mandatory after scored metrics exist: Compare adapters.

Step 11, recommended for final report tables: Aggregate final results.

Step 12, recommended in Colab: Zip and download or copy results to Drive.

Only Steps 0, 1, 10, 11, and 12 are cheap. Steps 2, 3, 8, and 9 load the 7B model or adapters. Steps 5 and 7 train adapters.

Optional dry-run plan runner:

```bash
python scripts/tennis/run_experiment_plan.py \
  --config config/config_tennis.yaml \
  --tennis-test data/tennis/tennis_test.json \
  --original-tiser-adapter model/tiser_qwen7b_full/adapter \
  --tennis-adapter model/tiser_tennis_only_qwen7b/adapter \
  --mixed-adapter model/tiser_tennis_mixed_replay_qwen7b/adapter \
  --limit 5
```

Without `--execute`, this writes `results/tennis_domain_adaptation/comparisons/run_tennis_experiments.sh` and prints the planned commands. Do not pass `--execute` until all prerequisites are present and you intend to run full evaluations.

## 7. Smoke Test Commands

Base model smoke evaluation, using the same 7B config but only five examples:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --condition base_qwen_smoke \
  --no-adapter \
  --limit 5 \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir results/tennis_domain_adaptation/scored/base_qwen_smoke
```

Optional original TISER smoke evaluation. This requires `model/tiser_qwen7b_full/adapter`:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --adapter-dir model/tiser_qwen7b_full/adapter \
  --condition original_tiser_smoke \
  --limit 5 \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir results/tennis_domain_adaptation/scored/original_tiser_smoke
```

The smoke config `config/config_tennis_smoke.yaml` uses Qwen2.5-0.5B and is suitable for base-model or training-pipeline smoke checks only. Do not load a 7B adapter with the 0.5B smoke config.

## 8. Full Evaluation Commands

E0, base Qwen model, no adapter:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --condition base_qwen \
  --no-adapter \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir results/tennis_domain_adaptation/scored/base_qwen
```

E1, original TISER adapter:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --adapter-dir model/tiser_qwen7b_full/adapter \
  --condition original_tiser \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir results/tennis_domain_adaptation/scored/original_tiser
```

E2, tennis-only adapter:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --adapter-dir model/tiser_tennis_only_qwen7b/adapter \
  --condition tennis_only \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir results/tennis_domain_adaptation/scored/tennis_only
```

If your E2 adapter was produced by `train_tennis.py` with the existing training docs, use:

```text
model/tiser_tennis_only_qwen7b/adapter
```

E3, mixed tennis plus TISER replay adapter:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --adapter-dir model/tiser_tennis_mixed_replay_qwen7b/adapter \
  --condition mixed_replay \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir results/tennis_domain_adaptation/scored/mixed_replay
```

For `aggregate_tennis_results.py` in the current repo, E3 must also be available as `mixed_replay`:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --adapter-dir model/tiser_tennis_mixed_replay_qwen7b/adapter \
  --condition mixed_replay \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir results/tennis_domain_adaptation/scored/mixed_replay
```

If your E3 adapter was produced by the existing training docs, use:

```text
model/tiser_tennis_mixed_replay_qwen7b/adapter
```

Optional forgetting evaluations on the original TISER sample, if `data/tennis/original_tiser_eval_sample.json` exists:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/original_tiser_eval_sample.json \
  --adapter-dir model/tiser_qwen7b_full/adapter \
  --condition original_tiser \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir results/tennis_domain_adaptation/scored_original_tiser_sample/original_tiser
```

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/original_tiser_eval_sample.json \
  --adapter-dir model/tiser_tennis_only_qwen7b/adapter \
  --condition tennis_only \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir results/tennis_domain_adaptation/scored_original_tiser_sample/tennis_only
```

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/original_tiser_eval_sample.json \
  --adapter-dir model/tiser_tennis_mixed_replay_qwen7b/adapter \
  --condition mixed_replay \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir results/tennis_domain_adaptation/scored_original_tiser_sample/mixed_replay
```

## 9. Training Commands

Do not train unless `data/tennis/tennis_train_traced_full.json` exists and has validated TISER outputs. If it is missing:

```text
Do not train yet. Generate and validate TISER traces first.
```

Tennis-only smoke training:

```bash
python scripts/tennis/train_tennis.py \
  --config config/config_tennis_smoke.yaml \
  --train-file data/tennis/tennis_train_traced_full.json \
  --run-name tiser_tennis_smoke \
  --subset 50 \
  --epochs 1
```

Expected adapter:

```text
model/tiser_tennis_smoke/adapter
```

Tennis-only full training:

```bash
python scripts/tennis/train_tennis.py \
  --config config/config_tennis.yaml \
  --train-file data/tennis/tennis_train_traced_full.json \
  --run-name tiser_tennis_only_qwen7b \
  --output-dir outputs/tiser_tennis_only_qwen7b \
  --model-dir model/tiser_tennis_only_qwen7b
```

Expected adapter:

```text
model/tiser_tennis_only_qwen7b/adapter
```

Mixed replay dataset creation:

```bash
python scripts/tennis/build_mixed_replay_data.py \
  --tennis-train data/tennis/tennis_train_traced_full.json \
  --tiser-train data/TISER_train.json \
  --tiser-replay-size 500 \
  --seed 42 \
  --output data/tennis/tennis_mixed_replay_train.json \
  --summary results/tennis_domain_adaptation/processed/mixed_replay_summary.json
```

Mixed replay full training:

```bash
python scripts/tennis/train_tennis.py \
  --config config/config_tennis.yaml \
  --train-file data/tennis/tennis_mixed_replay_train.json \
  --run-name tiser_tennis_mixed_replay_qwen7b \
  --output-dir outputs/tiser_tennis_mixed_replay_qwen7b \
  --model-dir model/tiser_tennis_mixed_replay_qwen7b
```

Expected adapter:

```text
model/tiser_tennis_mixed_replay_qwen7b/adapter
```

After training, update the notebook adapter path variables to the actual adapter directories before running E2 or E3 evaluation.

## 10. Trace Generation / Validation

`scripts/tennis/generate_tennis_traces.py` does not call an external model. It prepares prompts and validates externally generated outputs.

Prepare generation requests:

```bash
python scripts/tennis/generate_tennis_traces.py \
  --mode prepare \
  --input data/tennis/tennis_train.json \
  --output results/tennis_domain_adaptation/generations/tennis_trace_generation_requests.jsonl
```

External generation JSONL format:

```json
{"question_id":"tennis_000002","output":"<reasoning>...</reasoning>\n<timeline>...</timeline>\n<reflection>...</reflection>\n<answer>Yes</answer>"}
```

Required fields:

- `question_id`: must match a base tennis training record.
- `output`: must contain `<reasoning>`, `<timeline>`, `<reflection>`, and `<answer>`.

Validate and filter external generations:

```bash
python scripts/tennis/generate_tennis_traces.py \
  --mode validate \
  --input-generations results/tennis_domain_adaptation/generations/tennis_trace_generations.jsonl \
  --base-records data/tennis/tennis_train.json \
  --output data/tennis/tennis_train_traced_full.json \
  --report results/tennis_domain_adaptation/generations/tennis_trace_validation_report.md \
  --summary results/tennis_domain_adaptation/generations/tennis_trace_validation_summary.json
```

The validation step writes:

- `data/tennis/tennis_train_traced_full.json`
- `results/tennis_domain_adaptation/generations/tennis_trace_validation_report.md`
- `results/tennis_domain_adaptation/generations/tennis_trace_validation_summary.json`

Only valid traces are retained. Invalid or answer-mismatched traces are excluded.

## 11. Adapter Comparison

Run after at least one scored condition has a `metrics.json` file:

```bash
python scripts/tennis/compare_adapters.py \
  --results-dir results/tennis_domain_adaptation
```

Expected outputs:

- `results/tennis_domain_adaptation/comparisons/adapter_comparison.json`
- `results/tennis_domain_adaptation/comparisons/adapter_comparison.md`
- `results/tennis_domain_adaptation/comparisons/per_category_comparison.csv`

This comparison reports EM, token-level F1, malformed count/rate, deltas against `original_tiser` when available, and per-category EM/F1.

## 12. Final Aggregation

Run final aggregation after E0, E1, E2, and E3 metrics exist.

```bash
python scripts/tennis/aggregate_tennis_results.py \
  --results-dir results/tennis_domain_adaptation
```

If the final aggregation files already exist and you intentionally want to regenerate them:

```bash
python scripts/tennis/aggregate_tennis_results.py \
  --results-dir results/tennis_domain_adaptation \
  --force
```

Expected outputs:

- `results/tennis_domain_adaptation/comparisons/final_results_table.md`
- `results/tennis_domain_adaptation/comparisons/final_results_table.csv`
- `results/tennis_domain_adaptation/comparisons/final_results_summary.json`
- `results/tennis_domain_adaptation/comparisons/forgetting_analysis.md`
- `results/tennis_domain_adaptation/comparisons/category_analysis.md`

The aggregation script looks for tennis-test metrics under:

- `results/tennis_domain_adaptation/scored/base_qwen/metrics.json`
- `results/tennis_domain_adaptation/scored/original_tiser/metrics.json`
- `results/tennis_domain_adaptation/scored/tennis_only/metrics.json`
- `results/tennis_domain_adaptation/scored/mixed_replay/metrics.json`

It also looks for original TISER sample metrics under `results/tennis_domain_adaptation/scored_original_tiser_sample/{condition}/metrics.json` when available.

## 13. How to Know If the Run Worked

After a successful full run, these files should exist:

```text
results/tennis_domain_adaptation/scored/base_qwen/predictions.jsonl
results/tennis_domain_adaptation/scored/base_qwen/metrics.json
results/tennis_domain_adaptation/scored/original_tiser/metrics.json
results/tennis_domain_adaptation/scored/tennis_only/metrics.json
results/tennis_domain_adaptation/scored/mixed_replay/metrics.json
results/tennis_domain_adaptation/scored/mixed_replay/metrics.json
results/tennis_domain_adaptation/comparisons/adapter_comparison.md
```

For final aggregation with the current script, `scored/mixed_replay/metrics.json` is the important E3 path.

Each evaluation folder should contain:

- `predictions.jsonl`
- `metrics.json`
- `metrics_report.md`
- `run_meta.json`

The `metrics.json` file should include:

- `overall.em`
- `overall.f1`
- `overall.malformed_rate`
- `per_category`
- `condition`
- `model_name`
- `adapter_dir`

## 14. Common Failures and Fixes

CUDA out of memory:

- Reduce `BATCH_SIZE` to `1`.
- Reduce `MAX_NEW_TOKENS` to `128` or `256`.
- Restart the runtime to clear memory.
- Prefer A100 or L4 for full 7B runs.

Adapter path missing:

- Check the preflight table.
- Copy the adapter to Colab/Drive.
- Update `ORIGINAL_TISER_ADAPTER`, `TENNIS_ONLY_ADAPTER`, or `MIXED_REPLAY_ADAPTER`.
- If the adapter has not been trained, run the relevant training step only after trace validation.

Wrong base model for adapter:

- 7B adapters must be loaded with `Qwen/Qwen2.5-7B-Instruct`.
- Do not use `config/config_tennis_smoke.yaml` with 7B adapters, because it uses `Qwen/Qwen2.5-0.5B-Instruct`.

Hugging Face authentication:

- Run `huggingface_hub.login()`.
- Confirm the account has access to the model.
- Restart the runtime after installing or logging in if imports or auth state behave inconsistently.

Missing `tennis_train_traced_full.json`:

- Do not train yet.
- Run trace prepare mode.
- Generate outputs externally.
- Run validate mode to write `data/tennis/tennis_train_traced_full.json`.

Malformed predictions:

- Inspect each condition's `predictions.jsonl` and `metrics_report.md`.
- Malformed outputs often mean the model did not produce parseable `<answer>` tags for TISER prompt style.
- Keep `MAX_NEW_TOKENS` high enough for the model to produce the full trace and answer.

Notebook path issues:

- Run `!pwd` after `%cd {PROJECT_ROOT}`.
- The current directory must contain `scripts/tennis/evaluate_tennis.py`.
- If using Drive, confirm the mounted path is exactly where the repo lives.

Colab disconnects:

- Save results under `results/tennis_domain_adaptation/` inside Drive-backed `PROJECT_ROOT`.
- Run conditions one at a time.
- Zip results after each expensive condition if runtime stability is poor.

Results not saved to Drive:

- Use a Drive-backed `PROJECT_ROOT`, or copy the zip to Drive at the end.
- In the notebook artifact cell, set `COPY_ZIP_TO_DRIVE = True` and verify `DRIVE_ZIP_TARGET`.

Final aggregation missing E3:

- Ensure `results/tennis_domain_adaptation/scored/mixed_replay/metrics.json` exists.
- Evaluate E3 with `--condition mixed_replay --output-dir results/tennis_domain_adaptation/scored/mixed_replay` before aggregation.

## 15. What Results to Put in the Report

Use these files for report-ready numbers:

- `results/tennis_domain_adaptation/comparisons/adapter_comparison.md`
- `results/tennis_domain_adaptation/comparisons/final_results_table.md`
- `results/tennis_domain_adaptation/comparisons/category_analysis.md`
- `results/tennis_domain_adaptation/comparisons/forgetting_analysis.md`
- `results/tennis_domain_adaptation/comparisons/error_analysis/error_analysis_report.md`, if available

Current completed tennis-test metrics:

| Result | n | EM | F1 | malformed |
| --- | ---: | ---: | ---: | ---: |
| `base_qwen_standard_test224` | 224 | 0.379 | 0.472 | 0 |
| `base_qwen_tiser_test224` | 224 | 0.375 | 0.420 | 5 |
| `tennis_only_full600_test224` | 224 | 0.464 | 0.516 | 0 |
| `original_tiser_qwen7b_test224` | 224 | 0.580 | 0.701 | 0 |
| `tennis_from_tiser_e2_lr0.0002_bs4_ga4_r16_a32_d0p05_20260616_104036_011` | 224 | 0.732 | 0.856 | 0 |

Report the main conditions:

- E0: `base_qwen`
- E1: `original_tiser`
- E2: `tennis_only`
- E3: `mixed_replay`

Use these metrics:

- tennis-test EM
- tennis-test token-level F1
- malformed rate
- per-category EM/F1
- forgetting delta on the original TISER sample, if `original_tiser_eval_sample.json` was evaluated

Do not claim improvement unless `tennis_only` or the mixed replay condition is actually higher than `original_tiser` on `tennis_test`.
