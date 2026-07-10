# Tennis Domain Adaptation: Current Status and Next Steps

This extension is currently a tennis-domain temporal reasoning evaluation and
adapter-preparation pipeline. With the artifacts present in this repository, the
scientifically supported experiment is prompt-based Base Qwen evaluation on the
tennis test set.

## Repository State

Inspected paths:

- `data/tennis/`
  - Present: `tennis_train.json`, `tennis_dev.json`, `tennis_test.json`
  - Present raw data: `raw/tennis.json`, `raw/tennis_raw.json`
  - Present processed data: `processed/tennis_all_standard_prompt.json`,
    `processed/tennis_all_tiser.json`, `processed/tennis_raw_audited.json`
  - Missing: `tennis_train_traced.json`
  - Missing: `tennis_mixed_replay_train.json`
- `config/`
  - Present: `config.yaml`, `config_smoke.yaml`, `config_tennis.yaml`,
    `config_tennis_smoke.yaml`
- `scripts/tennis/`
  - Present: evaluation, training wrapper, data build, trace generation,
    result comparison, aggregation, diagnostics, and smoke-test scripts
- `src/tennis/`
  - Present: dataset construction, normalization, parser/scoring, prompts,
    schema, splits, and trace-generation helpers
- `results/tennis_domain_adaptation/`
  - Present directories: `comparisons/`, `generations/`, `metrics/`,
    `processed/`, `raw_audit/`, `scored/`
  - `scored/` currently contains only `.gitkeep`
- `model/`
  - Missing entirely in the inspected workspace
- `outputs/`
  - Missing entirely in the inspected workspace

Original TISER data is also missing from this checkout:

- Missing: `data/TISER_train.json`
- Missing: `model/tiser_qwen7b_full/adapter`

## Canonical Adapter Paths

Use these adapter paths consistently:

- Original TISER adapter: `model/tiser_qwen7b_full/adapter`
- Tennis-only adapter: `model/tiser_tennis_only_qwen7b/adapter`
- Mixed tennis plus TISER replay adapter:
  `model/tiser_tennis_mixed_replay_qwen7b/adapter`

The mixed-replay path above is canonical. Do not use
older mixed-replay adapter names that omit the `tiser_tennis_mixed_replay_qwen7b`
run namespace.

## What Works Now

The following is supported with the current repository artifacts:

- Tennis data loading from `data/tennis/tennis_test.json`
- Base Qwen evaluation through `scripts/tennis/evaluate_tennis.py`
- Standard direct-answer tennis prompt evaluation
- TISER-style tennis prompt evaluation
- Tennis answer parsing, normalization, exact match, token F1, and malformed
  answer accounting
- Writing `metrics.json`, `predictions.jsonl`, `metrics_report.md`, and
  `run_meta.json` under a chosen scored output directory
- Dry-run experiment planning for currently supported Base Qwen smoke runs

## Supported Experiments Now

Run these from the repository root. They do not use adapters.

Standard prompt, 100-example smoke:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --condition base_qwen_standard \
  --prompt-style standard \
  --no-adapter \
  --limit 100 \
  --output-dir results/tennis_domain_adaptation/scored/base_qwen_standard_smoke_100
```

TISER-style prompt, 100-example smoke:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --condition base_qwen_tiser \
  --prompt-style tiser \
  --no-adapter \
  --limit 100 \
  --output-dir results/tennis_domain_adaptation/scored/base_qwen_tiser_smoke_100
```

Generate the same command plan without executing:

```bash
python scripts/tennis/run_experiment_plan.py \
  --config config/config_tennis.yaml \
  --tennis-test data/tennis/tennis_test.json
```

This writes a shell script at:

```text
results/tennis_domain_adaptation/comparisons/run_tennis_experiments.sh
```

## Training Safety

`scripts/tennis/train_tennis.py` validates the selected training file before
calling the trainer. It now aborts by default if placeholder-style traces are
detected.

The current fallback file, `data/tennis/tennis_train.json`, contains placeholder
trace text. Training on it is not scientifically valid. It may only be used for
trainer plumbing checks with the explicit override:

```bash
python scripts/tennis/train_tennis.py \
  --config config/config_tennis_smoke.yaml \
  --allow-placeholder-traces
```

Do not use such runs for adapter-quality, domain-adaptation, transfer, or
forgetting claims.

## Blocked Experiments

Blocked until `data/tennis/tennis_train_traced.json` exists and contains
validated non-placeholder TISER-style outputs:

- Tennis-only adapter training
- Tennis-only adapter evaluation

Blocked until original TISER data and adapter artifacts are restored:

- Original TISER adapter evaluation on tennis
- Original TISER forgetting/sample evaluation
- Mixed tennis plus original TISER replay dataset build
- Mixed replay adapter training
- Mixed replay adapter evaluation

Specifically, do not build mixed replay until `data/TISER_train.json` exists.
Do not evaluate original TISER transfer until
`model/tiser_qwen7b_full/adapter` exists.

## Scientifically Valid Claims

Valid with current artifacts:

- The repository can evaluate Base Qwen on tennis temporal QA using standard and
  TISER-style prompts.
- The evaluation pipeline writes reproducible metrics, predictions, and run
  metadata.
- Parser and normalization behavior can be tested independently of model
  generation.

Not valid yet:

- Claims that the original TISER adapter transfers to tennis.
- Claims that tennis-only fine-tuning improves tennis temporal reasoning.
- Claims that mixed replay improves tennis performance or prevents forgetting.
- Claims based on training over placeholder traces.
- Any comparison among Base Qwen, original TISER adapter, tennis-only adapter,
  and mixed-replay adapter unless the missing adapters are restored or trained
  from validated data.

## Minimal Experiment Plan

Current supported path:

1. Run Base Qwen plus standard prompt on 100 tennis examples.
2. Run Base Qwen plus TISER-style prompt on the same 100 tennis examples.
3. Compare `metrics.json` and inspect malformed predictions in
   `predictions.jsonl`.

After traced tennis data exists:

1. Generate or restore `data/tennis/tennis_train_traced.json`.
2. Validate that traces are non-placeholder and answers match gold.
3. Train `model/tiser_tennis_only_qwen7b/adapter`.
4. Evaluate the tennis-only adapter on `data/tennis/tennis_test.json`.

After original TISER data and adapter exist:

1. Restore `data/TISER_train.json`.
2. Restore `model/tiser_qwen7b_full/adapter`.
3. Optionally build `data/tennis/original_tiser_eval_sample.json`.
4. Evaluate the original TISER adapter.
5. Build `data/tennis/tennis_mixed_replay_train.json`.
6. Train `model/tiser_tennis_mixed_replay_qwen7b/adapter`.
7. Evaluate mixed replay and compare against the available baselines.
