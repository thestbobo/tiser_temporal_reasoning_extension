# Tennis Domain Adaptation Training Commands

This extension prepares two adapter conditions:

- E2: tennis-only adapter
- E3: mixed adapter trained on tennis plus original TISER replay

Do not treat results from placeholder tennis traces as scientific evidence. The
preferred train file is `data/tennis/tennis_train_traced.json`. If it is absent,
`scripts/tennis/train_tennis.py` falls back to `data/tennis/tennis_train.json`
for inspection, detects the placeholder-style outputs, and aborts by default.
Use `--allow-placeholder-traces` only for plumbing checks.

## E2 Tennis-Only Smoke Training

Use this only to check the training pipeline:

```bash
python scripts/tennis/train_tennis.py \
  --config config/config_tennis_smoke.yaml \
  --train-file data/tennis/tennis_train_traced.json \
  --run-name tiser_tennis_smoke \
  --subset 50 \
  --epochs 1
```

Adapter output:

```text
model/tiser_tennis_smoke/adapter
```

Run artifacts:

```text
outputs/tiser_tennis_smoke
```

## E2 Tennis-Only Full Training

```bash
python scripts/tennis/train_tennis.py \
  --config config/config_tennis.yaml \
  --train-file data/tennis/tennis_train_traced.json \
  --run-name tiser_tennis_only_qwen7b \
  --output-dir outputs/tiser_tennis_only_qwen7b \
  --model-dir model/tiser_tennis_only_qwen7b
```

Because the baseline trainer stores artifacts under `<base>/<run_name>`, the
tennis wrapper treats `outputs/tiser_tennis_only_qwen7b` and
`model/tiser_tennis_only_qwen7b` as explicit run directories and avoids creating
duplicate `tiser_tennis_only_qwen7b/tiser_tennis_only_qwen7b` paths.

Adapter output:

```text
model/tiser_tennis_only_qwen7b/adapter
```

Run artifacts:

```text
outputs/tiser_tennis_only_qwen7b
```

## E3 Mixed Replay Dataset Build

This preserves every tennis row, samples 500 original TISER rows with seed 42,
adds source metadata, shuffles deterministically, and writes a summary without
modifying `data/TISER_train.json`.

```bash
python scripts/tennis/build_mixed_replay_data.py \
  --tennis-train data/tennis/tennis_train_traced.json \
  --tiser-train data/TISER_train.json \
  --tiser-replay-size 500 \
  --seed 42 \
  --output data/tennis/tennis_mixed_replay_train.json \
  --summary results/tennis_domain_adaptation/processed/mixed_replay_summary.json
```

Do not build mixed replay from `data/tennis/tennis_train.json` for scientific
training. If `data/tennis/tennis_train_traced.json` is unavailable, the fallback
command below is only a plumbing check and still requires `data/TISER_train.json`:

```bash
python scripts/tennis/build_mixed_replay_data.py \
  --tennis-train data/tennis/tennis_train.json \
  --tiser-train data/TISER_train.json \
  --tiser-replay-size 500 \
  --seed 42 \
  --output data/tennis/tennis_mixed_replay_train.json \
  --summary results/tennis_domain_adaptation/processed/mixed_replay_summary.json
```

## E3 Mixed Replay Training

```bash
python scripts/tennis/train_tennis.py \
  --config config/config_tennis.yaml \
  --train-file data/tennis/tennis_mixed_replay_train.json \
  --run-name tiser_tennis_mixed_replay_qwen7b \
  --output-dir outputs/tiser_tennis_mixed_replay_qwen7b \
  --model-dir model/tiser_tennis_mixed_replay_qwen7b
```

Adapter output:

```text
model/tiser_tennis_mixed_replay_qwen7b/adapter
```

Run artifacts:

```text
outputs/tiser_tennis_mixed_replay_qwen7b
```

## Evaluation After Training

Evaluate E2:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --adapter-dir model/tiser_tennis_only_qwen7b/adapter \
  --condition tennis_only_qwen7b \
  --output-dir results/tennis_domain_adaptation/scored/tennis_only_qwen7b
```

Evaluate E3:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --adapter-dir model/tiser_tennis_mixed_replay_qwen7b/adapter \
  --condition tennis_mixed_replay_qwen7b \
  --output-dir results/tennis_domain_adaptation/scored/tennis_mixed_replay_qwen7b
```

Optional smoke evaluation:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis_smoke.yaml \
  --adapter-dir model/tiser_tennis_smoke/adapter \
  --condition tennis_smoke \
  --output-dir results/tennis_domain_adaptation/scored/tennis_smoke \
  --limit 20 \
  --batch-size 1 \
  --max-new-tokens 128
```

## Pre-Training Validation

`scripts/tennis/train_tennis.py` validates before invoking the trainer:

- The selected train file exists.
- Records are non-empty JSON objects from a JSON array or JSONL file.
- Every record has non-empty `prompt` and `output`.
- Every `output` contains `<answer>`.
- The extracted `<answer>` matches `answer`, `gold_answer`, or `gold` for at
  least 95% of records by default.
- Placeholder-style traces abort training by default.
- `--allow-placeholder-traces` permits placeholder traces only for
  plumbing/smoke checks; those runs are not scientifically valid.
