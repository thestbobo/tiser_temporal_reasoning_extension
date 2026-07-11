```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/tiser_temporal_reasoning_extension")
```

```python
%cd {PROJECT_ROOT}
```

```bash
pwd
ls
```

```bash
pip install -r requirements.txt
```

```bash
nvidia-smi
```

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

```bash
python scripts/tennis/run_experiment_plan.py \
  --config config/config_tennis.yaml \
  --tennis-test data/tennis/tennis_test.json \
  --original-tiser-adapter model/tiser_qwen7b_full/adapter \
  --tennis-adapter model/tiser_tennis_only_qwen7b/adapter \
  --mixed-adapter model/tiser_tennis_mixed_replay_qwen7b/adapter \
  --limit 5
```

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

```bash
python scripts/tennis/generate_tennis_traces.py \
  --mode prepare \
  --input data/tennis/tennis_train.json \
  --output results/tennis_domain_adaptation/generations/tennis_trace_generation_requests.jsonl
```

```bash
python scripts/tennis/generate_tennis_traces.py \
  --mode validate \
  --input-generations results/tennis_domain_adaptation/generations/tennis_trace_generations.jsonl \
  --base-records data/tennis/tennis_train.json \
  --output data/tennis/tennis_train_traced_full.json \
  --report results/tennis_domain_adaptation/generations/tennis_trace_validation_report.md \
  --summary results/tennis_domain_adaptation/generations/tennis_trace_validation_summary.json
```

```bash
python scripts/tennis/train_tennis.py \
  --config config/config_tennis_smoke.yaml \
  --train-file data/tennis/tennis_train_traced_full.json \
  --run-name tiser_tennis_smoke \
  --subset 50 \
  --epochs 1
```

```bash
python scripts/tennis/train_tennis.py \
  --config config/config_tennis.yaml \
  --train-file data/tennis/tennis_train_traced_full.json \
  --run-name tiser_tennis_only_qwen7b \
  --output-dir outputs/tiser_tennis_only_qwen7b \
  --model-dir model/tiser_tennis_only_qwen7b
```

```bash
python scripts/tennis/build_mixed_replay_data.py \
  --tennis-train data/tennis/tennis_train_traced_full.json \
  --tiser-train data/TISER_train.json \
  --tiser-replay-size 500 \
  --seed 42 \
  --output data/tennis/tennis_mixed_replay_train.json \
  --summary results/tennis_domain_adaptation/processed/mixed_replay_summary.json
```

```bash
python scripts/tennis/train_tennis.py \
  --config config/config_tennis.yaml \
  --train-file data/tennis/tennis_mixed_replay_train.json \
  --run-name tiser_tennis_mixed_replay_qwen7b \
  --output-dir outputs/tiser_tennis_mixed_replay_qwen7b \
  --model-dir model/tiser_tennis_mixed_replay_qwen7b
```

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

```bash
python scripts/tennis/compare_adapters.py \
  --results-dir results/tennis_domain_adaptation
```

```bash
python scripts/tennis/aggregate_tennis_results.py \
  --results-dir results/tennis_domain_adaptation
```

```bash
python scripts/tennis/aggregate_tennis_results.py \
  --results-dir results/tennis_domain_adaptation \
  --force
```

```python
from pathlib import Path
import shutil

results_path = Path("results/tennis_domain_adaptation")
shutil.make_archive(
    "tennis_domain_adaptation_results",
    "zip",
    root_dir=results_path.parent,
    base_dir=results_path.name,
)
```
