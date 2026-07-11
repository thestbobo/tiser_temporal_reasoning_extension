# Tennis Domain Adaptation: Current Status and Next Steps

This document is the current source of truth for the tennis-domain result
database. Older planning docs may still describe intended paths, but the claims
below reflect the completed artifacts that should drive the report.

## Repository State

Dataset artifacts:

- `data/tennis/raw/tennis_raw.json`: 1,122 raw examples.
- `data/tennis/processed/tennis_all_tiser.json`: 1,122 converted TISER-style records.
- `data/tennis/tennis_train.json`: 785 examples.
- `data/tennis/tennis_dev.json`: 113 examples.
- `data/tennis/tennis_test.json`: 224 examples.
- `data/tennis/tennis_train_traced_50.json`: 50 externally generated/validated traces.
- `data/tennis/tennis_train_traced_full.json`: 600 externally generated/validated traces.

Result families:

- `results/tennis_domain_adaptation/scored/`: completed 0.5B base and
  tennis-only adapter evaluations.
- `results/tennis_from_tiser_experiments/scored/`: completed Qwen2.5-7B
  original-TISER transfer and tennis-from-TISER continued-adaptation grid
  evaluations.
- `results/tennis_from_tiser_experiments/comparisons/`: adapter-comparison
  summaries for the 7B tennis-from-TISER grid.

## Completed Tennis-Test Results

### 0.5B Standalone Tennis Subexperiment

These are the results used by the compact tennis table in the LaTeX report.

| Condition | Model | Prompt | n | EM | F1 | malformed |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `base_qwen_standard_test224` | Qwen2.5-0.5B-Instruct | standard | 224 | 0.379 | 0.472 | 0 |
| `base_qwen_tiser_test224` | Qwen2.5-0.5B-Instruct | TISER | 224 | 0.375 | 0.420 | 5 |
| `tennis_only_full600_test224` | Qwen2.5-0.5B-Instruct + tennis LoRA | TISER | 224 | 0.464 | 0.516 | 0 |

Interpretation:

- TISER-style prompting alone does not help the 0.5B base model on this tennis
  test set.
- The 0.5B tennis-only `full600` adapter improves over the 0.5B standard base
  prompt by +0.085 EM and +0.045 F1.
- This is a small-model subexperiment and should not be described as the full
  7B tennis-domain adaptation result.

### 7B Tennis-from-TISER Experiments

The original TISER adapter transfer and continued tennis adaptation were run.
Do not describe them as missing or future work.

| Condition | Model | Prompt | n | EM | F1 | malformed |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `original_tiser_qwen7b_test224` | Qwen2.5-7B-Instruct + original TISER LoRA | TISER | 224 | 0.580 | 0.701 | 0 |
| `tennis_from_tiser_e2_lr0.0002_bs4_ga4_r16_a32_d0p05_20260616_104036_011` | Qwen2.5-7B-Instruct + continued tennis LoRA | TISER | 224 | 0.732 | 0.856 | 0 |

Interpretation:

- The original 7B TISER adapter transfers to tennis above the 0.5B baselines.
- The best canonical 7B continued-adaptation run improves over the original
  TISER adapter by +0.152 EM and +0.155 F1 on the 224-example tennis test set.
- The grid contains additional 7B runs; use
  `results/tennis_from_tiser_experiments/comparisons/adapter_comparison.*` for
  the full ranking.

## Not Yet Supported

The following claims still need a completed, auditable result artifact before
they appear as final findings:

- Mixed tennis plus original-TISER replay results.
- Forgetting or preservation on an original TISER evaluation sample.
- Human-validated factual correctness of the synthetic tennis dataset beyond the
  existing schema, duplicate, split, and answer-normalization checks.

## Canonical Report Guidance

- If the report only presents the 0.5B table, explicitly label it as a
  preliminary 0.5B standalone tennis subexperiment.
- Treat original TISER-adapter transfer as completed: it has a 224-example
  result.
- Treat tennis-from-TISER continued adaptation as completed: the 7B grid exists,
  and the best canonical run is listed above.
- It is still correct to say mixed replay and original-TISER forgetting
  evaluation are not yet reportable unless their artifacts are restored from
  another machine.

## Next Steps

1. Decide whether the LaTeX report should include the completed 7B
   tennis-from-TISER results or explicitly scope the tennis section to the 0.5B
   subexperiment.
2. If mixed replay exists on another machine, copy its metrics and run metadata
   into the result database before making replay or forgetting claims.
3. Regenerate or update comparison summaries if result folders are renamed, so
   `final_results_table.*` and `adapter_comparison.*` agree on canonical
   condition names.
