# Tennis Extension LaTeX Update Summary

## Sections Modified

- `abstract`: updated tennis-extension status to state that final E0-E3 adapter-comparison metrics are missing.
- `Problem Statement`: replaced the earlier tennis smoke framing with `Tennis Domain Adaptation for Temporal Reasoning`.
- `Methodology`: added tennis dataset audit/conversion/split description, TISER trace reuse, metrics, and E0-E3 experiment matrix.
- `Experiments and Results`: replaced the smoke-only result subsection with final-domain-adaptation placeholder tables.
- `Analysis`: clarified that current tennis artifacts support evaluation readiness, not adaptation claims.
- `Conclusions`: added the remaining E0-E3 final evaluation work and required result files.

## Tables Added

- `tab:tennis-data`: compact tennis split/category table.
- `tab:tennis-final`: final tennis domain-adaptation matrix with TODO metric placeholders.

## TODOs Remaining

- Fill final tennis-test EM/F1 and malformed-rate values for E0, E1, E2, and E3.
- Fill original-TISER-sample EM/F1 and forgetting deltas.
- Fill per-category tennis EM/F1.
- Add interpretation only after real metrics exist.

## Result Files Expected

- `results/tennis_domain_adaptation/comparisons/final_results_table.md`
- `results/tennis_domain_adaptation/comparisons/final_results_table.csv`
- `results/tennis_domain_adaptation/comparisons/category_analysis.md`
- `results/tennis_domain_adaptation/comparisons/forgetting_analysis.md`
- Prediction/metric folders under:
  - `results/tennis_domain_adaptation/scored/base_qwen/`
  - `results/tennis_domain_adaptation/scored/original_tiser/`
  - `results/tennis_domain_adaptation/scored/tennis_only/`
  - `results/tennis_domain_adaptation/scored/mixed_replay/`
  - `results/tennis_domain_adaptation/scored_original_tiser_sample/original_tiser/`
  - `results/tennis_domain_adaptation/scored_original_tiser_sample/tennis_only/`
  - `results/tennis_domain_adaptation/scored_original_tiser_sample/mixed_replay/`
