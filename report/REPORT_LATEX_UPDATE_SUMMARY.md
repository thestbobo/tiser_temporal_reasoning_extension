# Report LaTeX Update Summary

**Date:** 2025-06-05
**Files edited:** `report/report.tex`

---

## Sections Updated

| Section | Change |
|---|---|
| **Abstract** | Rewritten: includes final baseline result (macro-EM 0.878 / F1 0.949), Context–Memory Conflict extension summary, key finding (grounded reader not auditor), reflection mention rate 3.8%. ~250 words. |
| **Introduction** | Removed all future-tense placeholders ("will later present…"). Added extension framing: knowledge-conflict robustness probe, not in the original paper, tests whether reflection audits under adversarial conflict. |
| **Problem Statement** | Kept TISER formulation. Added §2.2 Context–Memory Conflict Probe: memory elicitation, eligibility, counterfactual construction, evaluation protocol. Defined: context-faithfulness, memorisation, conflict-detection, silent override. |
| **Methodology** | Compressed baseline methodology (all hyperparameters retained). Added §3.2 extension methodology: M0 subset construction, M1–M2 memory/eligibility, M3 perturbation (C1/C2/C3 + control), M4–M5 run matrix (2×2), M6 scoring. |
| **Experiments/Results** | Added Table 1 (baseline per-split EM). Added Table 2 (run matrix: 4 cells × faithful-EM/F1, memorised-EM, malformed, reflection-mention). Added Table 3 (per-conflict-class for star cell). |
| **Analysis** | New section. Baseline: performance regime, prompt-template lesson, EM brittleness. Extension: H1 supported (+0.213/+0.226), H2 silent override (3.8% vs 2.3% null), H4 supported (C3 blind spot), H3 deferred. Caveats paragraph. |
| **Conclusions** | Rewritten: reproduction succeeded, TISER improves temporal QA, prompt formatting is critical, extension shows faithfulness but not auditing, C3 weakness, future work. |
| **Reproducibility** | Expanded: config-driven pipeline, fixed seed, run_meta.json with git SHA/lib versions, per-stage artifacts under outputs/conflict/<stage>/, repo link. |

---

## Metrics Inserted (from JSON artifacts)

| Metric | Value | Source |
|---|---|---|
| Baseline macro-EM | 0.878 | Extension Report §3 (citing `baseline_full/.../metrics.json`) |
| Baseline macro-F1 | 0.949 | Extension Report §3 |
| Baseline per-split EM (L2/L3/TQe/TQh) | 0.907/0.961/0.980/0.970 | Extension Report §3 |
| tiser×tiser faithful-EM | 0.787 | `results/context_memory_conflict/scored/tiser__tiser.metrics.json` |
| tiser×tiser faithful-F1 | 0.917 | `results/context_memory_conflict/scored/tiser__tiser.metrics.json` |
| tiser×tiser memorised-EM | 0.230 | `results/context_memory_conflict/scored/tiser__tiser.metrics.json` |
| tiser×tiser malformed | 0.002 | `results/context_memory_conflict/scored/tiser__tiser.metrics.json` |
| tiser×tiser reflection-mention | 0.038 | `results/context_memory_conflict/scored/tiser__tiser.metrics.json` |
| tiser×standard faithful-EM | 0.574 | `results/context_memory_conflict/scored/tiser__standard.metrics.json` |
| base×tiser faithful-EM | 0.561 | `results/context_memory_conflict/scored/base__tiser.metrics.json` |
| base×tiser reflection-mention | 0.121 | `results/context_memory_conflict/scored/base__tiser.metrics.json` |
| base×standard faithful-EM | 0.380 | `results/context_memory_conflict/scored/base__standard.metrics.json` |
| C1 faithful-EM | 0.936 | `results/context_memory_conflict/scored/tiser__tiser.metrics.json` (per_class.C1) |
| C2 faithful-EM | 0.971 | `results/context_memory_conflict/scored/tiser__tiser.metrics.json` (per_class.C2) |
| C3 faithful-EM | 0.366 | `results/context_memory_conflict/scored/tiser__tiser.metrics.json` (per_class.C3) |
| C3 memorised-EM | 0.477 | `results/context_memory_conflict/scored/tiser__tiser.metrics.json` (per_class.C3) |
| Control faithful-EM | 0.900 | `results/context_memory_conflict/scored/tiser__tiser.metrics.json` (per_class.control) |
| Conflict set size | 1,176 | `results/context_memory_conflict/conflicts/conflict_report.json` |
| Eligible items | 571 | `results/context_memory_conflict/eligible/yield_report.json` |
| In-scope items | 15,898 | `results/context_memory_conflict/subset/subset_report.json` |

---

## Tables Added

1. **Table 1** — Baseline reproduction (per-split EM, macro EM/F1)
2. **Table 2** — Context–Memory Conflict run matrix (4 cells: faithful-EM/F1, memorised-EM, malformed, reflection-mention)
3. **Table 3** — Per-conflict-class results for the star cell (C1/C2/C3/control)

---

## Citations

No new citations added. Existing bibliography retained:
- Bazaga et al. 2025 (TISER)
- Xiong et al. 2024 (TGQA)
- Tan et al. 2023 (TempReason)
- Chen et al. 2021 (TimeQA)
- Hu et al. 2022 (LoRA)

---

## LaTeX Packages Added

- `booktabs` (for `\toprule`/`\midrule`/`\bottomrule` in tables)
- `multirow` (available for multi-row cells if needed)

---

## Known Caveats

- TGQA per-split EM and per-split F1 values are not individually listed in the baseline table (committed JSON artifact for the full baseline is in a gitignored `outputs/` directory not present locally; the macro and 4-of-5 per-split values come from `docs/extensions/context_memory_conflict/Extension_ContextMemoryConflict_Report.md` §3 which cites the full artifact).
- The original paper comparison uses the README-reported Qwen2.5-7B + TISER numbers (macro-EM 0.911 / macro-F1 0.944). The report explicitly states the result is "in the same performance regime," not superior.
- Conflict-detection is described as a lexical proxy throughout.
- H3 is explicitly marked as deferred, not failed.
- The Experiment Plan §1.1 mentions a slightly different baseline value (0.887/0.955) from an earlier evaluation; the report uses the authoritative full-test value (0.878/0.949) from the Extension Report §3.

---

## Validation Checklist

- [x] No TODO placeholders remain
- [x] No "will later present results" phrases remain
- [x] Repo link is explicit: `https://github.com/thestbobo/tiser_temporal_reasoning_extension`
- [x] Baseline result updated to full-test (macro-EM 0.878 / F1 0.949)
- [x] Extension is clearly framed as beyond the original paper
- [x] H1, H2, H4 are discussed
- [x] H3 is explicitly marked as deferred/future work
- [x] Run matrix table is present (Table 2)
- [x] Per-conflict-class table is present (Table 3)
- [x] Report does NOT claim reflection truly detects conflicts
- [x] Report says conflict detection is based on a lexical proxy
- [x] No unsupported numerical claims
- [x] LaTeX source is clean, readable, and academically polished
