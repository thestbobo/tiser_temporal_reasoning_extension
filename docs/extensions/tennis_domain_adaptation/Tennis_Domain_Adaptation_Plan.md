# Tennis Domain Adaptation Plan

## 1. Objective

This extension studies whether supervised domain adaptation of TISER on tennis-domain temporal QA improves in-domain tennis temporal reasoning while preserving performance on the original general TISER temporal reasoning setting.

The target research question is:

> Does domain adaptation on tennis temporal QA improve in-domain tennis temporal reasoning without degrading general TISER temporal reasoning, and does it reduce failures on event-ordering relations?

The experiment should test five hypotheses:

- H1: Tennis-adapted TISER improves EM/F1 on tennis temporal QA compared with the original TISER adapter.
- H2: Tennis adaptation improves event-ordering and sequence questions more than simple yes/no questions.
- H3: Tennis-only adaptation may cause forgetting on original TISER in-domain splits.
- H4: Mixed training with tennis data plus a replay subset of original TISER data reduces forgetting.
- H5: TISER-style prompting remains stronger than standard answer-only prompting in the tennis domain.

The extension is intentionally narrow: it should measure adaptation effects, not introduce a new model architecture or rewrite the baseline pipeline.

Current result status: the dataset/split pipeline is complete, a 0.5B
standalone tennis subexperiment is complete, and the 7B original-TISER transfer
plus tennis-from-TISER continued-adaptation grid are complete. Mixed replay and
original-TISER forgetting evaluations are not yet reportable from the current
result database. For the canonical current numbers, see
`Current_Status_and_Next_Steps.md`.

## 2. Relation to TISER

The tennis extension should reuse the baseline TISER supervision format and model infrastructure. TISER is a supervised fine-tuning method where the model learns to emit structured temporal reasoning traces:

```text
<reasoning>...</reasoning>
<timeline>...</timeline>
<reflection>...</reflection>
<answer>...</answer>
```

For tennis examples, the same four-section output format should be used unless a condition explicitly compares against a standard answer-only prompt. The LoRA/QLoRA supervised fine-tuning infrastructure from the baseline should remain the primary training path:

- same base family: Qwen2.5 instruct model variants;
- same PEFT LoRA adapter mechanism;
- same prompt/completion masking assumptions from the baseline dataset loader;
- same deterministic generation settings for evaluation where possible;
- same EM/F1 scoring logic where the answer type permits reuse.

Tennis-specific code should live under `src/tennis/` and `scripts/tennis/`. Baseline TISER logic should not be modified until a concrete compatibility need is identified.

## 3. Dataset Scope

### Raw Schema

Each raw tennis temporal QA example should contain:

- `context`: short tennis-domain passage containing the temporal evidence.
- `question`: natural-language temporal question.
- `answer`: gold answer string.
- `category`: one category from the taxonomy in Section 4.
- `source`: provenance label, for example `synthetic_seed`, `manual`, `match_report`, or another auditable source name.
- `question_id`: stable unique identifier.

Current repository note: `data/tennis/raw/tennis.json` currently contains 1,122 examples with `context`, `question`, and `answer` only. The audit/conversion phase must add or derive `category`, `source`, and `question_id` before training or evaluation.

### Raw Dataset Audit

The raw tennis data was found at `data/tennis/raw/tennis.json` and copied to the canonical audit input path `data/tennis/raw/tennis_raw.json`. The audit script wrote the enriched dataset to `data/tennis/processed/tennis_raw_audited.json` and reports under `results/tennis_domain_adaptation/raw_audit/`.

- Number of examples: 1,122.
- Valid examples: 1,122.
- Malformed examples: 0.
- Duplicate count: 1 exact duplicate record after the first occurrence; 8 near-duplicate records across 14 near-duplicate pairs.
- Category distribution: `yes_no_before_after` 390, `which_first_last` 235, `immediate_before_after` 196, `duration_minutes` 147, `overlap_while_during` 115, `tournament_round_sequence` 19, `other_temporal` 17, `tennis_injury_or_medical` 3.
- Answer distribution: `Yes` 309, `No` 200, `45 minutes` 19, `12 minutes` 13, `7 minutes` 11, `15 minutes` 10, `9 minutes` 10, `The semifinal` 8, `25 minutes` 6; the remaining answers are mostly sparse event names or duration values and are listed in `tennis_raw_audit_summary.json`.
- Main limitations: raw records only include the minimal schema and no provenance/source field; category labels are rule-based and should be reviewed before training; near-duplicate detection uses text similarity rather than source/template groups; the audit does not verify factual correctness or answer entailment.

### TISER-Compatible Schema

The processed train/dev/test files should contain records compatible with the baseline TISER SFT/evaluation format:

- `dataset_name`: fixed value such as `tennis_temporal_qa`.
- `question_id`: copied from the audited raw record.
- `question`: question text.
- `answer`: normalized gold answer string.
- `prompt`: full user prompt, including context and instruction.
- `output`: gold TISER-style completion with `<reasoning>`, `<timeline>`, `<reflection>`, and `<answer>` tags.
- `category`: tennis temporal category.
- `source`: raw-data provenance.

For evaluation-only records, `output` can be omitted if the evaluation code only needs `prompt` and `answer`; for training records, `output` is required.

The prompt should include the context and question explicitly and should not rely on external tennis knowledge unless the condition is intentionally testing memory. A recommended prompt template is:

```text
Use the passage to answer the tennis temporal question. Produce the answer in the TISER format with reasoning, timeline, reflection, and answer sections.

Passage:
{context}

Question:
{question}
```

The final answer inside `<answer>` should be concise and should match the normalized gold answer as closely as possible.

### TISER Conversion

The audited raw records are converted by `scripts/tennis/build_tennis_data.py` using tennis-specific helpers in `src/tennis/schema.py`, `src/tennis/prompts.py`, and `src/tennis/build_dataset.py`.

Schema constants are defined for:

- raw records: `context`, `question`, `answer`;
- audited records: `context`, `question`, `answer`, `question_id`, `category`, `tags`, `source`;
- TISER train-style records: `dataset_name`, `question_id`, `question`, `answer`, `prompt`, `output`, `category`, `tags`, `source`;
- TISER test/dev-style records: `dataset_name`, `question_id`, `question`, `answer`, `prompt`, `category`, `tags`, `source`.

The TISER prompt template is:

```text
You are given a temporal context from the tennis domain and a question.
Reason only over the events explicitly stated in the context.
Use the temporal order, durations, overlaps, or tournament sequence when needed.

Temporal context:
{context}

Question:
{question}

Return the final answer in the required format.
```

The current conversion writes:

- `data/tennis/processed/tennis_all_tiser.json`;
- `data/tennis/processed/tennis_all_standard_prompt.json`;
- `results/tennis_domain_adaptation/processed/tennis_conversion_summary.json`.

With `--deterministic-output`, `tennis_all_tiser.json` includes conservative placeholder traces with `<reasoning>`, `<timeline>`, `<reflection>`, and `<answer>` sections. These traces are validation scaffolding only: they copy the gold answer into `<answer>` and do not yet provide high-quality, example-specific temporal reasoning. The audited file currently has no explicit `source` field, so converted records use `source: unknown` and the summary reports the defaulted source count.

### Trace Generation and Filtering

Tennis training should use generated, example-specific TISER traces rather than deterministic placeholder traces. The offline trace-generation pipeline lives in `src/tennis/trace_generation.py` and `scripts/tennis/generate_tennis_traces.py`.

Prepare generation requests with:

```bash
python scripts/tennis/generate_tennis_traces.py \
  --mode prepare \
  --input data/tennis/tennis_train.json \
  --output results/tennis_domain_adaptation/generations/tennis_trace_generation_requests.jsonl
```

The request JSONL contains `question_id`, `category`, `context`, `question`, `gold_answer`, and `generation_prompt`. Generation is intentionally external: this repository code does not call OpenAI, Gemini, Claude, or any other provider by default.

After external generation, save generated outputs as JSONL with `question_id` and `output`, then validate and filter:

```bash
python scripts/tennis/generate_tennis_traces.py \
  --mode validate \
  --input-generations results/tennis_domain_adaptation/generations/tennis_trace_generations.jsonl \
  --base-records data/tennis/tennis_train.json \
  --output data/tennis/tennis_train_traced_full.json \
  --report results/tennis_domain_adaptation/generations/tennis_trace_validation_report.md \
  --summary results/tennis_domain_adaptation/generations/tennis_trace_validation_summary.json
```

Filtering keeps only traces with non-empty outputs, all four required sections, an extractable `<answer>`, and a normalized final answer that matches the normalized gold answer. This answer-match filter is necessary because traces that end in the wrong final answer would teach the supervised model an incorrect target even if the reasoning text appears plausible.

Current traced training artifacts:

- `data/tennis/tennis_train_traced_50.json`: 50-trace smoke subset.
- `data/tennis/tennis_train_traced_full.json`: 600 validated traces used by
  the completed tennis-only/continued-adaptation runs.

### Dataset Splits

The converted TISER records were split with `scripts/tennis/split_tennis_data.py` using seed `42` and requested ratios `0.7/0.1/0.2`. Exact duplicate records are grouped before splitting so identical prompt/question/answer examples do not leak across train/dev/test.

Output files:

- `data/tennis/tennis_train.json`: 785 examples.
- `data/tennis/tennis_dev.json`: 113 examples.
- `data/tennis/tennis_test.json`: 224 examples.
- `results/tennis_domain_adaptation/processed/tennis_split_summary.json`.
- `results/tennis_domain_adaptation/processed/tennis_split_report.md`.

Category balance by split:

| Category | Train | Dev | Test |
| --- | ---: | ---: | ---: |
| `yes_no_before_after` | 273 | 39 | 78 |
| `which_first_last` | 165 | 23 | 47 |
| `immediate_before_after` | 137 | 20 | 39 |
| `duration_minutes` | 103 | 15 | 29 |
| `overlap_while_during` | 81 | 11 | 23 |
| `tournament_round_sequence` | 13 | 2 | 4 |
| `other_temporal` | 12 | 2 | 3 |
| `tennis_injury_or_medical` | 1 | 1 | 1 |

Duplicate leakage check: no exact duplicate leakage was detected across splits. The main split limitation is category imbalance: `tennis_injury_or_medical`, `other_temporal`, and `tournament_round_sequence` are too small for strong per-category conclusions. More examples should be generated or curated for those categories before final domain-adaptation claims.

## 4. Tennis Temporal Categories

Use the following closed category set:

- `yes_no_before_after`: binary before/after relation questions.
- `which_first_last`: questions asking which event/player/result came first or last.
- `immediate_before_after`: questions about the immediately preceding or following event.
- `duration_minutes`: questions requiring elapsed-time or match-duration reasoning in minutes.
- `overlap_while_during`: questions about events that overlap or occur during another event.
- `tournament_round_sequence`: questions about round order, progression, or tournament sequence.
- `tennis_ranking_or_date`: questions involving ranking chronology, dates, seedings, or ranking changes.
- `tennis_injury_or_medical`: questions involving injuries, medical timeouts, retirements, or treatment timing.
- `other_temporal`: valid temporal questions that do not fit the above categories.

Category assignment should be saved in processed data and propagated into predictions, scored outputs, metrics, and error-analysis tables.

## 5. Experimental Conditions

### E0: Base Qwen2.5 Model, No TISER Adapter

Evaluate the base instruct model with no LoRA adapter. This estimates tennis temporal QA performance from general pretraining/instruction tuning alone.

### E1: Original TISER Adapter

Completed for the 224-example tennis test set as
`original_tiser_qwen7b_test224` under
`results/tennis_from_tiser_experiments/scored/`. The original TISER LoRA
adapter transfers to tennis at EM 0.580 / F1 0.701 with zero malformed outputs.
Use this as the 7B transfer baseline when discussing continued tennis
adaptation.

### E2: Tennis-Only LoRA Adapter

Completed in two related forms:

- 0.5B standalone tennis-only `full600`: EM 0.464 / F1 0.516 on the
  224-example tennis test set, zero malformed outputs.
- 7B continued adaptation from the original TISER adapter: best canonical run
  `tennis_from_tiser_e2_lr0.0002_bs4_ga4_r16_a32_d0p05_20260616_104036_011`,
  EM 0.732 / F1 0.856 on the 224-example tennis test set, zero malformed
  outputs.

The 7B run is the stronger domain-adaptation result; the 0.5B run is useful as
a lightweight standalone subexperiment.

### E3: Mixed Adapter Trained on Tennis + TISER Replay

Not yet reportable from the current result database. After traced tennis data
and `data/TISER_train.json` exist on the run machine, train a LoRA adapter on
tennis train examples plus a replay subset of original TISER training data. The
replay subset should be documented by size, sampling seed, source split, and
category/dataset composition where available.

### E4: Standard Prompt vs TISER Prompt Comparison

Optional but recommended for H5. Compare answer-only prompting against TISER-style prompting for the same model/adapter where feasible:

- standard prompt: context + question + direct answer instruction;
- TISER prompt: context + question + structured reasoning/timeline/reflection/answer instruction.

This comparison can be run for E0, E1, and the final tennis-adapted adapters, but it should be clearly separated from adapter-training comparisons.

## 6. Evaluation Sets

- `tennis_test`: held-out tennis-domain temporal QA test set used for final in-domain reporting.
- `tennis_dev`: held-out development set used for smoke checks, prompt validation, early error analysis, and selecting non-training implementation choices.
- `original_TISER_sample`: fixed sample from original TISER in-domain evaluation splits used to estimate forgetting after tennis adaptation.
- optional category-balanced tennis subsets: subsets with approximately equal counts per tennis temporal category, used to reduce category-frequency confounds in H2 analysis.

Split creation should use stable seeds and should avoid near-duplicate leakage. If examples are generated from templates or repeated tennis narratives, split assignment should be grouped by template, match, tournament, player pair, or source cluster where those identifiers exist or can be derived.

## 7. Metrics

Use the following metrics for every applicable condition:

- Exact Match: normalized prediction answer equals normalized gold answer.
- Token-level F1: overlap F1 between normalized predicted answer and normalized gold answer.
- Malformed output rate: percentage of generations missing required TISER tags, containing multiple conflicting final answers, or failing answer extraction.
- Per-category EM/F1: EM and token-level F1 grouped by the categories in Section 4.
- Forgetting delta on original TISER sample: performance change on `original_TISER_sample` relative to the original TISER adapter.

Recommended forgetting formula:

```text
forgetting_delta = metric(E_adapted on original_TISER_sample) - metric(E1 on original_TISER_sample)
```

Negative values indicate degradation relative to the original TISER adapter. Report both EM and F1 deltas.

For event-ordering analysis, group these categories as ordering/sequence-focused:

- `yes_no_before_after`
- `which_first_last`
- `immediate_before_after`
- `tournament_round_sequence`

Then compare aggregate EM/F1 gains for this group against less sequence-focused categories.

### Tennis Evaluation Normalization

Tennis-domain scoring uses a tennis-specific normalizer in `src/tennis/normalize.py` and keeps the generic `src/eval/metrics.py` baseline scorer unchanged. The tennis scorer should be used for tennis prediction artifacts because the raw answer distribution contains ranking labels, duration answers, round names, and punctuation variants that the generic normalizer does not handle consistently.

The tennis normalizer:

- lowercases answers after Unicode NFKC normalization;
- normalizes curly apostrophes and Unicode dash variants;
- removes punctuation and English articles, then collapses whitespace;
- normalizes `No. 1`, `no 1`, `#1`, and `number 1` ranking answers to `1`;
- normalizes numeric `minute`/`minutes` duration answers to singular `minute`;
- lets answers such as `The semifinal` and `semifinal` match through article removal.

The normalizer deliberately avoids fuzzy player-name aliasing. For example, it should not collapse different people who share a token such as `Carlos Alcaraz` and `Juan Carlos Ferrero`. Remaining limitations, including scoreline-specific equivalences and accent-insensitive player matching, should be reported in tennis metrics outputs rather than patched into the generic evaluator.

## 8. Expected Result Tables

The final report should include at least these tables:

1. Dataset statistics table:
   - split;
   - number of examples;
   - category distribution;
   - answer-type distribution;
   - source distribution.

2. Main tennis test results:
   - condition;
   - prompt format;
   - tennis_test EM;
   - tennis_test F1;
   - malformed output rate.

3. Per-category tennis results:
   - condition;
   - category;
   - count;
   - EM;
   - F1;
   - delta vs E1.

4. Original TISER forgetting table:
   - condition;
   - original_TISER_sample EM;
   - original_TISER_sample F1;
   - EM delta vs E1;
   - F1 delta vs E1.

5. Mixed replay ablation table:
   - replay size or ratio;
   - tennis_test EM/F1;
   - original_TISER_sample EM/F1;
   - forgetting deltas.

6. Prompting comparison table:
   - condition;
   - standard prompt EM/F1;
   - TISER prompt EM/F1;
   - malformed output rate for TISER prompt;
   - TISER-minus-standard delta.

7. Error-analysis table:
   - error type;
   - affected category;
   - example count;
   - representative question_id;
   - short diagnosis.

### Final Experiment Matrix

The final core matrix is generated by `scripts/tennis/run_experiment_plan.py`. By default it verifies inputs, prints the commands, and writes a guarded runnable shell script to `results/tennis_domain_adaptation/comparisons/run_tennis_experiments.sh`; it does not run model evaluation unless `--execute` is passed.

```bash
python scripts/tennis/run_experiment_plan.py \
  --tennis-test data/tennis/tennis_test.json \
  --tiser-sample data/tennis/original_tiser_eval_sample.json \
  --original-tiser-adapter model/tiser_qwen7b_full/adapter \
  --tennis-adapter model/tiser_tennis_only_qwen7b/adapter \
  --mixed-adapter model/tiser_tennis_mixed_replay_qwen7b/adapter \
  --config config/config_tennis.yaml
```

The planned final evaluations are:

| ID | Condition | Evaluation set | Output folder |
| --- | --- | --- | --- |
| E0 | `base_qwen` | `tennis_test` | `results/tennis_domain_adaptation/scored/base_qwen/` |
| E1 | `original_tiser` | `tennis_test` | `results/tennis_domain_adaptation/scored/original_tiser/` |
| E2 | `tennis_only` | `tennis_test` | `results/tennis_domain_adaptation/scored/tennis_only/` |
| E3 | `mixed_replay` | `tennis_test` | `results/tennis_domain_adaptation/scored/mixed_replay/` |
| E1 | `original_tiser` | `original_tiser_sample` | `results/tennis_domain_adaptation/scored_original_tiser_sample/original_tiser/` |
| E2 | `tennis_only` | `original_tiser_sample` | `results/tennis_domain_adaptation/scored_original_tiser_sample/tennis_only/` |
| E3 | `mixed_replay` | `original_tiser_sample` | `results/tennis_domain_adaptation/scored_original_tiser_sample/mixed_replay/` |

Use `--limit N` for smoke runs and `--execute` only when intentionally launching evaluation. Existing completed result folders are skipped unless `--force` is passed to the planner or `FORCE=1` is set when running the generated shell script.

Completed result folders use more specific names than the original four-folder
plan. For report-ready numbers, prefer:

| Result family | Folder |
| --- | --- |
| 0.5B base standard | `results/tennis_domain_adaptation/scored/base_qwen_standard_test224/` |
| 0.5B base TISER prompt | `results/tennis_domain_adaptation/scored/base_qwen_tiser_test224/` |
| 0.5B tennis-only full600 | `results/tennis_domain_adaptation/scored/tennis_only_full600_test224/` |
| 7B original TISER transfer | `results/tennis_from_tiser_experiments/scored/original_tiser_qwen7b_test224/` |
| 7B best continued tennis adaptation | `results/tennis_from_tiser_experiments/scored/tennis_from_tiser_e2_lr0.0002_bs4_ga4_r16_a32_d0p05_20260616_104036_011/` |

Build the fixed original TISER sample before final forgetting runs:

```bash
python scripts/tennis/build_tiser_eval_sample.py \
  --input data/TISER_test.json \
  --output data/tennis/original_tiser_eval_sample.json \
  --per-split 100 \
  --seed 42
```

The sampler uses `dataset_name`/`split` values when available and otherwise samples globally with a warning. It writes `results/tennis_domain_adaptation/processed/tiser_eval_sample_summary.json`.

### Result Aggregation

Final report tables are produced by `scripts/tennis/aggregate_tennis_results.py`:

```bash
python scripts/tennis/aggregate_tennis_results.py
```

The aggregator reads the four tennis-test folders under `results/tennis_domain_adaptation/scored/` and original-sample forgetting folders under `results/tennis_domain_adaptation/scored_original_tiser_sample/` when present. Missing metrics are reported as `MISSING`, never as zero.
If these report files already exist, pass `--force` to regenerate them after new evaluations finish.

Outputs:

- `results/tennis_domain_adaptation/comparisons/final_results_table.md`
- `results/tennis_domain_adaptation/comparisons/final_results_table.csv`
- `results/tennis_domain_adaptation/comparisons/final_results_summary.json`
- `results/tennis_domain_adaptation/comparisons/forgetting_analysis.md`
- `results/tennis_domain_adaptation/comparisons/category_analysis.md`

`final_results_table` contains `condition`, `prompt_style`, `tennis_em`, `tennis_f1`, `tennis_malformed_rate`, `original_tiser_em`, `original_tiser_f1`, `forgetting_delta_em`, and `notes`. `category_analysis.md` reports per-category EM/F1 for every condition, deltas against `original_tiser`, best/worst categories, and a highlighted event-ordering group covering `yes_no_before_after`, `which_first_last`, `immediate_before_after`, and `tournament_round_sequence`.

### Error Analysis

After prediction files exist under `results/tennis_domain_adaptation/scored/{condition}/predictions.jsonl`, run:

```bash
python scripts/tennis/analyze_tennis_errors.py \
  --predictions-root results/tennis_domain_adaptation/scored \
  --output-dir results/tennis_domain_adaptation/comparisons/error_analysis
```

The analyzer is read-only with respect to predictions. It loads every available condition among `base_qwen`, `original_tiser`, `tennis_only`, and `mixed_replay`, groups rows by `question_id`, and compares correctness without treating missing conditions as failures.

Outputs:

- `results/tennis_domain_adaptation/comparisons/error_analysis/error_summary.json`
- `results/tennis_domain_adaptation/comparisons/error_analysis/error_analysis_report.md`
- `results/tennis_domain_adaptation/comparisons/error_analysis/fixed_by_tennis_only.jsonl`
- `results/tennis_domain_adaptation/comparisons/error_analysis/broken_by_tennis_only.jsonl`
- `results/tennis_domain_adaptation/comparisons/error_analysis/fixed_by_mixed_replay.jsonl`
- `results/tennis_domain_adaptation/comparisons/error_analysis/persistent_failures.jsonl`

The report includes condition-level overview metrics, category-wise error counts, malformed-output counts, qualitative examples fixed or harmed by tennis adaptation, examples fixed by mixed replay, persistent failures, and compact suspected failure-mode counts for temporal order reversal, duration arithmetic, overlap reasoning, answer formatting, malformed output, and memorization/context conflict. Qualitative examples include context, question, gold answer, all available condition predictions, raw-generation excerpts, category, and an automatic note. If no predictions exist yet, the script writes a missing-input report rather than inventing results. Pass `--force` to regenerate outputs after new predictions are available.

## 9. Implementation Roadmap

### Phase 1: Raw Data Audit

- Inspect `data/tennis/raw/tennis.json`.
- Validate required raw fields.
- Assign stable `question_id` values where missing.
- Assign `source` values.
- Label every example with exactly one category from Section 4.
- Detect duplicate or near-duplicate contexts/questions.
- Save audit summary under `results/tennis_domain_adaptation/raw_audit/`.

### Phase 2: Dataset Conversion

- Convert audited raw records into TISER-compatible records.
- Build prompts from context and question.
- Normalize gold answers.
- Preserve `category` and `source`.
- Save processed records under `data/tennis/processed/`.

### Phase 3: Trace Generation or Trace Construction

- Construct gold `<reasoning>`, `<timeline>`, `<reflection>`, and `<answer>` outputs for training.
- Prefer deterministic template-based traces for synthetic examples if the temporal relation is explicit.
- If model-assisted trace generation is used, record the generator model, prompt, temperature, seed, and review rules.
- Validate that all training outputs contain exactly one complete set of TISER tags.

### Phase 4: Split Creation

- Create `tennis_train`, `tennis_dev`, and `tennis_test` splits.
- Use a fixed random seed.
- Balance categories where possible.
- Prevent leakage from duplicated or template-related examples.
- Write split statistics to results and README notes.

### Phase 5: Baseline Evaluation

- Evaluate E0 and E1 on `tennis_dev` and `tennis_test`. The 7B E1 tennis-test
  evaluation is complete as `original_tiser_qwen7b_test224`.
- Evaluate E1 on `original_TISER_sample`.
- Save predictions, extracted answers, metrics, and malformed-output diagnostics.
- Use dev results only for implementation debugging, not final claims.

#### Tennis Evaluation Commands

Base model tennis smoke evaluation with a small model and no adapter:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis_smoke.yaml \
  --test-file data/tennis/tennis_test.json \
  --condition base_qwen \
  --no-adapter \
  --output-dir results/tennis_domain_adaptation/scored/base_qwen
```

Original TISER adapter tennis smoke evaluation. This uses the 7B tennis config because the adapter must match the base model it was trained from; `--limit 2` keeps it a smoke run:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --adapter-dir model/tiser_qwen7b_full/adapter \
  --condition original_tiser \
  --limit 2 \
  --max-new-tokens 64 \
  --batch-size 1 \
  --output-dir results/tennis_domain_adaptation/scored/original_tiser
```

Full tennis-test evaluation commands for the four core conditions:

```bash
python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --condition base_qwen \
  --no-adapter \
  --output-dir results/tennis_domain_adaptation/scored/base_qwen

python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --adapter-dir model/tiser_qwen7b_full/adapter \
  --condition original_tiser \
  --output-dir results/tennis_domain_adaptation/scored/original_tiser

python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --adapter-dir model/tiser_tennis_only_qwen7b/adapter \
  --condition tennis_only \
  --output-dir results/tennis_domain_adaptation/scored/tennis_only

python scripts/tennis/evaluate_tennis.py \
  --config config/config_tennis.yaml \
  --test-file data/tennis/tennis_test.json \
  --adapter-dir model/tiser_tennis_mixed_replay_qwen7b/adapter \
  --condition mixed_replay \
  --output-dir results/tennis_domain_adaptation/scored/mixed_replay
```

Adapter comparison aggregation after at least two condition folders contain `metrics.json`:

```bash
python scripts/tennis/compare_adapters.py \
  --results-dir results/tennis_domain_adaptation
```

Each condition folder writes `predictions.jsonl`, `metrics.json`, `metrics_report.md`, and `run_meta.json`. The comparison command writes `adapter_comparison.json`, `adapter_comparison.md`, and `per_category_comparison.csv` under `results/tennis_domain_adaptation/comparisons/`.

### Phase 6: Tennis-Only Training

- Train E2 using tennis train examples only. A 0.5B full600 adapter and a 7B
  continued-adaptation grid have already been evaluated; preserve their run
  metadata when updating report tables.
- Save the exact config, adapter path, run metadata, and training data manifest.
- Do not overwrite the original TISER adapter.
- Run the same tennis and original_TISER_sample evaluations as E1.

### Phase 7: Mixed Replay Training

- Build a replay subset from original TISER train data.
- Train E3 on tennis train examples plus replay.
- Save replay manifest with source ids and sampling seed.
- Optionally run small replay-ratio ablations if compute allows.

### Phase 8: Final Evaluation

- Evaluate E0, E1, E2, and E3 on all required evaluation sets.
- If E4 is included, run both standard and TISER prompts with matched decoding settings.
- Aggregate main metrics and per-category metrics.
- Compute forgetting deltas against E1.

### Phase 9: Error Analysis

- Inspect false positives/false negatives and malformed generations.
- Label error types, especially event-ordering failures.
- Compare ordering/sequence categories against yes/no and other categories.
- Track whether errors come from temporal relation mistakes, answer extraction failures, tennis terminology, arithmetic/duration computation, or prompt-format failure.

### Phase 10: Report Update

- Add methods, dataset statistics, results tables, and error analysis to the report.
- State limitations clearly, especially if the tennis data is synthetic or lightly audited.
- Include reproducibility instructions for rebuilding datasets and rerunning evaluations.

## 10. Reproducibility Checklist

Save the following artifacts for every experiment:

- configs:
  - training config;
  - evaluation config;
  - prompt-template version;
  - replay sampling config.
- processed datasets:
  - audited raw data with `question_id`, `category`, and `source`;
  - `tennis_train`;
  - `tennis_dev`;
  - `tennis_test`;
  - `original_TISER_sample`;
  - optional category-balanced subsets.
- predictions:
  - raw generations;
  - extracted answers;
  - gold answers;
  - question ids;
  - categories;
  - prompt format;
  - adapter condition.
- metrics:
  - overall EM/F1;
  - malformed output rate;
  - per-category EM/F1;
  - forgetting deltas;
  - prompt comparison deltas;
  - replay ablation metrics where applicable.
- run metadata:
  - model name;
  - adapter path;
  - checkpoint or commit identifier;
  - random seeds;
  - decoding parameters;
  - hardware notes;
  - timestamp;
  - script command.
- README notes:
  - data-build instructions;
  - evaluation instructions;
  - expected artifact locations;
  - known limitations and missing assumptions.

No full training, expensive scripts, deletion, or baseline TISER code changes are part of this planning step.
