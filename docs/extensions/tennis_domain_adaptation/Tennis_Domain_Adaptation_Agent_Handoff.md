# Agent Handoff: TISER Temporal Reasoning Codebase and Tennis Domain Adaptation Extension

This document is for the next agent/developer who will build the second extension of this Deep Natural Language Processing project. The new extension is expected to stay in the same research area, temporal reasoning in language models, but shift the focus to domain adaptation in tennis.

The repository currently contains:

- a faithful TISER baseline reproduction for temporal reasoning supervised fine-tuning;
- a completed first extension about context-memory conflict and faithfulness under counterfactual temporal contexts;
- committed result artifacts for that extension;
- reports and notebooks for reproducibility and write-up support.

The most important point for the next extension: reuse the existing TISER data/model/evaluation infrastructure, but add a tennis-specific temporal QA/domain-adaptation layer instead of rewriting the training stack.

## 1. Repository Identity

Project root:

```text
tiser_temporal_reasoning_extension/
```

Package name in `pyproject.toml`:

```text
tiser-baseline
```

Python requirement:

```text
>=3.10
```

The repository is built around the paper:

```text
TISER: Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models
Bazaga, Blloshmi, Byrne, de Gispert, ACL 2025
```

The baseline trains `Qwen/Qwen2.5-7B-Instruct` with QLoRA/LoRA supervised fine-tuning on the released TISER train data. The target output format is a structured four-section trace:

```text
<reasoning> ... </reasoning>
<timeline> ... </timeline>
<reflection> ... </reflection>
<answer> ... </answer>
```

The core idea is not a new model architecture. It is ordinary next-token supervised fine-tuning where the model learns to produce an explicit temporal reasoning trace, timeline, self-reflection, and short final answer.

## 2. High-Level Layout

```text
config/
  config.yaml
  config_smoke.yaml

scripts/
  fetch_data.py
  train.py
  evaluate.py

src/
  data/
  model/
  train/
  inference/
  eval/
  utils/

notebooks/
  colab_run.ipynb
  colab_smoke.ipynb

results/
  subset/
  memory/
  eligible/
  conflicts/
  run_inputs/
  generations/
  scored/

report/
  report.tex
  REPORT_LATEX_UPDATE_SUMMARY.md

README.md
docs/extensions/context_memory_conflict/Extension_ContextMemoryConflict_Report.md
docs/extensions/context_memory_conflict/Extension_ContextMemoryConflict_ExperimentPlan.md
```

There are two conceptual layers:

1. Baseline package code under `src/` and `scripts/`.
2. First-extension reports/results under `Extension_ContextMemoryConflict_*.md` and `results/`.

The checked-in `src/` tree does not currently include the first-extension pipeline modules mentioned in the report, such as `src/conflict/subset.py`, `src/conflict/memory.py`, or `src/conflict/score.py`. The results are committed, but those conflict-specific source modules appear to have been run elsewhere or not included in this checkout. Treat the committed result files as evidence/artifacts, and treat `src/` as the reusable baseline implementation.

## 3. Baseline Pipeline

The baseline has three CLI entrypoints:

```bash
python scripts/fetch_data.py --config config/config.yaml
python scripts/train.py --config config/config.yaml
python scripts/evaluate.py --config config/config.yaml
```

For a small smoke run:

```bash
python scripts/fetch_data.py --config config/config_smoke.yaml
python scripts/train.py --config config/config_smoke.yaml
python scripts/evaluate.py --config config/config_smoke.yaml
```

The full config is intended for substantial GPU compute. The README says the full 60k-example training run and 22k-example evaluation realistically need an A100-class GPU. `config_smoke.yaml` uses Qwen2.5-1.5B, 300 train examples, 1 epoch, and 20 eval samples per split for pipeline validation.

## 4. Configuration

Main config: `config/config.yaml`

Important fields:

- `run_name`: output namespace, default `tiser_qwen7b_full`.
- `paths.data_dir`: where TISER JSON files are stored.
- `paths.train_file`: `data/TISER_train.json`.
- `paths.test_file`: `data/TISER_test.json`.
- `paths.model_dir`: where LoRA adapters are saved.
- `paths.output_dir`: where predictions, metrics, and run metadata are written.
- `model.name`: default `Qwen/Qwen2.5-7B-Instruct`.
- `model.load_in_4bit`: default `true`, enabling QLoRA.
- `lora`: rank 16, alpha 32, dropout 0.05, target modules `[q_proj, k_proj, v_proj, o_proj]`.
- `train.max_seq_len`: 2048.
- `train.num_epochs`: 3.
- `eval.max_new_tokens`: 768 in the full config.
- `eval.do_sample`: `false`, so evaluation is greedy/deterministic.
- `splits`: in-domain test split whitelist for macro averaging.

Path resolution is handled in `src/utils/config.py`: config paths are resolved relative to the repository root so the same config works from local or Colab working directories.

## 5. Dataset Loading and Formatting

Main file: `src/data/dataset.py`

The repository expects the released TISER files:

```text
TISER_train.json
TISER_test.json
```

Training records are expected to contain:

```text
dataset_name, question_id, question, answer, prompt, output
```

Test records are expected to contain:

```text
dataset_name, question_id, question, prompt, answer
```

Important implementation detail:

- The training dataset is pre-tokenized manually.
- The code applies the model chat template to a user prompt plus assistant completion.
- Labels are masked up to the assistant header with `-100`.
- Only the gold completion trace contributes to loss.
- Packing is disabled because packing would break per-example completion-only masking.

This matters for the tennis extension. If adding tennis-domain SFT data, keep the same prompt/completion structure and same four-tag completion format unless the experiment explicitly studies alternative formats.

## 6. Data Fetching

Main file: `src/data/fetch.py`

The fetcher:

1. checks whether real local files already exist in `data/`;
2. tries to copy from `paths.local_source_dir`;
3. falls back to Hugging Face dataset download from `AmazonScience/TISER`;
4. falls back to git-lfs clone from `https://github.com/amazon-science/TISER`;
5. validates that required keys exist.

It rejects tiny git-lfs pointer files by checking file size and pointer header.

For a tennis extension, either:

- add a separate fetch/build script for tennis temporal data, or
- generalize the fetcher so it can fetch both TISER and tennis-domain files.

Do not overload `TISER_train.json`/`TISER_test.json` with tennis data unless the experiment is intentionally mixed-domain and clearly records provenance.

## 7. Model Loading and LoRA

Main file: `src/model/loader.py`

The model stack supports:

- QLoRA with 4-bit nf4 quantization through `bitsandbytes`;
- bf16/fp16 base loading when 4-bit is disabled;
- PEFT LoRA adapters;
- automatic compute dtype: bf16 if CUDA bf16 is supported, otherwise fp16;
- right padding during training and left padding during batched generation.

Training path:

```text
load_model_and_tokenizer()
  -> load tokenizer
  -> load base model
  -> prepare for k-bit training if needed
  -> return model/tokenizer
```

Inference path:

```text
load_adapter_for_inference(cfg, adapter_dir)
  -> load tokenizer
  -> load base model
  -> attach PEFT adapter
  -> set eval mode
  -> clear Qwen sampling defaults for greedy generation
```

For tennis domain adaptation, the likely model conditions are:

- base Qwen2.5-7B-Instruct;
- original TISER adapter;
- tennis-adapted adapter;
- possibly mixed TISER+tennis adapter if catastrophic forgetting is part of the research question.

## 8. Training

Main file: `src/train/trainer.py`

Training uses TRL `SFTTrainer`.

Important details:

- `DataCollatorForSeq2Seq` is used because labels are already masked.
- `label_pad_token_id=-100` preserves completion-only loss masking.
- `packing=False`.
- output trainer logs/checkpoints go under:

```text
outputs/<run_name>/trainer/
```

- final LoRA adapter goes under:

```text
model/<run_name>/adapter/
```

The CLI `scripts/train.py` supports overrides:

```bash
python scripts/train.py --config config/config.yaml --subset 2000
python scripts/train.py --config config/config.yaml --epochs 1
python scripts/train.py --config config/config.yaml --output-dir outputs_custom
```

For the tennis extension, create a separate config such as:

```text
config/config_tennis.yaml
config/config_tennis_smoke.yaml
```

Suggested run names:

```text
tiser_tennis_adapt_qwen7b
tiser_tennis_adapt_smoke
```

## 9. Inference and Parsing

Generation file: `src/inference/generate.py`

Parser file: `src/inference/parser.py`

Generation wraps every prompt in the same chat template used during training:

```text
user prompt -> assistant generation prompt
```

The model then generates the full four-section trace in one pass. There is no iterative reasoning loop.

Answer extraction is defensive:

- prefer the last well-formed `<answer> ... </answer>` block;
- if the closing tag is missing, use everything after the last opening `<answer>`;
- if no answer tag exists, return empty answer and mark malformed.

For tennis, keep this parser if the output format stays TISER-compatible. If tennis questions require structured answers such as scorelines, tournament rounds, dates, or player names, add normalization in evaluation rather than changing the parser first.

## 10. Evaluation

Main files:

```text
src/eval/metrics.py
src/eval/run_eval.py
```

Metrics:

- SQuAD-style normalized exact match;
- token F1;
- per-split aggregation;
- unweighted macro average over configured split whitelist;
- malformed output count.

Normalization currently:

- lowercases;
- removes punctuation;
- removes English articles;
- collapses whitespace.

Known limitation from the first extension report: Unicode/mojibake and dash variants can affect EM. For tennis, this can matter for player names, tournament names, and score strings. Add Unicode normalization before serious scoring.

The evaluation output goes to:

```text
outputs/<run_name>/predictions.jsonl
outputs/<run_name>/metrics.json
outputs/<run_name>/run_meta.json
```

Each prediction row includes:

```text
question_id
dataset_name
gold
raw_generation
pred_answer
malformed
em
f1
```

## 11. Utilities and Reproducibility

Utility files:

```text
src/utils/config.py
src/utils/io.py
src/utils/seeding.py
```

The code records:

- timestamp;
- git SHA when available;
- library versions for `torch`, `transformers`, `peft`, and `trl`;
- resolved config.

Seeding covers:

- Python `random`;
- NumPy;
- PyTorch CPU/CUDA;
- Transformers;
- deterministic cuDNN setting.

For the tennis extension, preserve the run metadata pattern. Domain adaptation experiments become hard to interpret if dataset version, adapter path, and prompt style are not recorded.

## 12. Completed First Extension: Context-Memory Conflict

The first extension asks whether a TISER-trained model follows the temporal context it reads or the factual memory it already has.

Main write-up:

```text
docs/extensions/context_memory_conflict/Extension_ContextMemoryConflict_Report.md
```

Design/decision record:

```text
docs/extensions/context_memory_conflict/Extension_ContextMemoryConflict_ExperimentPlan.md
```

Core experimental idea:

1. Start from temporal QA items.
2. Remove the context and ask the model closed-book.
3. Keep only items where the model confidently remembers the correct answer.
4. Edit the temporal context so it now supports a different answer.
5. Run a 2x2 matrix:

```text
model:  base Qwen vs TISER fine-tune
prompt: standard prompt vs TISER four-tag prompt
```

Conflict classes:

- `C1`: date-shift, moving dates so another entity covers the queried time.
- `C2`: entity-swap, replacing the answer entity with a distractor.
- `C3`: order-reversal, swapping temporal intervals so a before/after relation flips.
- `control`: row reorder without answer change.

Committed result folders:

```text
results/context_memory_conflict/subset/
results/context_memory_conflict/memory/
results/context_memory_conflict/eligible/
results/context_memory_conflict/conflicts/
results/context_memory_conflict/run_inputs/
results/context_memory_conflict/generations/
results/context_memory_conflict/scored/
```

Important counts:

- in-scope parsed items: 15,898;
- eligible confident-and-correct memory items: 571;
- conflict rows built: 1,176;
- classes: C1 203, C2 520, C3 333, control 120.

Headline scored result for the star cell, `tiser` model with `tiser` prompt:

```text
faithful EM: 0.787
faithful F1: 0.917
memorised EM: 0.230
malformed rate: 0.002
reflection mention rate: 0.038
```

Full 2x2 faithful EM:

```text
tiser model + tiser prompt:     0.787
tiser model + standard prompt:  0.574
base model  + tiser prompt:     0.561
base model  + standard prompt:  0.380
```

Main conclusion:

- TISER makes the model much more context-faithful.
- The `<reflection>` step usually does not explicitly notice the contradiction.
- Date-shift and entity-swap are mostly handled.
- Order-reversal is the weakness: the fine-tuned model often falls back to memorized event order.

This is directly relevant to tennis domain adaptation because tennis has many temporal-ordering questions:

- before/after tournament wins;
- ranking changes over seasons;
- head-to-head records at a date;
- title holder before a later champion;
- player/coaching/sponsor/team changes over intervals;
- injuries or returns across seasons.

The next extension should explicitly test whether domain adaptation improves or worsens this kind of temporal ordering.

## 13. Current Reports

The repository has two main human-facing reports:

```text
docs/extensions/context_memory_conflict/Extension_ContextMemoryConflict_Report.md
docs/extensions/context_memory_conflict/Extension_ContextMemoryConflict_ExperimentPlan.md
```

The report says the first extension is closed experimentally and now in analysis/write-up mode. It documents hypotheses H1-H4:

- H1: TISER prompt improves context faithfulness.
- H2: reflection names the conflict.
- H3: entity fame predicts faithfulness.
- H4: faithfulness depends on conflict type.

Verdicts:

- H1 supported.
- H2 resolved as "silent override": faithful answers usually do not explicitly identify the contradiction.
- H3 deferred.
- H4 supported, with C3 order-reversal as the failure mode.

There is also a LaTeX report under:

```text
report/report.tex
```

Use it only as write-up context. The implementation source of truth is still `src/`, `scripts/`, configs, and result JSON files.

## 14. Environment and Dependencies

Dependencies are pinned in `requirements.txt`.

Important packages:

```text
transformers==4.46.3
trl==0.12.2
peft==0.13.2
bitsandbytes>=0.45.3
accelerate==1.1.1
datasets==3.1.0
huggingface_hub==0.26.2
pyyaml==6.0.2
tqdm==4.67.1
```

Torch is intentionally not pinned in `requirements.txt`. On Colab, use the bundled torch to avoid CUDA/bitsandbytes conflicts. Locally, install a torch build compatible with the machine CUDA version.

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

## 15. What the Tennis Domain Adaptation Extension Should Do

The new extension should study whether adapting TISER to tennis-domain temporal language improves temporal reasoning in that domain, and whether it preserves general temporal reasoning.

Recommended research question:

```text
Does domain adaptation on tennis temporal QA improve in-domain tennis temporal reasoning without degrading general TISER temporal reasoning, and does it reduce failures on event-ordering relations?
```

Recommended hypotheses:

- H1: Tennis-adapted TISER improves EM/F1 on tennis temporal QA compared with the original TISER adapter.
- H2: Tennis adaptation improves temporal ordering questions more than simple entity/date lookup questions.
- H3: Tennis adaptation causes some forgetting on original TISER in-domain splits unless mixed with replay data.
- H4: A TISER-style prompt remains more effective than a plain prompt in the tennis domain.

## 16. Tennis Dataset Design

Build a tennis-domain temporal QA dataset with the same broad shape as TISER:

```json
{
  "dataset_name": "tennis_temporal_test",
  "question_id": "...",
  "question": "...",
  "answer": "...",
  "prompt": "...",
  "output": "..."
}
```

For training rows, include `output`. For test rows, `output` is not required by the current loader, but keeping it in a separate dev file can help debugging.

Suggested tennis temporal categories:

- `tennis_ranking_at_date`: player ranking or rank band at a date.
- `tennis_title_holder`: who held a title/tournament championship in a year.
- `tennis_before_after_match`: which event/player/tournament came before or after another.
- `tennis_head_to_head_until_date`: head-to-head leader before a date.
- `tennis_coach_interval`: who coached a player during a date interval.
- `tennis_injury_return`: player availability/status across a season.
- `tennis_tournament_surface_change`: tournament surface/location/sponsor over time.
- `tennis_grand_slam_sequence`: order of finals, wins, losses, or opponents.

Recommended answer types:

- player names;
- tournament names;
- dates/years;
- ranking numbers;
- short categorical values such as `hard`, `clay`, `grass`;
- scorelines only if a robust normalizer is added.

Keep contexts compact and explicit. A good tennis prompt should include a timeline table or list of dated facts and a question that requires choosing the correct fact at a date or relative to another event.

Example prompt shape:

```text
You are given a temporal context and a question. Reason over the dated events.

Temporal context:
- 2019-06-10 to 2019-07-14: Player A was ranked No. 2.
- 2019-07-15 to 2019-09-08: Player A was ranked No. 1.
- 2019-09-09 to 2019-11-03: Player A was ranked No. 3.

Question: What was Player A's ranking on 2019-08-01?
```

Expected TISER-style completion:

```text
<reasoning>
The question asks for the ranking on 2019-08-01. That date falls inside the interval from 2019-07-15 to 2019-09-08.
</reasoning>
<timeline>
2019-06-10 to 2019-07-14: ranked No. 2
2019-07-15 to 2019-09-08: ranked No. 1
2019-09-09 to 2019-11-03: ranked No. 3
</timeline>
<reflection>
The selected interval contains the queried date, so the answer is consistent with the timeline.
</reflection>
<answer>No. 1</answer>
```

## 17. Recommended Implementation Plan for Tennis

Add new code without disturbing the existing baseline:

```text
src/tennis/
  __init__.py
  schema.py
  build_dataset.py
  prompts.py
  splits.py
  normalize.py
  eval.py

scripts/tennis/
  build_tennis_data.py
  train_tennis.py
  evaluate_tennis.py
  compare_adapters.py

config/
  config_tennis.yaml
  config_tennis_smoke.yaml
```

Minimum viable path:

1. Build or ingest tennis temporal facts.
2. Convert them into TISER-compatible prompt/output records.
3. Fine-tune a LoRA adapter from the same Qwen base using tennis training data.
4. Evaluate three adapters/conditions:

```text
base Qwen2.5-7B-Instruct
original TISER adapter
tennis-adapted adapter
```

5. Evaluate on:

```text
tennis test set
original TISER test subset
```

6. Report:

```text
tennis EM/F1
original TISER EM/F1
per-category tennis scores
malformed rate
```

7. Add a prompt-axis comparison if time permits:

```text
standard prompt vs TISER four-tag prompt
```

## 18. Avoiding Methodological Problems

Be careful about leakage. Tennis facts are likely memorized by the base model, especially Grand Slam winners and famous rankings. Decide whether the experiment is:

- context-grounded tennis reasoning, where the answer must be derived from supplied context;
- closed-book tennis knowledge;
- conflict between tennis memory and supplied context;
- domain adaptation for specialized temporal language.

Those are different experiments.

Recommended controls:

- Evaluate on synthetic or less-famous tennis facts where memory is less useful.
- Include date-shift and order-reversal tennis cases.
- Include original TISER evaluation to measure forgetting.
- Keep a held-out tennis test set by player/tournament/year, not only by random rows.
- If generated with an LLM, validate gold answers programmatically where possible.

Suggested split strategies:

- train on seasons up to year N, test on later years;
- train on some tournaments, test on held-out tournaments;
- train on ATP, test on WTA or vice versa;
- train on famous players, test on less-famous players;
- train on lookup/date-binding, test on before/after ordering.

## 19. Evaluation Improvements Recommended Before Tennis Results

Add a stronger normalizer for tennis:

- Unicode normalize with `unicodedata.normalize("NFKC", text)`;
- normalize dash variants;
- normalize apostrophes;
- optionally strip rank prefixes such as `No.`, `#`, `number`;
- normalize player-name aliases carefully;
- handle scoreline punctuation separately if scorelines are used.

Do not over-normalize names. For example, `Carlos Alcaraz` and `Juan Carlos Ferrero` must remain distinct.

Add category-level aggregation:

```text
dataset_name or qtype -> EM/F1
```

The current `aggregate()` function groups by `dataset_name`. Tennis rows should encode meaningful dataset names or add a `qtype` field and implement a tennis-specific aggregation.

## 20. Suggested Experiment Matrix

Model/adaptation axis:

```text
base
tiser_original
tennis_only
mixed_tiser_tennis
```

Prompt axis:

```text
standard
tiser_four_tag
```

Evaluation axis:

```text
tennis_in_domain
tennis_heldout_temporal
original_tiser_subset
order_reversal_stress
```

A compact first run:

```text
original TISER adapter + TISER prompt on tennis test
tennis adapter + TISER prompt on tennis test
original TISER adapter + TISER prompt on original TISER sampled eval
tennis adapter + TISER prompt on original TISER sampled eval
```

This directly tests adaptation gain and forgetting.

## 21. Files to Reuse Directly

Reuse these as-is unless there is a specific reason to change them:

```text
src/model/loader.py
src/train/trainer.py
src/inference/generate.py
src/inference/parser.py
src/utils/config.py
src/utils/io.py
src/utils/seeding.py
```

Likely extend or wrap:

```text
src/data/dataset.py
src/eval/metrics.py
src/eval/run_eval.py
scripts/train.py
scripts/evaluate.py
```

Do not mix tennis-specific assumptions into generic baseline files unless the abstraction is clean. Prefer tennis-specific modules under `src/tennis/`.

## 22. Known Gaps in This Checkout

The first-extension report refers to modules and scripts that are not present in the checked-in `src/` tree:

```text
src/conflict/subset.py
src/conflict/memory.py
src/conflict/eligibility.py
src/conflict/perturb.py
src/conflict/prompts.py
src/conflict/run.py
src/conflict/score.py
scripts/conflict/01_*.py ... 06_*.py
config/conflict.yaml
```

The corresponding artifacts exist under `results/`, so the experiment can be understood and reported, but reproducing it from this checkout would require recovering or reimplementing those modules.

For the tennis extension, commit the source modules as well as result summaries. Future agents should not have to infer the pipeline from JSON artifacts.

## 23. Practical First Tasks for the Next Agent

1. Create `config/config_tennis_smoke.yaml` and point it at a tiny tennis JSON dataset.
2. Add `src/tennis/prompts.py` to produce TISER-style tennis prompts and completions.
3. Add `src/tennis/normalize.py` with Unicode-aware tennis answer normalization.
4. Add a small fixture dataset under a git-friendly path, for example `tests/fixtures/tennis_temporal_small.json`.
5. Add `scripts/tennis/evaluate_tennis.py` that can load an adapter and score tennis rows.
6. Run a smoke evaluation before any training.
7. Only then train a tennis LoRA adapter.

Recommended deliverables:

- `Tennis_Domain_Adaptation_Report.md`;
- `config/config_tennis.yaml`;
- tennis data build scripts;
- tennis evaluation metrics JSON;
- at least one small committed fixture;
- clear statement of dataset provenance.

## 24. One-Sentence Mental Model

This codebase teaches Qwen to answer temporal QA by producing a TISER reasoning/timeline/reflection/answer trace; the first extension showed that this improves context faithfulness but not genuine conflict awareness, and the tennis extension should test whether domain-specific temporal adaptation improves tennis reasoning while preserving the general temporal reasoning ability already learned.
