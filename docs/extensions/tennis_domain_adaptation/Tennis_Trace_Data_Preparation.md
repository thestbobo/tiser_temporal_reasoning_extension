# Tennis Trace Data Preparation

Tennis-only adapter training requires supervised trace targets, not only final
answers. The current `data/tennis/tennis_train.json`, `tennis_dev.json`, and
`tennis_test.json` files include `output` fields, but those outputs are
placeholder traces. They are useful for plumbing checks only and are not valid
training targets for a meaningful tennis adapter.

## Current State

Existing tennis data files:

- `data/tennis/tennis_train.json`
- `data/tennis/tennis_dev.json`
- `data/tennis/tennis_test.json`
- `data/tennis/processed/tennis_all_standard_prompt.json`
- `data/tennis/processed/tennis_all_tiser.json`
- `data/tennis/processed/tennis_raw_audited.json`

Missing required full traced training file:

- `data/tennis/tennis_train_traced.json`

Example only:

- `data/tennis/tennis_train_traced_sample.json`

## Expected Schema

`src/data/dataset.py` documents the training schema as:

```python
("dataset_name", "question_id", "question", "answer", "prompt", "output")
```

The training dataset builder consumes `prompt` as the user message and `output`
as the supervised assistant completion. `scripts/tennis/train_tennis.py` also
validates that an answer can be extracted from `<answer>...</answer>` and
compared with the gold `answer`.

Every traced training record should preserve at least these fields:

- `dataset_name`
- `question_id`
- `question`
- `answer`
- `prompt`
- `output`

Category metadata such as `category`, `tags`, and `source` should be preserved
when available because validation and diagnostics use it.

## Why Placeholders Are Invalid

Placeholder traces repeat generic text such as:

- `The question asks for a temporal relation in the given tennis context`
- `A detailed generated timeline will be added in the trace-generation step`
- `The final answer is checked against the provided gold answer`

These strings do not explain the example-specific temporal evidence. Training on
them can verify that the trainer runs, but it does not teach temporal reasoning
or support adapter-quality claims.

## Generate Trace Requests

Use the offline helper to export prompts for external trace generation. It does
not call any API.

```bash
python scripts/tennis/prepare_tennis_trace_generation.py \
  --input data/tennis/tennis_train.json \
  --output results/tennis_domain_adaptation/trace_generation/tennis_trace_requests.jsonl \
  --limit 50 \
  --format jsonl
```

The exported prompt includes the temporal context, question, category, and gold
answer. Including the gold answer is intentional because this is supervised
trace construction, not evaluation.

Generated outputs must use strict TISER-style format:

```text
<reasoning>
...
<timeline>...</timeline>
<reflection>...</reflection>
</reasoning>
<answer>...</answer>
```

The text inside `<answer>` must match the record's gold `answer` after tennis
normalization.

## Validate Traced Data

After externally generated traces are merged into records with the schema above,
validate the candidate file:

```bash
python scripts/tennis/validate_tennis_traces.py \
  --input data/tennis/tennis_train_traced.json \
  --failed-output results/tennis_domain_adaptation/trace_validation/failed_examples.json
```

The validator accepts JSON arrays or JSONL and checks:

- the file exists and parses;
- every record has the required training schema fields;
- every record has non-empty `prompt` and `output`;
- `output` contains `<reasoning>` and `<answer>` sections;
- the extracted answer matches the gold answer after tennis normalization;
- placeholder traces are absent;
- counts are reported by category;
- failed examples are written to JSON.

The sample file can be checked with:

```bash
python scripts/tennis/validate_tennis_traces.py \
  --input data/tennis/tennis_train_traced_sample.json \
  --failed-output results/tennis_domain_adaptation/trace_validation/sample_failed_examples.json
```

## Safe To Train

It is safe to start tennis-only adapter training only after:

- `data/tennis/tennis_train_traced.json` exists;
- the validator reports zero invalid records;
- failed examples JSON is empty;
- placeholder count is zero;
- no external trace-generation artifacts are mixed into train data without
  answer validation;
- the lightweight test suite still passes.

Until then, tennis-only training remains blocked except for explicit
plumbing-only smoke checks.
