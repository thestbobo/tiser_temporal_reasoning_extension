# Tennis Trace Generation Instructions

Tennis TISER training examples require generated outputs with exactly these four XML-like sections:

```text
<reasoning>...</reasoning>
<timeline>...</timeline>
<reflection>...</reflection>
<answer>...</answer>
```

The final answer inside `<answer>` must match the gold answer. Trace filtering is required because supervised fine-tuning should not learn reasoning traces that end in the wrong answer. This follows the TISER quality-control pattern: keep only generated traces whose extracted final answer matches the gold answer after normalization.

## Prepare External Generation Requests

Create JSONL prompts for an external or manual LLM workflow:

```bash
python scripts/tennis/generate_tennis_traces.py \
  --mode prepare \
  --input data/tennis/tennis_train.json \
  --output results/tennis_domain_adaptation/generations/tennis_trace_generation_requests.jsonl
```

Prepare mode does not call an API. It reads the tennis split and writes one JSON object per line with:

```json
{
  "question_id": "tennis_000002",
  "category": "yes_no_before_after",
  "context": "Djokovic lost the opening game on serve...",
  "question": "Did Djokovic call the trainer before Medvedev broke serve?",
  "gold_answer": "Yes",
  "generation_prompt": "Generate a TISER training trace..."
}
```

Use `generation_prompt` as the prompt sent to the external model. The prompt instructs the model to use only the provided context, produce no markdown fences, return only the four sections, keep reasoning concise, include an explicit ordered timeline, and make `<answer>` exactly equal to `gold_answer`.

## Expected Generation JSONL Format

Save external generations as JSONL with one object per line:

```json
{"question_id":"tennis_000002","output":"<reasoning>...</reasoning>\n<timeline>...</timeline>\n<reflection>...</reflection>\n<answer>Yes</answer>"}
```

Required fields:

- `question_id`: must match a record in the base tennis split.
- `output`: generated TISER trace containing `<reasoning>`, `<timeline>`, `<reflection>`, and `<answer>`.

Do not invent traces in code or copy the gold answer into a placeholder trace for training. The generated trace should explain the example-specific temporal evidence from the context.

## Validate and Filter Traces

Validate generated outputs and write the filtered training set:

```bash
python scripts/tennis/generate_tennis_traces.py \
  --mode validate \
  --input-generations results/tennis_domain_adaptation/generations/tennis_trace_generations.jsonl \
  --base-records data/tennis/tennis_train.json \
  --output data/tennis/tennis_train_traced.json \
  --report results/tennis_domain_adaptation/generations/tennis_trace_validation_report.md \
  --summary results/tennis_domain_adaptation/generations/tennis_trace_validation_summary.json
```

Validation checks that each output is non-empty, long enough to be a real trace, contains all four required sections, has an extractable `<answer>`, and that the normalized extracted answer equals the normalized gold answer. Only valid traces are retained in `data/tennis/tennis_train_traced.json`; invalid traces are summarized in the Markdown report and JSON summary.

The validation report includes total records, valid traces, invalid traces, answer mismatches, malformed tag counts, category-wise retention, and examples of invalid outputs.
