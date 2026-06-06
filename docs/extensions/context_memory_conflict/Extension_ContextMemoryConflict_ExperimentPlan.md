---
tags: [polito, dnlp, project, t5-temporal-reasoning, tiser, extension, knowledge-conflict, reflection, experiment-plan]
created: 2026-06-01
status: living-document
aliases: [Context-Memory Conflict Experiment Plan, E6 Experiment Plan, Faithfulness Probe Plan]
---

# Experiment Plan — Context–Memory Conflict Probe of TISER

> [!info] Where this sits
> - **What the course requires:** [[Project_Constraints_and_Topic_Brief]]
> - **The standalone pitch** (motivation, prior work, the 4-step explainer): [[Extension_ContextMemoryConflict_Presentation]]
> - **The baseline we build on** (reproduction, results, bugs): `TISER_training_notes.md` (repo)
> - This file is the **operational plan**: §1 (overview) and §2 (hypotheses) are **frozen**. Everything from §3 down is a **living document** expanded during development.

> [!success] Status — experimenting phase CLOSED (2026-06-04)
> The build-and-run phase is **complete**: modules **M0→M6** ran end-to-end (subset → memory → eligibility gate → conflict set → run matrix → faithfulness scoring), producing the 2×2 run matrix + control. The project is now in its **analysis phase** (inspect/aggregate the results we already have). The originally-planned **M7 (LLM judge), M8 (entity-fame), M9 (human validation)** and the **GPT-4o ceiling cell** are **NOT being run** for this deliverable — they are recorded as **§11 Future Paths**. Hypotheses **H1, H2, H4 are tested & supported**; **H3 is deferred** (it requires M8).

> [!abstract] One-line version
> Feed TISER a context that contradicts what the model has memorised, and measure whether it follows the **text** or its **memory**,  and whether the `<reflection>` step (the paper's load-bearing stage) actually *notices* the contradiction or silently rubber-stamps an answer.

> [!important] Working principle — asset-backed provenance (project-wide, from 2026-06-01)
> Every analysis, statistic, or claim must be backed by an **inspectable, committed artifact** — a script that produced it and an output file that can be re-opened and cited — never ephemeral terminal output. This is for (a) the team's own verification, (b) the report's evidence trail, and (c) transparency to the professor/TAs. Concretely: each pipeline stage (§7) writes a versioned artifact + `run_meta.json` under `outputs/conflict/<stage>/`, and the §3 evidence will be re-emitted by committed scripts (`scripts/conflict/analysis/`) rather than left as one-off shell output.

---

## 1. Overview / specification  `[FROZEN]`

### 1.1 What we run

We take our **frozen baseline**: `Qwen2.5-7B-Instruct` + LoRA SFT, reproduced at macro-EM 0.887 / F1 0.955, see `TISER_training_notes.md` — and run a **purely inference-only** behavioural probe. No fine-tuning. We:

1. Select **real-entity** items from `TISER_test.json` (TimeQA easy/hard, TempReason L2/L3 — entities with Wikipedia/Wikidata grounding the model plausibly memorised).
2. Elicit the model's **parametric memory** `m` (closed-book: ask the question with *no* context).
3. Construct a **counterfactual context** `c'` by a deterministic edit, so that the **context-implied answer ≠ `m`**.
4. Run the full 4-tag TISER pipeline on `(q, c')` and read the trace.
5. Score three things: does the answer follow the **context** (faithful) or **memory** (unfaithful); does `<reflection>` **name** the conflict; and does it produce the faithful answer **without** ever mentioning the conflict (*silent override*).

### 1.2 Task formalisation

- **Input:** question `q`, counterfactual context `c'` engineered so `answer(c') ≠ m`.
- **Task:** temporal-reasoning QA via the 4-stage TISER pipeline (`<reasoning>`/`<timeline>`/`<reflection>`/`<answer>`).
- **Output of interest:** not accuracy, but **faithfulness**: which source the answer follows, and the **mechanism** visible in `<reflection>`.

### 1.3 Where this extension falls (taxonomy)

This is **primarily a behavioural / robustness probe** under a controlled adversarial intervention (knowledge conflict). In the brief's category list (§8) it is the **"Other analysis/robustness"** category (the explicitly-allowed, discuss-with-TAs slot), with strong adjacency to **data enrichment** (we construct a labelled counterfactual eval set, see §4).

> [!note] Is it an "ablation"? Partly — and that's a bonus.
> An **ablation** *removes a component* and measures the performance drop (TISER's own Table 4 deletes `<reflection>` → 91.1→70.5 EM). We do **not** remove anything; we change the *input* and measure a *new property*. **But** our run matrix (§6) contrasts *TISER-prompt-with-reflection* vs *standard-prompt* and *fine-tuned* vs *off-the-shelf* , measuring faithfulness **with vs without** the reflection machinery **is** an ablation of the reflection stage, on the faithfulness axis the paper never measured. Clean framing for the report: *"a knowledge-conflict robustness probe whose run matrix embeds an ablation of the reflection stage on a faithfulness metric."*

### 1.4 Scope & constraints

- **Inference-only.** Reuses the frozen adapter; the only GPU cost is generation (a few GPU-hours). 
- **Splits in scope:** TimeQA easy/hard + TempReason L2/L3 (real entities). **TGQA excluded** (synthetic/fictional entities → no parametric memory to conflict with; confirmed in §3).

---

## 2. Hypotheses  `[FROZEN]`

> [!tip] Design property
> Every cell below is a publishable sentence, there is **no "experiment failed" outcome**, only different findings. This is the "negative results are fine if you measured the right thing" posture (brief open-question #2).

| # | Hypothesis | If TRUE → story | If FALSE → story |
|---|---|---|---|
| **H1** | The TISER pipeline is **more context-faithful** than plain CoT on the same conflict items | timeline+reflection structure forces the model onto the text → validates the "grounded reasoner" claim *beyond* the paper | structure is cosmetic for faithfulness; memory still wins → bounds the value of reflection |
| **H2** | When TISER is faithful, `<reflection>` **explicitly names the conflict** ("context says X, but I assumed Y") | reflection works *mechanistically* — it really audits | **silent override**: right answer, reflection never mentions the clash → faithfulness is incidental, not reasoned |
| **H3** ⏸️ | Faithfulness depends on **entity fame** (stronger memory → harder to override the context) | quantifies a memory-strength → faithfulness curve, a genuinely new measurement | flat → conflict resolution is entity-independent |
| **H4** | Faithfulness / detection depend on the **type of conflict** (date-shift vs entity-swap vs order-reversal — see §4) | TISER resists some temporal-conflict kinds but not others → a fine-grained map of the reflection stage's competence | flat across types → conflict handling is type-agnostic |

> [!note] Outcome (2026-06-04): **H1 ✅, H2 ✅ (silent override), H4 ✅; H3 ⏸️ deferred.** H1/H2/H4 are answered from the M6 run-matrix results (see [[Extension_ContextMemoryConflict_Report]] §5). **H3 is deferred to §11 Future Paths** because it depends on the entity-fame signal (M8), which we chose not to run for this deliverable.

> [!note] H4 is new vs the pitch
> H4 was added here because §3's finding that contexts are fully *structured* makes per-class perturbation cheap and reliable (§4). It is the hypothesis that the "data-enrichment / perturbation-taxonomy" structuring exists to test. H1–H3 are unchanged from [[Extension_ContextMemoryConflict_Presentation]].

---
---

## 3. Evidence gathered so far (living)

> Collected 2026-06-01 from the frozen baseline artifacts (`outputs/baseline_full/`) and the full `TISER_test.json` (22,014 records, fetched locally to `…/Deep Natural Language Processing/full_data/`).

### 3.1 Reflection is confirmatory, not corrective `→ motivates H2`

Parsing `<reflection>` from the baseline's 2,500 in-domain traces (500/split, greedy):

| Split | Reflection parsed | "Confirm / no-error" language | Any revision language |
|---|---:|---:|---:|
| tgqa_test | 500 | 83.4% | 3.6% |
| tempreason_l2_test | 500 | 90.4% | 3.0% |
| tempreason_l3_test | 500 | 78.0% | 4.0% |
| timeqa_easy_test | 500 | 80.4% | 0.2% |
| timeqa_hard_test | 500 | 93.8% | 1.6% |
| **in-domain pooled** | **2,500** | **84.8%** | **2.3%** |
| _tot_semantic (OOD, ref.)_ | _432_ | _79.9%_ | _12.0%_ |

**Reading:** on the agreeing contexts the paper trained/tested on, reflection has collapsed into a **confirmatory ritual**, it revises ≈2% of the time. Notably it revises **~5–6× more on OOD ToT (12%)**, i.e. it *can* activate when the input is genuinely hard. So the probe's core question is sharp and pre-loaded: **does an engineered conflict push the revise/detect rate up (genuine audit), or does the 2.3% rubber-stamp persist (→ silent override, H2-FALSE)?** Given the 84.8% prior, *silent override is the predicted outcome* — a crisp, defensible prediction, not a fishing trip.

Representative agreeing-context reflection (TempReason L2):
```
<reflection> The reasoning correctly identifies that September 1950 falls within the
period when Jaroslav Pelikan worked at Concordia Seminary. There are no errors or
improvements needed in the reasoning process. </reflection>
```

### 3.2 Contexts are fully structured → deterministic perturbation `→ feasibility`

Every prompt has the exact shape: fixed instruction preamble → `Question: <q>` → `Temporal context: <c>` → `### Answer:`. **Context isolation parses 100% of all 22,014 records.** The real-entity context formats:

| Split | Context format (verbatim shape) | Median year-mentions | Answer-token ∈ context |
|---|---|---:|---:|
| TempReason L2/L3 | `Entity works for ORG from Mon, YYYY to Mon, YYYY.` (×N) | 12 | **100%** |
| TimeQA easy/hard | `YYYY - YYYY : Entity's team are ( X ) , ( Y ).` (×N) | 8 | 98–100% |
| _TGQA (excluded)_ | `(EVENT) starts/ends at YYYY` (×N), **fictional entities** | 11 | _71%_ |

**Reading:** (a) date-shift and entity-swap are **pure string operations** on these templates — no LLM rewriter needed for the in-scope splits, which collapses the pitch's "~$20–30 API / 70% deterministic" to **~100% deterministic, near-zero generation cost**. (b) The answer is literally a span in the context, so a context edit **deterministically** changes the context-implied gold. (c) TGQA's fictional entities + 71% answer-in-context confirm its exclusion from the *memory*-conflict probe.

### 3.3 Assets confirmed in hand

- Frozen LoRA adapter + tokenizer (`outputs/baseline_full/…/adapter/`), all 3 epoch checkpoints; network volume still live for new runs.
- `run_meta.json` confirms reproducibility: `git_sha 42ca369` (= current `main`), bf16 (`load_in_4bit: false`), pinned `transformers 4.46.3 / peft 0.13.2 / trl 0.12.2 / torch 2.4.1`.
- Full `TISER_test.json` (with contexts) fetched locally.

---

## 4. Conflict construction & perturbation taxonomy (living)

> This is the "data-enrichment" piece. The user's instinct to **classify and label the perturbations** is correct and valuable — below is the design plus a **critical, objective evaluation** of it.

### 4.1 Disentangling two things called "data enrichment"

> [!warning] They are not the same
> 1. **Counterfactual conflict-set construction** (this §) — building a *new, labelled adversarial eval set* by perturbing real contexts. **This is the genuine data-enrichment contribution.**
> 2. **The GPT-4o row in the run matrix** (§6) — that is a **reference/ceiling baseline model**, i.e. *model exploration*, **not** data enrichment. Don't conflate them in the report.

### 4.2 Proposed perturbation classes (the labelled independent variable)

Each conflict item carries a `conflict_type` label, so faithfulness/detection can be reported **per class** (this is what H4 tests).

| Class                  | Edit (deterministic)                                                                                                | Applies to       | What it probes                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------- | --------------------------------------------- |
| **C1: Date-shift**     | move event date ranges so the entity satisfying the queried time-point changes                                      | L2, TimeQA       | pure **time-event binding** under conflict    |
| **C2: Entity-swap**    | replace the answer entity in context with a plausible same-type distractor drawn from the dataset's own entity pool | L2, TimeQA       | **relational/factual substitution** vs memory |
| **C3: Order-reversal** | reorder / re-date events so a `before/after` relation flips                                                         | L3 (event-event) | **event-event ordering** under conflict       |
|                        |                                                                                                                     |                  |                                               |

Each edit is validated to satisfy the **conflict invariant**: `answer(c') ≠ m` **and** `answer(c')` is unique & unambiguous given `q`.


> [!note] Design decisions are locked
> Per-class N, memory-confidence, control arm, and the rest are **settled in §7.5 (Locked decisions)** .

---

## 5. Metrics (living)

| Metric | Definition | Reads as |
|---|---|---|
| **Context-faithfulness rate** | fraction of conflict items where `<answer>` matches `answer(c')`, not `m` | "does it trust the text?" (= `1 − memorisation rate`, Longpre 2021) |
| **Conflict-detection rate** | fraction where `<reflection>` explicitly flags the reasoning↔timeline/context mismatch (rubric + 2-LLM-judge) | "does reflection *notice*?" — compare against the §3.1 baseline revise-rate of **2.3%** as the null |
| **Silent-override rate** | faithful answer **AND** reflection never mentions the conflict | "right answer, wrong mechanism" (H2-FALSE signature) |
| EM (guardrail) | standard exact-match, sanity only | nothing broke |

Stratifiers: `conflict_type` (H4), entity-fame bucket (H3).

---

## 6. Run matrix (living)

The conflict set is an **evaluation** set (a probe), **not** training data, nothing is re-trained. We run several **configurations** at inference on the *same* set. Two independent comparison axes:

- **Prompt axis** (same model, structure vs none) → *does the timeline+reflection machinery improve faithfulness?* (**H1**)
- **Model axis** (same prompt, vanilla vs our fine-tune) → *does TISER SFT make the model more/less context-faithful?* — pointed, given the 84.8% rubber-stamp finding (§3.1).

| Config                      | Standard prompt | TISER prompt | Purpose                         |
| --------------------------- | :-------------: | :----------: | ------------------------------- |
| Qwen2.5-7B off-the-shelf    |        ✓        |      ✓       | base-model floor (model axis)   |
| **Qwen2.5-7B-TISER (ours)** |        ✓        |    **★ **    | **the star cell + H1 contrast** |
| GPT-4o                      |        ✓        |      ✓       | closed-model reference ceiling  |

6 cells total × a few-hundred-item conflict set

---

## 7. Implementation pipeline (module spec)  `[implementation entry-point]`

> [!info] Build target
> A dedicated branch **`ext/context-memory-conflict`** off `main` at the frozen baseline. The pipeline is a **linear chain of config-driven modules**, each consuming the previous artifact and emitting a new versioned one + `run_meta.json`. It deliberately mirrors existing repo conventions: `load_config`/`AttrDict`, `outputs/<run_name>/`, `write_run_meta`, `generate_batch`, `parse_answer`, `normalize_answer`. **This section is what gets handed to the implementing Claude instance.**

### 7.1 What "is it in the model's memory?" actually means  `[conceptual contract]`

We do **not** test whether the question, answer, or prompt *text* is "in memory". We measure the model's **parametric belief about the answer**:

- Ask the question **closed-book** (no `Temporal context`). Record the produced answer `m`.
- An item is **memorised & eligible** iff `m` is **stable/confident** *and* `m == original gold` (the model genuinely knows the real-world fact).
- We then build `c'` so the **context asserts something ≠ `m`**. Items with no confident belief, or a confident *wrong* belief, are dropped (or bucketed for a secondary analysis), because they cannot form a clean memory↔context collision.

> [!note] Whose memory (D-MEM, locked)
> Primary memory `m` is elicited from **our fine-tuned TISER model** — it is the actual experimental subject, so its belief is the most *valid* anchor. We **also** elicit from off-the-shelf `Qwen2.5-7B-Instruct` as a **cross-check** (LoRA is attention-only on a frozen base → world knowledge is essentially shared; M1 is cheap, so we get this for free). The conflict is anchored on the **original gold** as the shared "world/memory" label (**D-ANCHOR**); under strict eligibility (**D-ELIG**) `m == gold` on the kept set, so a *single* conflict set stays comparable across every run-matrix cell. *Caveat to watch:* closed-book is out-of-distribution for the fine-tuned model (trained to always read a context) — M1's artifact reports the malformed-rate; if it is too high, fall back to the base model as primary.
>
> **Consequence to state openly:** the eligibility filter biases the set toward **well-known entities** — i.e. the *hardest* case for context-faithfulness (strongest memory to override). This is a feature (it is the stress test, and it supplies the fame gradient for H3), not a bug — but it conditions H1's claim and must be reported as such.

### 7.2 Module table

Loc legend: **I** = internal (in-repo, runs on our infra) · **X** = external (API / KB / manual — documented, run separately).

| ID      | Module (file)                                           | Purpose                                                                                                      | Input artifact                                                                                  | Output artifact                                                                                                           | Compute                                 | Reuses                                                                                           | Key constraints / decisions                                                                                                                                                                                                                                                                                                                                              |    Loc    |
| ------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------: |
| **M0**  | `src/conflict/subset.py` ← `01_build_subset.py`         | parse in-scope records; isolate + structure-parse context; tag applicable conflict classes per item          | `TISER_test.json`                                                                               | `subset/items.jsonl` `{id, split, q, q_type, ctx_raw, ctx_events[], answer_orig, applicable_classes[]}`                   | CPU                                     | `dataset._load_records`; verified `Temporal context:…### Answer:` regex                          | in-scope splits only (TimeQA, TempReason; **drop TGQA/ToT**); per-format parsers (L2/L3 prose vs TimeQA range-list differ); seeded                                                                                                                                                                                                                                       |     I     |
| **M1**  | `src/conflict/memory.py` ← `02_elicit_memory.py`        | closed-book belief elicitation                                                                               | `items.jsonl` + model cfg (**primary = fine-tuned TISER**; + off-the-shelf base as cross-check) | `memory/memory_<model>.jsonl` `{id, m, samples[], agreement, greedy}`                                                     | **GPU**                                 | adapter loader (TISER) + base loader (no adapter) + `generate_batch` w/ plain closed-book prompt | run for **both** models; **D-K**: k=5, T≈0.7, keep majority if ≥3/5; report malformed-rate (closed-book is OOD for the fine-tune — fall back to base if too high)                                                                                                                                                                                                        |     I     |
| **M2**  | `src/conflict/eligibility.py` ← `03_filter_eligible.py` | eligibility + **YIELD GATE**                                                                                 | `memory_*.jsonl` + `items.jsonl`                                                                | `eligible/eligible.jsonl` + `yield_report.json` (counts per split/q_type/class) + `confident_wrong.jsonl` (bonus bucket)  | CPU                                     | `metrics.exact_match`                                                                            | **D-ELIG (strict)**: keep iff `agreement≥0.6` ∧ `EM(m, answer_orig)=1`; **also persist `confident-but-wrong` ids** (`m`≠gold) as a flagged bucket for a later optional analysis; **GO/NO-GO**: if eligible-per-class < target N → revise scope before building anything downstream                                                                                       |     I     |
| **M3**  | `src/conflict/perturb.py` ← `04_build_conflicts.py`     | deterministic perturbation engine, per class                                                                 | `eligible.jsonl`                                                                                | `conflicts/conflict_set.jsonl` `{id, conflict_type, ctx_prime, answer_ctx_prime, m, answer_orig, edit_meta, validity_ok}` | CPU                                     | M0 structured parsers                                                                            | **class assignment is substrate-driven, not equal** (C1: L2/TimeQA; C2: L2/TimeQA(+L3); C3: L3 only); **build for N-class (flag-gated), run 1-class first** (**D-NCLASS**) — rows keyed by `(item_id, conflict_type)`; enforce invariants `answer(c')≠m`, single-coverage, `distractor≠m`; emit a small **no-conflict control arm** (`answer(c')=m`, **D-CTRL**); seeded |     I     |
| **M4**  | `src/conflict/prompts.py` (used by `05`)                | build run inputs per prompt-style; inject `c'` into TISER template + build **standard** (non-TISER) template | `conflict_set.jsonl`                                                                            | `run_inputs/<style>.jsonl`                                                                                                | CPU                                     | TISER preamble extracted verbatim from data                                                      | standard-prompt template is **new** (direct-answer, **D-STD**); chat-template wrap stays identical to training (the #1 baseline bug)                                                                                                                                                                                                                                     |     I     |
| **M5** ✅ | `05_run_inference.py`                                   | execute run-matrix cells (4 local Qwen)                                                                      | `run_inputs/*.jsonl` + model cfg                                                                | `generations/<model>__<style>.jsonl` `{id, raw_generation}`                                                               | **GPU** (vLLM/H100)                      | `generate_batch`; `load_adapter_for_inference` (TISER); base loader (off-the-shelf)              | **4 local cells run** (tiser/base × tiser/standard, n=1,176 each); GPT-4o ceiling cell **not run** → §11 Future Paths                                                                                                                                                                                                                                                     |     I     |
| **M6** ✅ | `src/conflict/score.py` ← `06_score.py`                 | parse trace; faithfulness + silent-override (lexical proxy) + guardrail EM                                   | `generations/*` + `conflict_set.jsonl`                                                          | `scored/<model>__<style>.jsonl` + `<cell>.metrics.json`                                                                   | CPU                                     | extend `parser.parse_answer` w/ `<reflection>`/`<timeline>` parsers; `normalize_answer`          | faithfulness = `answer ≈ answer(c')` not `m`; silent-override read from a **lexical** reflection-mention proxy (authoritative LLM judge **deferred** → §11)                                                                                                                                                                                                               |     I     |
| **M10** 🔄 | `07_analyse.py` *(analysis phase — current)*           | aggregate; stratify by conflict_type; bootstrap CIs; H1/H2/H4 tests; figures + error analysis                | `scored/*`                                                                                      | `analysis/*.{json,csv,png}` + report tables                                                                               | CPU                                     | `aggregate`                                                                                      | **two views**: (a) headline faithfulness one-variant-per-item; (b) within-item paired for H4; **bootstrap CIs + paired McNemar** (H1 prompt-axis, model-axis); qualitative reflection inspection (silent override) + the en-dash normalisation fix (Report §7)                                                                                                            |     I     |

### 7.3 External / out-of-repo steps — NOT RUN for this deliverable

All external/API/manual steps (M5 GPT-4o ceiling, M7 LLM-judge, M8 entity-fame, M9 human validation) were **scoped out** of the graded deliverable to concentrate effort on analysing the results we already have. They are preserved as **§11 Future Paths** (with the rationale for each). The executed pipeline is **fully internal/offline**: M0–M6 + the M10 analysis pass.

### 7.4 Compute & dependency notes

- **GPU stages: M1, M5 (local cells) only.** Everything else is CPU/laptop. The single live GPU dependency was the RunPod network volume (adapter + base model). Closed-book elicitation (M1) and the 4 inference cells (M5) were a few GPU-hours total.
- **Executed dependency order:** M0 → M1 → M2 **(GATE = GO)** → M3 → M4 → M5 → M6 → M10 (analysis). The GATE was the only go/no-go; everything downstream was built only after yield was confirmed. (M7/M8/M9 → §11 Future Paths.)

### 7.5 Locked decisions (settled — build to these)

- **D-MEM** — Primary memory `m` is elicited from **our fine-tuned TISER model** (the experimental subject); off-the-shelf base Qwen is run **too**, as a cross-check. Fall back to base as primary only if the fine-tune's closed-book malformed-rate is too high (M1 reports it).
- **D-ANCHOR** — The "memory/world" label is the **original gold**; per-model `m` is also recorded. Under strict D-ELIG, `m == gold` on the kept set, so **one** conflict set serves all run cells. The context label `answer(c')` is always freshly computed from each edit.
- **D-K** — Confidence via **self-consistency**: `k=5` samples at `T≈0.7`; an item's `m` = majority answer, kept only if the majority appears in **≥3/5** samples. Threshold sensitivity is re-reportable from the same artifact for free.
- **D-ELIG** — **Strict** for the headline: keep iff **confident** (D-K) **and** **correct** (`m == gold`). Items that are confident-but-wrong (`m ≠ gold`) are **not** discarded — their ids are persisted as a flagged bucket for an optional later analysis.
- **D-NCLASS** — Perturbation engine is **built for N-class** (generate every applicable class-variant per item, rows keyed by `(item_id, conflict_type)`) but **run in 1-class mode first** to get the pipeline working end-to-end; flip on multi-variant once yield is known. Analysis keeps two views (headline = one-variant-per-item; H4 = within-item paired).
- **D-CLASSN** — Target **≥150 eligible items/class**; **C3 (order-reversal, L3-only) is secondary** if L3 yield is low. This target governs the M2 GATE verdict.
- **D-CTRL** — Include a small (~100–150 item) **no-conflict control** arm (edit the context but keep `answer(c') = m`) as a faithfulness floor and a guard against "the edit just broke the context" objections.
- **D-STD** — The **standard prompt** is a plain direct-answer prompt (no 4-tag instruction); it is a **baseline arm, not a replacement** for our method. H1 = TISER-prompt vs standard-prompt on the same model. An optional CoT ("think step by step", no timeline/reflection) middle arm can be added if time allows; default scope is standard vs TISER.
- **D-CB-PROMPT** *(settled 2026-06-02, P1)* — The closed-book elicitation prompt (M1) is **`tiser_no_context`**: take each record's own `prompt` and **delete only the `Temporal context:` block**, keeping the verbatim 4-tag preamble + the Question + `### Answer:`. The answer span lives only in the context, so its removal makes this a genuine memory test, not answer-leakage; output stays `<answer>`-parseable (reuse `parse_answer`). Rationale: closest to the SFT distribution, injects no novel instruction text. A `tiser_empty_note` variant (explicit "(none provided — answer from your own knowledge.)" placeholder) is a one-line config switch / CLI flag, kept as the fallback if the fine-tune's malformed-rate is high (then escalate to the D-MEM base-as-primary fallback).

---

## 7.6 Implementation notes — design deviations & code  *(living; spec-level only)*

> [!info] Run log & live results moved out
> The **execution narrative and results** (what was run, hurdles hit, numbers observed) live in the research-report doc **[[Extension_ContextMemoryConflict_Report]]**, updated live per run. This section keeps only **durable design deviations and the code surface** — the planning/spec record.

Built on `ext/context-memory-conflict`. Pipeline lives in `src/conflict/` with thin CLIs in `scripts/conflict/`; config is `config/conflict.yaml` (a `conflict:` block on top of the baseline keys `load_config` requires). Artifacts under `outputs/conflict/<stage>/` with `run_meta.json`, per the §7.1 provenance rule. Module status is tracked in the roadmap (§8) and the report.

**Deviations from the §7.2 spec (all minor, asset-backed)**
1. **Parsers are more general than the spec's verbatim templates.** Live data has far more variety than §3.2's two example shapes: TempReason predicates vary (`works for`, `plays for`, `is the head coach of`, `holds the position of`, …) and can be **object-first**; 3–4-digit (historical) years; TimeQA relations beyond "team" (`spouse`, `school`, `position`, …), **month-bearing and single-point** intervals, **nested parentheses** in values, and bare (non-possessive) subjects. Parsers were generalised (predicate-agnostic `head` for L2/L3; balanced-paren scanner + optional-month/point intervals for TimeQA) to reach the 100%-parse guarantee.
2. **No per-event `object` field for L2/L3.** A common-prefix attempt to split subject/relation/object was only ~59% correct (object-first relations, shared object prefixes), so we **store the robust fields only** (`head`, dates, `raw`). Answer-entity localisation for C2 is **deferred to M3** via `answer_orig` membership in the context (the gold *is* a literal substring of L2/L3 contexts).
3. **CPU stages avoid the GPU stack.** M0/M2 use a local `load_records` (mirrors `dataset._load_records`) and a conflict-local `write_run_meta` (best-effort lib versions) so they run on a laptop without `datasets`/`torch`. M1 uses the real `set_seed`/loaders on GPU.
4. **Style-aware answer parsing in M6.** The `standard` prompt (D-STD) emits a **bare** direct answer with no `<answer>` tag, so parsing it for tags (as the TISER trace requires) flags every row malformed. `src/conflict/score.py` therefore **dispatches on style**: `parse_answer` (tag extraction) for `tiser`, `parse_plain` (whole stripped completion is the answer; malformed only if empty, in `src/inference/parser.py`) for `standard`. *(This surfaced as a bug — the first standard-cell scoring read `malformed_rate = 1.000`; recorded in Report §4.7.)*

**Baseline-code touches (only these, additive & back-compatible)**
- `src/inference/generate.py`: `generate_batch` gained keyword-only `temperature`/`top_p`/`num_return_sequences`; **defaults leave the greedy path byte-identical** (baseline reproduction unaffected).
- `src/model/loader.py`: new `load_base_for_inference(cfg)` (off-the-shelf base, no adapter) + extracted `_clear_sampling_defaults`.
- `src/inference/vllm_engine.py` *(new)*: vLLM-backed generation, **drop-in for `generate_batch`** (same flat `i*k` contract). Wraps prompts in the identical chat template, feeds token-ids, supports LoRA via `LoRARequest`. Sets `VLLM_USE_DEEP_GEMM=0` (bf16 model never needs the FP8 path; recent vLLM otherwise hard-fails at warmup). **Decision:** M1/M5 generation runs on vLLM (~10–50× HF throughput); HF path retained behind a flag for the frozen-baseline reproduction.
- `config/conflict.yaml`: `conflict.memory.engine: vllm | hf` (+ `gpu_memory_utilization`, `max_model_len`). `memory.py` dispatches on it; the **HF loader is imported lazily** inside the `hf` branch so the vLLM path needs neither `peft` nor `bitsandbytes`.
- `src/conflict/eligibility.py`: **empty-`m` guard** — a malformed/empty closed-book majority is treated as non-confident (dropped), so it cannot pollute the `confident_wrong` bucket.

**Forward (P4/M3): inputs ready** — `eligible.jsonl` carries `ctx_events` + `applicable_classes` + `m`; the perturbation engine (C1/C2/C3) builds on it. The M3–M6 build spec is the active planning frontier (see the handoff/plan). *Open spec note for M3:* extend `eligibility._cross_model_agreement` to report the **eligible-subset** and **both-confident** agreement, not only the full-set figure (the full-set number is noise-dominated — see report).

---

## 8. Roadmap (living)

> Re-expressed against the §7 modules. Phases 0–2 done (asset-backed in §3); 3+ map 1:1 to modules on the `ext/context-memory-conflict` branch.

- [x] **P0 — Data** — `TISER_test.json` (22k, contexts) fetched to `…/Deep Natural Language Processing/full_data/`. ✅
- [x] **P1 — Prompt anatomy** — context isolable 100% (§3.2). ✅
- [x] **P2 — Baseline trace mining** — reflection 84.8% confirm / 2.3% revise in-domain (§3.1). ✅ *(to be re-emitted as a committed script per §7.1)*
- [x] **P3 — M0+M1+M2 → YIELD GATE** — **DONE, verdict = GO** *(2026-06-02)*. M0 15,898 items / 100% parse; M1 closed-book elicitation run on RunPod (vLLM, both models); M2 gate **GO** with 571 eligible (C1 234 · C2 571 · C3 337, all ≥150). Numbers + cross-model analysis in [[Extension_ContextMemoryConflict_Report]]. *(was the highest-risk milestone — cleared.)*
- [x] **P4 — M3+M4** — **DONE** *(2026-06-03)*. Conflict set = **1,176 labelled rows** (C1 203 · C2 520 · C3 333 · control 120; 100% per-class validity) + TISER/standard run inputs (1,176 each). Numbers in [[Extension_ContextMemoryConflict_Report]] §4.5–4.6.
- [x] **P5 — M6 scorer** — **DONE** *(2026-06-03)*. Trace/reflection parser + faithfulness/silent-override-proxy scorer; required a style-aware fix for standard cells (see §7.6 deviation). Silent override is read from a **lexical reflection-mention proxy**; the authoritative LLM **judge (M7) is deferred → §11 Future Paths**.
- [x] **P6 — M5** — **DONE** *(2026-06-03)*. All four local Qwen cells (model × prompt), n=1,176 each, vLLM/H100. Headline faithful-EM: tiser×tiser **0.787**, tiser×standard 0.574, base×tiser 0.561, base×standard 0.380 → H1 **+0.213**, SFT axis **+0.226**. GPT-4o cell deferred (optional). Report §4.7.
- [ ] **P7 — M10 analysis (current phase)** — aggregate the four cells: bootstrap CIs on every faithful-EM, paired **McNemar** for the H1 prompt-axis and the model-axis, per-class breakdown (the C3 inversion), qualitative reflection inspection for silent override, and the en-dash normalisation fix (Report §7). **No new generations** — pure analysis of existing artifacts.
- [ ] **P8 — Write-up** — "grounded reader vs. fluent confirmer" narrative + tables, off the Report doc.
- ⏸️ **Future Paths (NOT in this deliverable → §11):** M7 LLM-judge (harden H2), M8 entity-fame (unlock H3), M9 human validation, GPT-4o ceiling cell.

---

## 9. Risk register (living)

| Risk | Severity | Mitigation / status |
|---|---|---|
| **Conflict-set yield** (after `confident m` ∧ `m ≠ answer(c')`, set may shrink) | ~~High~~ **Resolved** | YIELD GATE cleared **2026-06-02**: **GO**, 571 eligible (C1 234 · C2 571 · C3 337, all ≥150). See [[Extension_ContextMemoryConflict_Report]]. |
| Memory elicitation reliability (is `m` stable?) | Medium | self-consistency k=5/T≈0.7, keep ≥3/5 (**D-K**, §7.5); closed-book OOD for fine-tune → M1 reports malformed-rate, base-model fallback |
| Per-class statistical power | Medium | 3 classes, pre-registered balanced N ≥150/class, report CIs |
| Edit validity (C1 ambiguity, C2 distractor) | Medium | post-edit invariant asserts + 200-item manual spot-check |
| Judge subjectivity (conflict-detection) | Medium | rubric + 2 judges + inter-judge agreement; anchored to §3.1 lexical baseline |
| Context isolation brittleness | **Resolved** | 100% parse across 22k records (§3.2) |
| Adapter availability | **Resolved** | adapter local + network volume live (§3.3) |

---

## 10. References

Full annotated bibliography (Longpre 2021 knowledge-conflict, Xu 2024 survey, the 2025 temporal-conflict papers, Kim & Hwang counterfactual-consistency, TISER) lives in [[Extension_ContextMemoryConflict_Presentation#4. Papers cited (and why each is here)]]. Course constraints and the paper summary: [[Project_Constraints_and_Topic_Brief]].

---

## 11. Future Paths (scoped out of this deliverable)

> [!info] Why these are here, not in the pipeline
> The experimenting phase delivered a complete study with **M0–M6** (H1/H2/H4 answered). The steps below would *deepen* the work but each adds external API cost, manual labour, or a new data dependency without changing the core finding. They are recorded so a future iteration (or the report's "future work" paragraph) can pick them up cleanly. Each already has a stub in the original module spec.

| Future path | Was | Unlocks | Why deferred | What exists already |
|---|---|---|---|---|
| **LLM conflict-detection judge** | M7 (`src/conflict/judge.py`) | Authoritative H2 (replaces the lexical mention proxy) | External API + ≥2-judge agreement + human calibration; the **lexical proxy already tells the silent-override story** (3.8% vs the 2.3% null) | `reflection_text` is persisted in every `scored/*__tiser.jsonl` row → a judge can be run later with **no re-generation** |
| **Entity-fame signal** | M8 (`src/conflict/fame.py`) | **H3** (fame → faithfulness curve) | Wikidata/pageviews lookup + noisy name→KB-id linking; H3 is the one hypothesis with no substrate yet | `eligible.jsonl` carries the entities; the eligible set is already fame-biased (well-known entities) |
| **Human validation** | M9 | Edit-validity evidence + judge calibration | Manual labelling; M3 already reports **100% per-class construction validity** and the control arm guards against broken edits | `conflict_sample.jsonl` (30 rows) is a ready spot-check sheet |
| **GPT-4o ceiling cell** | M5 external arm | Closed-model reference ceiling | API cost; not load-bearing for H1/H2/H4 (the 4 local cells already give both axes) | `run_inputs/{tiser,standard}.jsonl` are model-agnostic → feed any model, score with the same M6 |

**If exactly one is ever resumed, do M7** (the judge): it is the cheapest, hardens the most interesting claim (silent override / H2), and needs no new model runs.
