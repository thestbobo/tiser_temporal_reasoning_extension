---
tags: [polito, dnlp, project, t5-temporal-reasoning, tiser, extension, knowledge-conflict, report, living-document]
created: 2026-06-02
updated: 2026-06-04
status: experimenting-closed · analysis-phase
aliases: [Context-Memory Conflict Report, E6 Report, Faithfulness Probe Report]
---

# Context–Memory Conflict in TISER — Research Report

> [!info] What this document is, read me first
> This is the **handoff doc for the write-up**. It tells you, end-to-end and in plain language: *what we built, what we ran, what came out, and what it means.* The **design/decision record** lives in [[Extension_ContextMemoryConflict_ExperimentPlan]]; the **baseline** in `TISER_training_notes.md`. The **experimenting phase is closed** (modules M0→M6 ran); we are now in the **analysis phase** (no new model runs, just aggregating the results below).

---

## 1. Abstract

We probe whether the fine-tuned **TISER** model (Qwen2.5-7B-Instruct + LoRA SFT) trusts **what it reads** or **what it remembers**, and whether its `<reflection>` step actually *notices* a contradiction. We take real-entity temporal-QA items the model **provably knows** (it answers them correctly with *no context*), then **deterministically edit the context** so it points to a *different* answer than memory. Running the edited items through a 2×2 matrix of {fine-tune vs vanilla Qwen} × {TISER 4-tag prompt vs plain prompt}, we find:

- **The TISER pipeline makes the model a faithful reader (H1 ✅):** it follows the edited text **78.7%** of the time, **+0.213** over the same model with a plain prompt and **+0.226** over vanilla Qwen with the TISER prompt.
- **But it does not *notice* the conflict (H2 ✅, "silent override"):** when it gives the faithful answer, its reflection mentions the contradiction only **3.8%** of the time, essentially the 2.3% "rubber-stamp" rate we measured on normal data. It is right for the wrong reason.
- **One conflict type defeats it (H4 ✅):** date-shift and entity-swap are easy (0.94 / 0.97 faithful), but **order-reversal collapses to 0.37**, the only case where memory beats the text. The model reverts to the event order it remembers.
- **H3 (does fame predict faithfulness?) is deferred** — it needs the entity-fame signal we scoped out.

**Deliverable:** a labelled **1,176-row counterfactual eval set** + a fully-scored 2×2 run matrix with a control arm, every stage backed by a committed artifact.

---

## 2. Hypotheses & verdicts

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H1** | The TISER prompt (timeline+reflection) is **more context-faithful** than a plain prompt | ✅ **supported** | faithful-EM **0.787** (TISER prompt) vs **0.574** (plain), same fine-tune → **+0.213**; +0.181 on vanilla Qwen |
| **H2** | When faithful, `<reflection>` **names the conflict** (else: *silent override*) | ✅ **silent override** | faithful 78.7% but reflection names the clash only **3.8%** (≈ the **2.3%** baseline null). Lexical proxy; an LLM judge is future work |
| **H3** | Faithfulness depends on **entity fame** | ⏸️ **deferred** | needs the fame signal (M8, scoped out → Plan §11) |
| **H4** | Faithfulness depends on **conflict type** | ✅ **supported** | C1 0.936 · C2 0.971 · **C3 0.366** (only class where memorised > faithful) |

> [!tip] The story in one line
> TISER turns Qwen into a **grounded reader** but not a **genuine auditor**: it follows the text without realising the text contradicts it, except on event-ordering conflicts, where memory wins.

---

## 3. Method (as actually run)

**Subject.** Our frozen baseline: `Qwen2.5-7B-Instruct` + attention-only LoRA SFT (TISER reproduction). Full-test scores: **macro-EM 0.878 / macro-F1 0.949** (`…/baseline_full/tiser_qwen7b_full_FULL/…/metrics.json`; per-split: L2 0.907, L3 0.961, TimeQA-easy 0.980, TimeQA-hard 0.970). The probe is **inference-only**, nothing is re-trained.

**Pipeline (6 stages + analysis).** Each consumes the previous artifact and writes a new one + `run_meta.json`:

```
M0 subset ─► M1 closed-book memory ─► M2 eligibility (YIELD GATE) ─► M3 conflict set
   ─► M4 run inputs ─► M5 run matrix (GPU) ─► M6 faithfulness scoring ─► M10 analysis
```

**The eligibility gate (the core idea).** An item is usable only if the model **truly knows** the fact:
- **Memory `m`:** ask the question **closed-book** (delete the `Temporal context:` block — the answer span lived only there, so this is a genuine memory test, not leakage). 5 samples at T=0.7; `m` = majority answer, kept iff it appears in **≥3/5** samples.
- **Eligible:** keep iff `m` is **confident** (≥3/5) **and correct** (`m == gold`). On the kept set `m == gold`, so one conflict set serves every run-matrix cell.
- **Gate:** GO iff each primary class (C1, C2) has ≥150 eligible items. → **Passed: 571 eligible.**

**Conflict types (the labelled independent variable).** Deterministic string edits on the templated contexts:

| Class | Edit | Applies to | Probes |
|---|---|---|---|
| **C1 date-shift** | move date ranges so a different entity covers the queried time | L2, TimeQA | time-event binding |
| **C2 entity-swap** | replace the answer entity with a same-type distractor | L2, L3, TimeQA | factual substitution |
| **C3 order-reversal** | swap intervals so a *before/after* relation flips | L3 only | event-event ordering |
| **control** | reorder rows but keep the answer unchanged | all | sanity floor |

**Run matrix (2×2).** Model axis = *does SFT change faithfulness?*; prompt axis = *does the reasoning machinery change faithfulness?* Only TISER-prompt cells emit a `<reflection>`, so H2 is read only on those.

|  | plain prompt | TISER prompt |
|---|:--:|:--:|
| **vanilla Qwen** (`base`) | ✓ | ✓ |
| **our fine-tune** (`tiser`) | ✓ | ★ star cell |

---

## 4. Worked examples: dataset sample → transformed input → model behaviour

> [!important] Two different "samples", don't confuse them
> **(A) Dataset sample** = the original row from `TISER_test.json` (true context, real answer). **(B) Transformed input** = what we actually feed the model: either the *closed-book* prompt (context deleted, to measure memory) or the *perturbed* prompt (context edited to lie). The model never sees the dataset row as-is in the probe.

All three examples are **unanimous-memory** items (the model answered the closed-book question identically in all 5/5 samples, the strongest "it really knows this" evidence).

### Example A: Valentina Tereshkova (TempReason L2) → silent override

**(A) Dataset sample** — `id: L2_Q44371_P102_0`
```
Question:     Which political party did Valentina Tereshkova belong to in Jun, 1985?
Real gold:    Communist Party of the Soviet Union
True context: Communist Party of the Soviet Union (1960–1991); Our Home–Russia (1995–2003);
              Russian Party of Life (2003–2008); United Russia (2008–2022)
```

**(B1) Memory probe — closed-book input** (context deleted):
```
...[TISER 4-tag instruction]...
Question: Which political party did Valentina Tereshkova belong to in Jun, 1985?
### Answer:
→ <answer> Communist Party of the Soviet Union </answer>   (5/5 samples) ⇒ m = gold ✅ KEPT
```

**(B2) Transformed input: C1 date-shift** (move the famous party off the 1985 slot):
```
ctx' : Communist Party... (1995–2003); Our Home–Russia (1960–1991); ...
⇒ context now answers "Our Home–Russia"   (≠ memory)
```

**Model outputs (C1):**

| cell | answer | faithful | note |
|---|---|:--:|---|
| ★ tiser×tiser | "Our Home–Russia" | ✅ | reflection: *"June 1985 is within Jan 1960–Jan 1991 … There are no errors."* → **silent override** |
| tiser×standard | "…the Communist Party of the Soviet Union…" | ✗ | plain prompt reverts to memory |
| base×standard | "…did not belong to any…party…" | ✗ | vanilla confused |

**C2 entity-swap** (`ctx'` puts an absurd "University of Chicago" in the slot): the fine-tune **still obeys** → answers "University of Chicago", reflection *"confirms that Valentina Tereshkova was a member of the University of Chicago … no errors."* The model follows even a *nonsensical* edit, and never flags it.

### Example B: Kyriakos Velopoulos (TempReason L3) → the C3 failure

**(A) Dataset sample**: `id: L3_Q6452486_P102_2`
```
Question:     Which political party did Kyriakos Velopoulos belong to BEFORE Greek Solution?
Real gold:    New Democracy
True context: Popular Orthodox Rally (2004–2012); New Democracy (2012–2015); Greek Solution (2016–2022)
```
Closed-book: `<answer> New Democracy </answer>` (5/5) ✅ KEPT.

**(B) Transformed input: C3 order-reversal** (swap the two earlier intervals so the *before* answer flips):
```
ctx' : Popular Orthodox Rally (2012–2015); New Democracy (2004–2012); Greek Solution (2016–2022)
⇒ context now answers "Popular Orthodox Rally"   (memory still says "New Democracy")
```

**Model outputs (C3):**

| cell | answer | faithful | memorised |
|---|---|:--:|:--:|
| ★ tiser×tiser | **"New Democracy"** | ✗ | ✅ |
| base×tiser | "Popular Orthodox Rally" | ✅ | ✗ |

★ reflection: *"New Democracy is the correct answer, as it precedes the Popular Orthodox Rally, which in turn precedes Greek Solution. There are no errors."*, the fine-tune **ignores the reordered dates, reverts to the remembered order, and invents a justifying sequence the context does not state.** Strikingly, **vanilla Qwen reads the dates correctly here.** On order-reversal, the SFT *hurts*.

### Example C: Sewell Chan (TimeQA easy) → clean faithfulness + the control

**(A) Dataset sample**: id: /wiki/Sewell_Chan#P108_easy_1`
```
Question:     Who did Sewell Chan work for from 2004 to 2018?
Real gold:    The New York Times
True context: 2000–2004 The Washington Post; 2004–2018 The New York Times; 2018–2019 Los Angeles Times
```
Closed-book: `<answer> The New York Times </answer>` (5/5) ✅ KEPT.

- **C1** (`ctx'` gives the 2004–2018 slot to "The Washington Post"): **all 4 cells answer "The Washington Post"** ✅ — date-shift on a clean 3-row timeline is easy for everyone.
- **C2** (swap NYT → "US National Institute of Allergy and Infectious Diseases"): **all 4 cells obey** ✅ even though it is absurd.
- **control** (reorder rows, answer unchanged): **all 4 cells answer "The New York Times"** (faithful = memorised = 1) → proves the *edits don't break the context*; a low C3 score is a real conflict effect, not edit damage.

---

## 5. Results

### 5.1 Pipeline funnel (what survived each stage)

| Stage                                   |        Count |                                                             |
| --------------------------------------- | -----------: | ----------------------------------------------------------- |
| M0 in-scope items (100% context-parse)  |   **15,898** | L2 5,397 · L3 4,426 · TimeQA-easy 2,997 · TimeQA-hard 3,078 |
| M2 eligible (confident **and** correct) |      **571** | of these, 131 are unanimous (5/5), 205 at 4/5, 235 at 3/5   |
| confident-but-wrong (bonus bucket)      |        2,101 | model insists on a *false* fact → no clean conflict         |
| M3 conflict rows built                  |    **1,176** | C1 203 · C2 520 · C3 333 · control 120 (100% validity)      |
| M5 generations per cell                 | **1,176 ×4** | tiser/base × tiser/standard                                 |

### 5.2 Headline run matrix (M6)

| Cell (model × prompt) | n | **faithful-EM** | faithful-F1 | memorised-EM | malformed | reflection-mention |
|---|---:|---:|---:|---:|---:|---:|
| **tiser × tiser** ★ | 1176 | **0.787** | 0.917 | 0.230 | 0.002 | **0.038** |
| tiser × standard | 1176 | 0.574 | 0.771 | 0.297 | 0.000 | — |
| base × tiser | 1176 | 0.561 | 0.739 | 0.295 | 0.003 | 0.121 |
| base × standard | 1176 | 0.380 | 0.605 | 0.332 | 0.000 | — |

- **Prompt axis (H1):** 0.787 − 0.574 = **+0.213**.
- **Model axis (SFT):** 0.787 − 0.561 = **+0.226**.
- **H2:** the most faithful cell mentions the conflict the *least* (3.8%); vanilla base *talks* about it more (12.1%) yet is *less* faithful → mentioning ≠ resolving.

### 5.3 Per-class (★ star cell)

| Class | n | faithful-EM | memorised-EM | reading |
|---|---:|---:|---:|---|
| **C1** date-shift | 203 | 0.936 | 0.010 | follows the text |
| **C2** entity-swap | 520 | 0.971 | 0.002 | follows the text |
| **C3** order-reversal | 333 | **0.366** | **0.477** | **memory wins** |
| **control** | 120 | 0.900 | 0.900 | sanity floor holds |

### 5.4 Cross-model memory overlap (color, not a gate)

When **both** models are confident closed-book, they agree **81%** (222/274), world knowledge is largely shared (consistent with attention-only LoRA), and the SFT roughly **doubles** closed-book confidence (base confident on only 274/571 of the eligible facts). On the full 15,898 the raw overlap is 11.9%, noise-dominated, ignore it.

---

## 6. What we conclude

- **H1: TISER is a more faithful reader.** The timeline+reflection scaffold forces the model onto the text; strip it (plain prompt) and memory reasserts itself. The SFT compounds this (+0.226).
- **H2: but it is silent override, not auditing.** Every faithful reflection in §4 says *"there are no errors"*, it never says *"this contradicts what I know."* Faithfulness is **incidental, not reasoned**. This is the sharpest result: it bounds the value of the reflection stage the TISER paper sells.
- **H4: order-reversal is the blind spot.** C1/C2 (entity/date binding) are nearly solved; C3 (event-event ordering) is where the fine-tune reverts to remembered order and even fabricates a supporting timeline (Example B). 
- **Control validates the instrument:** edits don't break contexts (0.900 floor), so the C3 drop is a genuine conflict effect.

---

## 7. Caveats (state these in the write-up)

1. **Single-decode point estimates.** All numbers are one greedy decode, no CIs yet. The **analysis phase** adds bootstrap CIs + a paired **McNemar** test (H1 prompt-axis, model-axis) — cheap, CPU-only, no re-generation.
2. **H2 rests on a lexical proxy.** "Mentions conflict" = a keyword match over `<reflection>`. It's brittle both ways; an LLM judge would harden it (future work). The qualitative reading in §4 backs the proxy.
3. **The eligible set is a stress sample.** By construction it is **well-known entities** (strongest memory, hardest faithfulness case). Report "78.7% faithful" as *"on the items where memory should fight hardest,"* not a general rate. (This bias is also exactly the H3 fame gradient — see Plan §11.)
4. **A normalisation edge case.** Some gold strings carry a mojibake (`u2013` for an en-dash); vanilla base sometimes outputs the *correct* en-dash and is wrongly scored unfaithful (seen in Example A / C1, base cell). The analysis pass applies a Unicode-normalising EM — it can only *raise* the base cells slightly, not change the story.

---

## 8. Asset map (every claim is backed by a file)

> Repo branch `ext/context-memory-conflict`; artifacts under `outputs/conflict/<stage>/`. `outputs/` is gitignored → only the small JSON reports are committed; the large `*.jsonl` live locally / on the pod.

| Phase | Module | Key artifact(s) | rows | what it holds |
|---|---|---|---:|---|
| **M0** subset | `src/conflict/subset.py` | `subset/items.jsonl` · `subset_report.json` | 15,898 | parsed in-scope items + structured `ctx_events` + applicable classes |
| **M1** memory | `src/conflict/memory.py` | `memory/memory_{tiser,base}.jsonl` *(on RunPod)* | 15,898 | closed-book `m`, 5 samples, agreement, greedy — **per-sample traces are on the pod** |
| **M2** eligibility | `src/conflict/eligibility.py` | `eligible/eligible.jsonl` | **571** | the kept set: `m`, `agreement`, `prompt_no_context`, `applicable_classes` |
| **M3** conflicts | `src/conflict/perturb.py` | `conflicts/conflict_set.jsonl` · `conflict_report.json` · `conflict_sample.jsonl` | 1,176 (+30 sample) | `ctx_prime`, `answer_ctx_prime`, `m`, `edit_meta`, per-class validity + drop reasons |
| **M4** run inputs | `src/conflict/prompts.py` | `run_inputs/{tiser,standard}.jsonl` | 1,176 ×2 | exact prompts fed to the model, per style |
| **M5** generations | `src/conflict/run.py` | `generations/{tiser,base}__{tiser,standard}.jsonl` | 1,176 ×4 | raw model completions + `run_meta.json` (git SHA, vLLM ver) |
| **M6** scored | `src/conflict/score.py` | `scored/<cell>.jsonl` + `<cell>.metrics.json` | 1,176 ×4 | per-row faithful/memorised/guardrail EM+F1, `reflection_text`, mention flag |
| **baseline** | (frozen) | `…/baseline_full/tiser_qwen7b_full_FULL/…/metrics.json` + `predictions.jsonl` | 22,014 | macro-EM 0.878 / F1 0.949, per-split |

Thin CLIs: `scripts/conflict/0{1..6}_*.py`. Config: `config/conflict.yaml`. Stage reports carry the headline counts (`subset_report.json`, `conflict_report.json`, `<cell>.metrics.json`).

---

## 9. Status & what's next

- **Experimenting phase: CLOSED**, M0→M6 complete; H1/H2/H4 answered.
- **Analysis phase (current):** aggregate the four `scored/*` files → bootstrap CIs, paired McNemar (H1 + model axis), per-class breakdown (the C3 inversion), Unicode-EM fix, a small reflection gallery for silent override. **No new generations.**
- **Future Paths (scoped out):** LLM conflict-detection judge (hardens H2), entity-fame (unlocks H3), human validation, GPT-4o ceiling. Full rationale in [[Extension_ContextMemoryConflict_ExperimentPlan]] §11.

---

## 10. Reproducibility

- Each stage writes a versioned artifact + `run_meta.json` (git SHA, lib versions incl. vLLM, resolved config) under `outputs/conflict/<stage>/`.
- M1/M5 GPU runs: **vLLM on H100 SXM**, bf16, `VLLM_USE_DEEP_GEMM=0`; adapter at `model/tiser_qwen7b_full/adapter`. `base` = plain Qwen2.5-7B-Instruct.
- Design/decisions: [[Extension_ContextMemoryConflict_ExperimentPlan]]. Motivation/prior work: [[Extension_ContextMemoryConflict_Presentation]].
