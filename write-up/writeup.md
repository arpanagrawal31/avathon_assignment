# Enterprise Email Triage via QLoRA Fine-Tuning of a 3B SLM

**Author:** Arpan Agrawal  **Track:** B — SLM Fine-Tuning (Gen AI)  **Scenario:** S2 — Gen AI for Enterprise Documents

---

## 1 · Problem & Domain

**The problem.** Operations teams triage a firehose of internal email by hand: what is this asking for, how urgent is it, who owns the follow-up, does it need to be escalated. I turn a single cleaned email body into strict, machine-actionable JSON against a fixed schema, emitting `null` / empty when a field is genuinely absent rather than guessing:

```json
{
  "intent": "request | status_update | scheduling | approval_request | escalation | fyi | other",
  "urgency": "low | medium | high",
  "requires_response": true,
  "action_items": [ {"owner": "string|null", "task": "string", "deadline": "string|null"} ],
  "escalation_flag": false
}
```

**Data source.** The public **Enron email corpus** (FERC release; Kaggle `wcukierski/enron-email-dataset`, ~517k messages). A load-bearing design choice: **`owner` and `deadline`, when non-null, must be verbatim substrings of the email.** That turns hallucination into a free mechanical check (`span in body`) and forces grounding rather than plausible invention. `task` may be paraphrased.

**Why this is a real AI opportunity, and why Track B (fine-tuning) fits.** Triage is high-volume, repetitive, and structured — the shape that pays back automation — but the hard part is *behavioural*: rigid schema adherence, a consistent 7-way taxonomy, and calibrated abstention. Also helps in keeping corporate emails on-prem vs exposing to 3P API.

---

## 2 · Approach & Algorithm Decisions

### Fine-tuning vs. prompting / few-shot / RAG / zero-shot

I ran the zero-shot base model through the identical harness first, so this is measured, not asserted. Prompting alone leaves the base Qwen2.5-3B at **86% schema validity** and **26% intent accuracy** — it drifts across the taxonomy.

- **RAG is wrong by construction.** There is no external corpus to retrieve — the email *is* the whole context.
- **Few-shot** burns context window per call and still does not lock output structure or taxonomy at 3B scale.


**Structured-output tooling with schema validation** would constrain the *space* of outputs; it cannot improve the *choice* within that space. That semantic judgment requires **weight adaptation**.

### Base model: Qwen2.5-3B-Instruct

Constraint: ≤13B, open-weight, must fit QLoRA on a free T4/P100. Rejected alternatives:

- **Mistral-7B-v0.3** — weaker structured-output priors; 7B QLoRA roughly triples epoch time on a T4, which would have cost me the ablation within the time budget.
- **Llama-3.2-3B-Instruct** — genuinely close. Chose Qwen2.5-3B for stronger out-of-the-box JSON adherence, which raises the *baseline* floor and makes the fine-tuning delta a cleaner read.

### QLoRA vs. full fine-tuning vs. other PEFT

Full fine-tuning of 3B needs ~40 GB+ in optimizer states alone; a T4 has 16 GB — infeasible. **QLoRA** (4-bit NF4 base + double quant, fp16 LoRA adapters) fits comfortably. Among PEFT methods, **IA³** has far fewer trainable params but less capacity to learn a 7-way taxonomy; **prompt-tuning** adds soft tokens without touching the weights that carry the semantic mapping — both rejected on capacity, not format.

### Rank / alpha / targets and the ablation

`r=16` balances capacity against overfit at ~360 training examples; `α=32` is the conventional 2x scaling; `dropout=0.05`. I target attention **+ MLP** because MLP layers carry format/style adaptation. I tested that hypothesis directly with an **attention-only** ablation at identical r, seed, data, epochs, LR, and batch.

Full config: `2 epochs · LR 2e-4 cosine, warmup 0.03 · effective batch 16 (bs 2 × grad-accum 8) · paged_adamw_8bit · fp16 · max_seq 1280 · seed 42`.

### Data pipeline & the teacher-label

From 517k emails: exact-dedup and trigram based near-dedup → ~204k clean pool → sample **~360 training** (targeted for 800 samples but stopped early due to rate limit) + **50 held-out** messages. Labels do not exist, so a teacher LLM (**Gemini-3.5-Flash-Lite**) synthesises them; the 50-example test set is then **hand-audited into a gold set** (needed minor corrections in 4-5 examples only).

---

## 3 · Results & Error Analysis

All metrics on the 50-example hand-audited gold set. Base is zero-shot through the identical harness and generation budget. Classification metrics for the base are scored only on its parseable outputs.

| Metric | Base (0-shot) | Tuned — attn+MLP | Tuned — attn-only |
|---|:---:|:---:|:---:|
| Schema validity | 86% | **100%** | **100%** |
| Intent accuracy (7-way) | 26% | **70%** | 64% |
| Action-item count acc. | 63% | **72%** | **72%** |
| requires_response acc. | 77% | **86%** | 78% |
| Urgency accuracy (3-way) | 79% | 68% | 74% |
| Abstention recall | 88% | 78% | 85% |
| Unsupported-span rate | 0% (15 spans) | 7% (2/27) | 4% (1/25) |
| escalation_flag acc. | 98% | 98% | 98% |

**Clear Improvements:** Schema validity 86 → 100 and intent 26 → 70 are the real capability lift — the model learned to always emit valid JSON and to internalise the taxonomy, neither of which prompting delivered.

**Grey Area:** Fine-tuning made the model a more *active extractor*: it predicts many more action-item spans (15 → 27), gets the count right more often (63 → 72), but abstains slightly less (88 → 78/85) and occasionally grabs a non-verbatim span (0 → 4–7%). Base Qwen is **conservative** — it abstains well and rarely hallucinates a span, but cannot produce valid JSON or the taxonomy; the tuned model is **assertive** — it locks format and learns intent, at a small precision/grounding cost.

**Where it fails, and why.**

1. **Urgency did not improve** (79 → 68/74). Urgency is subjective; could be due to *teacher-label noise* not a model regression.
2. **Unsupported spans rose** (0 → 4–7%). The assertive shift trades a little grounding precision for recall.
3. **`escalation_flag` is uninformative.** 98% is majority-class dominated (escalations are rare in the gold set); probably F1 could be a better metric here.

**Ablation.** No clean winner: attn+MLP leads intent (+6) and requires_response (+8); attn-only leads urgency (+6) and abstention (+7) with fewer unsupported spans at **~1/4 the trainable params** (~7.4M vs ~30M). Both hit 100% schema. The honest call is **attn-only is roughly as good for a quarter of the cost.** A larger eval set could help with significance of these small differences in performance.

**Learning vs. memorization.** Every number above is on emails the model **never saw in training** — the 50 gold messages are sampled disjoint from the ~360 training messages and hand-audited, so the gains are generalisation. Overfitting pressure is low by design (2 epochs, ~1% of params, dropout 0.05). The stronger control I would add next is an **entity-perturbation test** — swap names/dates/companies in the gold set and confirm scores hold; a memoriser would collapse on grounding, a learner would not. I expect it to hold, since `owner`/`deadline` are learned as a copy-from-context operation, not memorised entities.

### 3a · Hallucination risk — baseline vs. fine-tuned

The grounding rule the hallucination detector: a non-null `owner`/`deadline` must be a verbatim substring of the source email, so `unsupported_span_rate` is a deterministic hallucination measure.

| Model | owner/deadline spans | Ungrounded (fabricated) | Rate |
|---|:---:|:---:|:---:|
| Base (zero-shot) | 25 | 0 | 0.0% |
| Tuned — attn+MLP | 27 | 2 | **7.4%** |
| Tuned — attn-only | 25 | 1 | 4.0% |

**Fine-tuning introduced a hallucination risk the base model did not have.** The base fabricated *zero* spans — because it is conservative and predicts few spans at all. The tuned model extracts more aggressively (25 → 27 spans) and, in doing so, occasionally asserts an `owner`/`deadline` that is not literally in the email. This is the precision cost of the recall-leaning shift, and it is exactly where a fine-tuned extractor is most dangerous: a fabricated owner or deadline is a *confident, plausible, wrong* operational instruction.

**Mitigation** The grounding rule is enforced at the serving boundary: any ungrounded `owner`/`deadline` is coerced to `null` (the model abstains on that field rather than asserting a fabrication). `task` is grounding-exempt by design and never altered. This drives the unsupported-span rate to an **enforced 0%** with a single auditable rule — a concrete, zero-model structured-output guard rather than a hope that the weights behave.

### 3b · Catastrophic forgetting probe

I probed 15 general MCQs (MMLU-style knowledge + reasoning) and 10 open-ended instruction-following prompts, scored base vs. tuned through the same harness:

| | Base | Tuned — attn+MLP | Tuned — attn-only |
|---|:---:|:---:|:---:|
| General MCQ accuracy | 86.7% (13/15) | 86.7% (13/15) | 86.7% (13/15) |
| JSON-bleed rate (open-ended) | 0/10 | 0/10 | 0/10 |
| Instruction-following | coherent | coherent | coherent |

**No measurable regression.** General MCQ accuracy is unchanged, and critically the **JSON-bleed rate is 0%** — asked for a haiku, the tuned model still writes a haiku, not triage JSON. Both ablation variants behave identically.

---

## 4 · Production & Limitations

**Production consideration — a validation boundary.** Serve the adapter merged into a 4-bit 3B on a single T4 with batched generation. Because the schema is a hard contract, the format layer should be **grammar-constrained decoding** (guaranteed-valid JSON and enums on the first pass, no wasted retries), and the content layer the **grounding guard** (null any `owner`/`deadline` that is not a verbatim span). Together these convert the residual 7% unsupported-span rate into a guaranteed-valid, guaranteed-grounded external contract — the fine-tuned weights supply the *semantics*, structured-output tooling supplies the *guarantees*.

**Limitation to address before real deployment.** Performance is **ceilinged by silver-label quality** — the model can be at most as good as the teacher, and the flat urgency metric is that ceiling showing through on a subjective field. The next step is a modest set of **human-labelled in-house emails** to fine-tune on the domain-real distribution and re-anchor the subjective fields, plus multi-seed evaluation on a larger gold set to turn the ablation sub-deltas and the forgetting/hallucination bounds into real findings.
