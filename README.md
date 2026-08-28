# Enterprise Email Triage — SLM Fine-Tuning (Track B)

**Author:** Arpan Agrawal · **Track:** B — SLM Fine-Tuning (Gen AI) · **Scenario:** S2 — Gen AI for Enterprise Documents

Fine-tune **Qwen2.5-3B-Instruct** with **QLoRA** to turn a raw corporate email into strict, grounded
triage JSON. Full rationale, results, and error analysis are in [`write-up/writeup.pdf`](write-up/writeup.pdf).

## Problem statement

Given a single cleaned email body, emit JSON against a fixed schema, using `null`/empty when a field is
genuinely absent rather than guessing:

```json
{
  "intent": "request | status_update | scheduling | approval_request | escalation | fyi | other",
  "urgency": "low | medium | high",
  "requires_response": true,
  "action_items": [{"owner": "string|null", "task": "string", "deadline": "string|null"}],
  "escalation_flag": false
}
```

**Grounding rule:** `owner` and `deadline`, when non-null, must be **verbatim substrings** of the email
(`task` may be paraphrased). This makes hallucination a mechanical check (`span in body`).

- **Domain:** enterprise operational document triage.
- **Data source:** public **Enron email corpus** (FERC release; Kaggle `wcukierski/enron-email-dataset`,
  ~517k emails). Labels don't exist → synthesised with a teacher LLM (permitted); the 50-example test set
  is **hand-audited into a gold set**.
- **Why Track B:** the task is behavioural (schema adherence, taxonomy, grounded abstention) and the data
  (corporate email) is privacy-sensitive → an on-prem fine-tuned SLM, not a prompt/RAG/API. See the write-up.

## Repo layout

```
data/   sampling.py, teacher_label.py, train_labeled.jsonl, gold/         # pipeline + labeled sets
src/    schema.py, prompts.py, evaluate.py, baseline_inference.py,
        train.py, eval_finetuned.py                                       # model + harness
results/ baseline_results/  qlora_results/  qlora_adapters/               # all metrics + adapters
write-up/ writeup.pdf                                                     # 2-page write-up
```

## Setup

GPU with compute capability ≥7.0 (Kaggle **T4** recommended; free tier). Python 3.10–3.12.

```bash
pip install -r requirements.txt
pip install -U bitsandbytes    # separate + unpinned so it matches the host CUDA (see requirements.txt)
# On Kaggle: RESTART the session after install before importing torch.
```

`torch` is provided by the Kaggle/Colab GPU image. Teacher labeling additionally needs
`pip install google-generativeai` and a `GOOGLE_API_KEY` (env var or Kaggle secret).

## Reproduce all results end-to-end

Run from the repo root. Steps 1–2 rebuild the dataset from scratch; the committed
`data/train_labeled.jsonl` and `data/gold/` let you skip straight to step 3.

**1 — Sample & clean** (dedup → ~360 train + 50 gold):
```bash
python data/sampling.py          # writes data/train.jsonl, data/gold/gold.jsonl, clean_pool_stats.json
```

**2 — Teacher-label** (labels train + gold; gold is then hand-audited):
```bash
GOOGLE_API_KEY=... python data/teacher_label.py   # -> data/train_labeled.jsonl, data/gold/gold_teacher_labeled.jsonl
```

**3 — Baseline** (zero-shot base model through the eval harness):
```bash
python src/baseline_inference.py   # -> results/baseline_results/baseline_gold.json (+ _preds.jsonl)
```

**4 — Fine-tune** (QLoRA; resumable from the latest checkpoint on re-run):
```bash
python src/train.py                     # run #1: attn+MLP  -> results/qlora_adapters/qlora-r16-attn_mlp
python src/train.py --variant attn_only # ablation: attn-only -> results/qlora_adapters/qlora-r16-attn_only
```

**5 — Evaluate fine-tuned** (same prompt + generation budget as the baseline → controlled comparison):
```bash
python src/eval_finetuned.py                     # -> results/qlora_results/finetuned_r16_attn_mlp.json
python src/eval_finetuned.py --variant attn_only # -> results/qlora_results/finetuned_r16_attn_only.json
```

## Key hyperparameters (logged in `src/train.py`)

QLoRA 4-bit NF4 + double quant, fp16 · `r=16, alpha=32, dropout=0.05, bias=none` ·
targets `q,k,v,o (+ gate,up,down` for attn+MLP`)` · 2 epochs · LR 2e-4 cosine, warmup 0.03 ·
effective batch 16 (bs 2 × grad-accum 8) · `paged_adamw_8bit` · `max_seq_length=1280` · seed 42.

## Results (50-example gold set)

| Metric | Base 0-shot | Tuned attn+MLP | Tuned attn-only |
|---|---|---|---|
| Schema validity | 86% | **100%** | **100%** |
| Intent accuracy (7-way) | 26% | **70%** | 64% |
| Action-item count acc. | 63% | **72%** | **72%** |
| requires_response acc. | 77% | **86%** | 78% |
| Urgency accuracy | 79% | 68% | 74% |
| Abstention recall | 88% | 78% | 85% |
| Unsupported-span rate | 0% | 7% | 4% |

Headline lift: schema validity 86→100 and intent 26→70. Fine-tuning produces a **recall-leaning shift**
(more spans, better count; slightly less abstention, a small grounding cost). Full analysis, the
attn-only ablation cost columns, and the n=50 single-seed caveat are in the write-up.
