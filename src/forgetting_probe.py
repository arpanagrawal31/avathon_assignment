"""
Catastrophic-forgetting probe: does the QLoRA fine-tune degrade general capability?

Runs the SAME prompts through the base model (no adapter) and the tuned model
(base + adapter), on two axes:

  1. Capability retention  -- 15 general-knowledge / reasoning MCQs, exact-match
     on the letter. Answers "did world knowledge / reasoning survive?" with a number.

  2. Format reflex ("JSON bleed") -- 10 open-ended instruction-following prompts
     that expect a natural-language answer. We flag every response that spuriously
     emits triage-schema JSON instead of prose. This is the forgetting mode a narrow
     JSON fine-tune actually produces: the model over-associates *any* prompt with
     "reply in the triage schema". The base model's bleed rate is the control (~0).

Note on mitigation for the write-up: LoRA adapters are separable and the base
weights are frozen, so any regression here is recoverable at serving time (detach
the adapter, or route general queries to the base model). Full fine-tuning could
not make that claim.

Usage (Kaggle, GPU):
    !python src/forgetting_probe.py --variant attn_mlp
    !python src/forgetting_probe.py --variant attn_only --adapter path/to/dir
Writes results/forgetting_probe_<variant>.json and results/forgetting_probe.md
(a human-readable side-by-side, appended per variant).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID       = "Qwen/Qwen2.5-3B-Instruct"
PROBE_PATH     = "data/forgetting_probe.jsonl"
OUTPUT_DIR     = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
MAX_NEW_TOKENS = 256
# General assistant framing -- deliberately NOT the triage system prompt, so we are
# testing the model as a general assistant, which is the capability at risk.
GEN_SYSTEM     = "You are a helpful assistant."
# Triage-schema keys: their presence in a prose answer is the JSON-bleed signal.
SCHEMA_KEYS    = ("intent", "urgency", "requires_response", "action_items", "escalation_flag")


def load_base():
    """4-bit base model, no adapter."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    return model, tok


def attach_adapter(base_model, adapter_dir: str):
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model


def generate(model, tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


# ── Scoring ─────────────────────────────────────────────────

def score_mcq(response: str, answer: str) -> bool:
    """First standalone A-D letter in the response must equal the gold letter."""
    m = re.search(r"\b([ABCD])\b", response.upper())
    return m is not None and m.group(1) == answer.upper()


def has_json_bleed(response: str) -> bool:
    """
    True if a natural-language answer instead emitted triage-schema JSON.
    Signal = the response contains a JSON object AND >=2 schema keys, or it
    starts with '{'. Two keys avoids flagging an incidental word like 'intent'.
    """
    stripped = response.lstrip()
    if stripped.startswith("{") or stripped.startswith("```"):
        return True
    key_hits = sum(1 for k in SCHEMA_KEYS if f'"{k}"' in response)
    return key_hits >= 2


def run_probe(model, tokenizer, probe) -> dict:
    rows = []
    mcq_total = mcq_correct = 0
    open_total = bleed_count = 0
    for item in probe:
        resp = generate(model, tokenizer, item["prompt"])
        row = {"type": item["type"], "prompt": item["prompt"], "response": resp}
        if item["type"] == "mcq":
            mcq_total += 1
            row["gold"] = item["answer"]
            row["correct"] = score_mcq(resp, item["answer"])
            mcq_correct += int(row["correct"])
        else:
            open_total += 1
            row["json_bleed"] = has_json_bleed(resp)
            bleed_count += int(row["json_bleed"])
        rows.append(row)
    return {
        "mcq_total": mcq_total,
        "mcq_correct": mcq_correct,
        "mcq_accuracy": round(mcq_correct / mcq_total, 4) if mcq_total else None,
        "open_total": open_total,
        "json_bleed_count": bleed_count,
        "json_bleed_rate": round(bleed_count / open_total, 4) if open_total else None,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["attn_mlp", "attn_only"], default="attn_mlp")
    parser.add_argument("--adapter", default=None,
                        help="Override adapter dir (default adapter/qlora-r16-<variant>)")
    args = parser.parse_args()

    adapter_dir = args.adapter or f"adapter/qlora-r16-{args.variant}"
    if not Path(adapter_dir).exists():
        sys.exit(f"Adapter dir not found: {adapter_dir}")

    probe = [json.loads(l) for l in open(PROBE_PATH) if l.strip()]
    print(f"Loaded {len(probe)} probe prompts "
          f"({sum(1 for p in probe if p['type']=='mcq')} mcq, "
          f"{sum(1 for p in probe if p['type']=='open')} open)")

    # Base run, then attach adapter and re-run the same base object.
    base_model, tok = load_base()
    print("\n[1/2] Probing BASE model...")
    t0 = time.time()
    base_res = run_probe(base_model, tok, probe)
    print(f"  base MCQ acc {base_res['mcq_accuracy']:.1%}, "
          f"JSON-bleed {base_res['json_bleed_rate']:.1%}  ({time.time()-t0:.0f}s)")

    print("\n[2/2] Probing TUNED model...")
    tuned_model = attach_adapter(base_model, adapter_dir)
    t0 = time.time()
    tuned_res = run_probe(tuned_model, tok, probe)
    print(f"  tuned MCQ acc {tuned_res['mcq_accuracy']:.1%}, "
          f"JSON-bleed {tuned_res['json_bleed_rate']:.1%}  ({time.time()-t0:.0f}s)")

    out = {"variant": args.variant, "adapter": adapter_dir,
           "base": base_res, "tuned": tuned_res}
    json_path = OUTPUT_DIR / f"forgetting_probe_{args.variant}.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {json_path}")
