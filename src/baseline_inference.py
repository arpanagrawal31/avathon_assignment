"""
Baseline inference: zero-shot on Qwen2.5-3B-Instruct (4-bit).
Run on Kaggle T4. Produces results/baseline_gold.json.
"""
# !pip install transformers accelerate bitsandbytes pydantic -q

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(".").resolve()))
from src.prompts import BASELINE_SYSTEM, BASELINE_PROMPT
from src.evaluate import evaluate, print_summary

# ── Config ──────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen2.5-3B-Instruct"
GOLD_PATH        = "data/gold/gold_teacher_labeled.jsonl"
OUTPUT_DIR       = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
MAX_NEW_TOKENS   = 512


def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
        attn_implementation="sdpa",
    )
    mdl.eval()
    return mdl, tok


def generate(model, tokenizer, messages: list[dict]) -> str:
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
                            skip_special_tokens=True)


def build_messages(email_body: str | None = None):
    msgs = [{"role": "system", "content": BASELINE_SYSTEM}]
    msgs.append({"role": "user",
                  "content": BASELINE_PROMPT.format(email_body=email_body)})
    return msgs


def run_inference(model, tok, gold):
    preds = []
    for i, rec in enumerate(gold):
        print(f"  [{i+1}/{len(gold)}] idx={rec['orig_idx']}",
              end=" ", flush=True)
        t0 = time.time()
        raw = generate(model, tok, build_messages(rec["body"]))
        print(f"({time.time()-t0:.1f}s)")
        preds.append({"orig_idx": rec["orig_idx"],
                       "body": rec["body"], "raw_output": raw})
    return preds


def save_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def run_and_eval(model, tok, gold):
    """Run inference + evaluate + save everything."""
    print(f"\n{'-'*50} BASELINE {'-'*50}")
    preds = run_inference(model, tok, gold)

    pred_path = str(OUTPUT_DIR / "baseline_gold_preds.jsonl")
    save_jsonl(preds, pred_path)

    results = evaluate(preds, gold)
    results["metadata"] = {"model": MODEL_ID}
    out_path = str(OUTPUT_DIR / "baseline_gold.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results["metrics"])


if __name__ == "__main__":
    model, tok = load_model()

    with open(GOLD_PATH) as f:
        gold = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(gold)} gold examples")

    run_and_eval(model, tok, gold)

    print(f"\n{'='*50} DONE {'='*50}")
