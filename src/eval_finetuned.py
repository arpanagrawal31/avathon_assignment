"""
Fine-tuned inference + evaluation: Qwen2.5-3B-Instruct + QLoRA adapter -> gold set.

Loads the 4-bit base model with the trained LoRA adapter and runs the SAME
prompt used in training and in baseline_inference.py, with the SAME generation
budget (greedy, max_new_tokens=512), so base-vs-tuned is a controlled comparison.

Usage (Kaggle):
    !python src/eval_finetuned.py                      # attn_mlp  (run #1)
    !python src/eval_finetuned.py --variant attn_only  # ablation
    !python src/eval_finetuned.py --adapter path/to/dir --tag custom
Writes results/finetuned_r16_<variant>.json (+ _preds.jsonl).
"""
# !pip install transformers accelerate bitsandbytes peft pydantic -q

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(".").resolve()))
from src.prompts import BASELINE_SYSTEM, BASELINE_PROMPT
from src.evaluate import evaluate, print_summary

# ── Config ──────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen2.5-3B-Instruct"
GOLD_PATH      = "data/gold/gold_teacher_labeled.jsonl"
OUTPUT_DIR     = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
MAX_NEW_TOKENS = 512      # same budget as baseline_inference.py -> fair comparison


def load_model(adapter_dir: str):
    """4-bit base + LoRA adapter. Tokenizer from the adapter dir (saved by train.py)."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tok


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


def build_messages(email_body: str) -> list[dict]:
    # No few-shot exemplars: the fine-tuned model was trained on exactly this
    # system+user format, so zero-shot here mirrors training.
    return [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user",   "content": BASELINE_PROMPT.format(email_body=email_body)},
    ]


def save_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["attn_mlp", "attn_only"],
                        default="attn_mlp",
                        help="Which trained adapter to evaluate")
    parser.add_argument("--adapter", default=None,
                        help="Override adapter dir (default adapter/qlora-r16-<variant>)")
    parser.add_argument("--tag", default=None,
                        help="Override output tag (default r16_<variant>)")
    args = parser.parse_args()

    adapter_dir = args.adapter or f"adapter/qlora-r16-{args.variant}"
    tag         = args.tag or f"r16_{args.variant}"

    if not Path(adapter_dir).exists():
        sys.exit(f"Adapter dir not found: {adapter_dir}  "
                 f"(run train.py --variant {args.variant} first)")

    print("=" * 55)
    print(f"  Fine-tuned eval -- adapter: {adapter_dir}")
    print("=" * 55)

    model, tok = load_model(adapter_dir)

    with open(GOLD_PATH) as f:
        gold = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(gold)} gold examples\n")

    preds = []
    for i, rec in enumerate(gold):
        print(f"  [{i+1}/{len(gold)}] idx={rec['orig_idx']}", end=" ", flush=True)
        t0 = time.time()
        raw = generate(model, tok, build_messages(rec["body"]))
        print(f"({time.time()-t0:.1f}s)")
        preds.append({"orig_idx": rec["orig_idx"],
                      "body": rec["body"], "raw_output": raw})

    pred_path = str(OUTPUT_DIR / f"finetuned_{tag}_preds.jsonl")
    save_jsonl(preds, pred_path)

    results = evaluate(preds, gold)
    results["metadata"] = {"model": MODEL_ID, "adapter": adapter_dir,
                           "mode": f"finetuned_{tag}"}
    out_path = str(OUTPUT_DIR / f"finetuned_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results["metrics"])          # NOTE: takes ONE arg (see evaluate.py)

    print(f"\n{'='*55}")
    print(f"  DONE -> {out_path}")
    print(f"         {pred_path}")
    print(f"{'='*55}")
    print("  Compare vs results/baseline_gold.json")
