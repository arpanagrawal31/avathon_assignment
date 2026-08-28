"""
QLoRA fine-tuning: Qwen2.5-3B-Instruct -> email triage extraction. Kaggle T4 GPU

Hyperparameters:
  r=16, alpha=32, attn+MLP targets, 2 epochs, paged_adamw_8bit, fp16, seed=42.

Usage (Kaggle):
    !python src/train.py                      # variant=attn_mlp (default, run #1)
    !python src/train.py --variant attn_only  # ablation: attention-only, r=16
Resume after a disconnect:
    !python src/train.py [--variant ...]      # auto-detects the latest checkpoint

The two variants write to separate checkpoint/adapter dirs, so the ablation
run never clobbers run #1 and each resumes independently. Everything else
(seed, data, epochs, LR, batch) is identical -> a clean attn+MLP vs attn-only
comparison, as the plan's ablation requires.

Pinned stack this script targets (current TRL SFTConfig API):
    transformers>=4.46  trl>=0.12  peft>=0.13  accelerate>=1.0
    bitsandbytes>=0.44  datasets>=3.0
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(".").resolve()))
from src.prompts import BASELINE_SYSTEM, BASELINE_PROMPT

# -- Config ------------------------------------------------------------------
MODEL_ID          = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_PATH        = "data/train_labeled.jsonl"

# -- Ablation variants (r=16 for both; only target_modules differ) -----------
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP  = ["gate_proj", "up_proj", "down_proj"]
VARIANT_TARGETS = {
    "attn_mlp":  ATTN + MLP,   # run #1 (default) -- attention + MLP
    "attn_only": ATTN,         # ablation         -- attention only
}

MAX_SEQ_LENGTH = 1280   # raised from 768: distribution check showed p95=1104,
                        # and tail-truncation destroys the JSON target (see
                        # load_training_data). Covers ~95%; the rest are dropped.
BATCH_SIZE     = 2
GRAD_ACCUM     = 8        # effective batch = 16
EPOCHS         = 2
LR             = 2e-4
WARMUP_RATIO   = 0.03
SEED           = 42


# -- Data --------------------------------------------------------------------

def load_training_data(path: str, tokenizer) -> Dataset:
    """Load labeled JSONL -> HF Dataset of chat-formatted text.

    Silently skips unlabeled and malformed rows (there is one corrupt JSON
    line in the current train_labeled.jsonl; it is dropped here, leaving 359).
    """
    raw, skipped = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if d.get("label") is not None and d.get("body"):
                raw.append(d)

    print(f"Loaded {len(raw)} labeled examples from {path} "
          f"(skipped {skipped} malformed lines)")

    texts, lengths = [], []
    for d in raw:
        messages = [
            {"role": "system",    "content": BASELINE_SYSTEM},
            {"role": "user",      "content": BASELINE_PROMPT.format(email_body=d["body"])},
            {"role": "assistant", "content": json.dumps(d["label"], ensure_ascii=False)},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        texts.append(text)
        lengths.append(len(tokenizer.encode(text)))

    srt = sorted(lengths)
    print(f"  Token lengths: min={srt[0]}  med={srt[len(srt)//2]}  "
          f"p95={srt[int(0.95*len(srt))]}  max={srt[-1]}")

    # -- Drop over-length examples instead of truncating them ----------------
    # The sequence ends with the assistant JSON target; tail-truncation would
    # clip/delete that target and leave owner/deadline spans that no longer
    # appear in the (truncated) email -> corrupt, ungrounded labels. Dropping
    # the long tail keeps every retained example's target intact.
    kept = [{"text": t} for t, l in zip(texts, lengths) if l <= MAX_SEQ_LENGTH]
    dropped = len(texts) - len(kept)
    print(f"  Dropped {dropped}/{len(texts)} over {MAX_SEQ_LENGTH} tokens "
          f"({100*dropped/len(texts):.1f}%)  ->  training on {len(kept)}")

    return Dataset.from_list(kept)


# -- Model -------------------------------------------------------------------

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",       # no FlashAttention-2 on T4/P100
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
    )
    model.config.use_cache = False        # required with gradient checkpointing
    return model, tokenizer


# -- LoRA --------------------------------------------------------------------

def build_lora_config(variant: str) -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=VARIANT_TARGETS[variant],
        task_type="CAUSAL_LM",
    )


# -- Training args -----------------------------------------------------------

def make_training_args(n_examples: int, output_dir: str) -> SFTConfig:
    steps_per_epoch = max(n_examples // (BATCH_SIZE * GRAD_ACCUM), 1)
    total_steps = steps_per_epoch * EPOCHS

    # Checkpoint ~every half-epoch, floored so it always fires on a small set.
    # (The plan's save_steps=100 was sized for 800 examples; at 359 it would
    #  never trigger -> no resume point. This keeps the *intent*: save often.)
    save_steps = max(steps_per_epoch // 2, 5)

    print(f"  Steps/epoch:  {steps_per_epoch}")
    print(f"  Total steps:  {total_steps}")
    print(f"  Save every:   {save_steps} steps")

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,                        # T4/P100 have no bf16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=save_steps,
        save_strategy="steps",
        save_total_limit=3,
        logging_steps=5,
        seed=SEED,
        report_to="none",
        dataloader_pin_memory=False,
        # SFT-specific (were direct SFTTrainer kwargs in old TRL):
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )


def find_latest_checkpoint(output_dir: str):
    """Return path to the latest checkpoint, or None."""
    p = Path(output_dir)
    if not p.exists():
        return None
    ckpts = sorted(p.glob("checkpoint-*"),
                   key=lambda x: int(x.name.split("-")[1]))
    return str(ckpts[-1]) if ckpts else None


def assert_mask_matches(dataset, tokenizer, response_template_ids):
    """Fail loudly if the completion-only template never matches.

    DataCollatorForCompletionOnlyLM masks the ENTIRE label to -100 when the
    response-template token IDs are not found as a subsequence -> the model
    trains on nothing and loss stays flat. Verify on one real example first.
    """
    ids = tokenizer.encode(dataset[0]["text"], add_special_tokens=False)
    n = len(response_template_ids)
    found = any(ids[i:i + n] == response_template_ids
                for i in range(len(ids) - n + 1))
    if not found:
        raise RuntimeError(
            "Completion-only response template not found in tokenized example. "
            f"template_ids={response_template_ids}. "
            "Fix the response_template string before training, or every label "
            "will be masked and the model will learn nothing."
        )
    print(f"  Completion-only mask OK (template {response_template_ids} matched)")


# -- Main --------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANT_TARGETS),
                        default="attn_mlp",
                        help="LoRA target set: attn_mlp (run #1) or attn_only (ablation)")
    args = parser.parse_args()
    variant = args.variant

    checkpoint_dir    = f"checkpoints/qlora-r16-{variant}"
    final_adapter_dir = f"adapter/qlora-r16-{variant}"
    lora_config = build_lora_config(variant)

    print("=" * 55)
    print(f"  QLoRA -- Qwen2.5-3B-Instruct -> Email Triage  [{variant}]")
    print("=" * 55)

    print("\n[1/4] Loading model ...")
    model, tokenizer = load_model()

    print("\n[2/4] Loading data ...")
    dataset = load_training_data(TRAIN_PATH, tokenizer)

    print("\n[3/4] Applying LoRA ...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[4/4] Setting up trainer ...")
    training_args = make_training_args(len(dataset), checkpoint_dir)

    # Train only on the assistant JSON, not the system/user prompt.
    # Qwen chat template marks responses with "<|im_start|>assistant\n".
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False,
    )
    assert_mask_matches(dataset, tokenizer, response_template_ids)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,        # was `tokenizer=` in TRL <=0.11
    )

    # -- Log full config -----------------------------------------------------
    print(f"\n{'-'*55}")
    print(f"  Model:           {MODEL_ID}")
    print(f"  Variant:         {variant}")
    print(f"  LoRA r / alpha:  {lora_config.r} / {lora_config.lora_alpha}")
    print(f"  LoRA targets:    {lora_config.target_modules}")
    print(f"  LoRA dropout:    {lora_config.lora_dropout}")
    print(f"  Train examples:  {len(dataset)}")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  Batch (eff):     {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}")
    print(f"  LR:              {LR}  (cosine, warmup {WARMUP_RATIO})")
    print(f"  Optimizer:       paged_adamw_8bit")
    print(f"  Max seq len:     {MAX_SEQ_LENGTH}")
    print(f"  Precision:       fp16    Seed: {SEED}")
    print(f"{'-'*55}")

    resume_from = find_latest_checkpoint(checkpoint_dir)
    print(f"\n>>> {'RESUMING from ' + resume_from if resume_from else 'Starting fresh'}")

    print(f"\n{'='*55}\n  TRAINING START\n{'='*55}\n")
    trainer.train(resume_from_checkpoint=resume_from)

    # -- Save adapter --------------------------------------------------------
    Path(final_adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    print(f"\n{'='*55}\n  TRAINING COMPLETE  [{variant}]\n{'='*55}")
    print(f"  Checkpoints -> {checkpoint_dir}/")
    print(f"  Adapter     -> {final_adapter_dir}/")
    print(f"  Next: fine-tuned inference + evaluate")
