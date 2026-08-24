"""
Teacher Labeling (Label train + gold emails with Gemini 3.5 Flash Lite
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from src.schema import TriageOutput, check_grounding, validate_json_string
from src.prompts import TEACHER_SYSTEM, TEACHER_PROMPT

import google.generativeai as genai

try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
except Exception:
    import os
    API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    if not API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY env var or Kaggle secret")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 512
MAX_RETRIES = 5
RATE_LIMIT_DELAY = 4.5        # seconds between calls — free tier is 15 RPM
RATE_LIMIT_BACKOFF = 60       # seconds to wait on 429 before retrying


def label_one_email(
    model: genai.GenerativeModel,
    email_body: str,
) -> tuple[TriageOutput | None, str | None]:
    """Call teacher model, validate output with Pydantic"""
    prompt = TEACHER_PROMPT.format(email_body=email_body)

    for attempt in range(MAX_RETRIES):
        try:
            resp = model.generate_content(
                [
                    {"role": "user", "parts": [{"text": TEACHER_SYSTEM + "\n\n" + prompt}]},
                ],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )

            raw = resp.text.strip()

            # Validate with Pydantic
            label, err = validate_json_string(raw)
            if err:
                raise ValueError(err)

            return label, None

        except Exception as e:
            is_rate_limit = "429" in str(e) or "quota" in str(e).lower()
            if attempt < MAX_RETRIES - 1:
                if is_rate_limit:
                    print(f"    Rate limited, waiting {RATE_LIMIT_BACKOFF}s...")
                    time.sleep(RATE_LIMIT_BACKOFF)
                else:
                    time.sleep(2 ** (attempt + 1))
            else:
                return None, f"Failed after {MAX_RETRIES} attempts: {e}"

    return None, "Exhausted retries"


def label_file(
    input_path: str,
    output_path: str,
    desc: str = "emails",
) -> dict:
    """Label all emails in a JSONL file. Resumable, skips already-done rows"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input emails
    with open(input_path) as f:
        emails = [json.loads(line) for line in f]
    print(f"Labeling {len(emails)} {desc} from {input_path}")
    print(f"Output → {output_path}")

    # Resume: load already-labeled orig_idx values
    done = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["label"] is not None:
                    done[rec["orig_idx"]] = rec
        print(f"  Resuming - {len(done)} already labeled")

    # Init Gemini model
    model = genai.GenerativeModel(MODEL_NAME)

    stats = Counter()
    grounding_stats = Counter()

    with open(output_path, "a") as out_f:
        for i, email in enumerate(emails):
            idx = email["orig_idx"]

            # Skip if already done
            if idx in done:
                stats["skipped"] += 1
                continue

            # Label
            label, err = label_one_email(model, email["body"])

            if err:
                stats["failed"] += 1
                print(f"  ✗ [{i+1}/{len(emails)}] idx={idx}: {err}")
                # Write a record with null label so we can identify failures
                record = {**email, "label": None, "label_error": err}
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                time.sleep(RATE_LIMIT_DELAY)
                continue

            # Grounding check
            grounding = check_grounding(label, email["body"])
            grounding_stats["total_spans"] += grounding["total_spans"]
            grounding_stats["unsupported"] += grounding["unsupported"]

            # Write labeled record
            record = {
                **email,
                "label": label.model_dump(),
                "grounding": {
                    "total_spans": grounding["total_spans"],
                    "unsupported": grounding["unsupported"],
                },
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            done[idx] = record
            stats["labeled"] += 1

            # Progress
            if (stats["labeled"]) % 25 == 0:
                total_done = stats["labeled"] + stats["skipped"]
                print(f"  [{total_done}/{len(emails)}] labeled={stats['labeled']}, "
                      f"failed={stats['failed']}")

            time.sleep(RATE_LIMIT_DELAY)

    total_spans = grounding_stats["total_spans"]
    unsupported = grounding_stats["unsupported"]

    print(f"  Labeled:    {stats['labeled']}")
    print(f"  Skipped:    {stats['skipped']} (already done)")
    print(f"  Failed:     {stats['failed']}")
    if total_spans > 0:
        print(f"  Grounding:  {total_spans} spans checked, "
              f"{unsupported} unsupported "
              f"({100*unsupported/total_spans:.1f}%)")

    return dict(stats)


def print_label_distribution(output_path: str):
    """Print intent/urgency distribution of labeled data."""
    path = Path(output_path)
    if not path.exists():
        return

    intents = Counter()
    urgencies = Counter()
    action_counts = Counter()
    n = 0

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("label") is None:
                continue
            n += 1
            label = rec["label"]
            intents[label["intent"]] += 1
            urgencies[label["urgency"]] += 1
            action_counts[len(label["action_items"])] += 1

    if n == 0:
        return

    print(f"LABEL DISTRIBUTION ({n} examples)")

    print("\n  Intent:")
    for k, v in intents.most_common():
        print(f"    {k:<22s} {v:>4d}  ({100*v/n:.1f}%)")

    print("\n  Urgency:")
    for k, v in urgencies.most_common():
        print(f"    {k:<22s} {v:>4d}  ({100*v/n:.1f}%)")

    print("\n  Action items per email:")
    for k, v in sorted(action_counts.items()):
        print(f"    {k} items{'':<16s} {v:>4d}  ({100*v/n:.1f}%)")

    abstaining = action_counts.get(0, 0)
    print(f"\n  Abstention rate (0 action items): {abstaining}/{n} ({100*abstaining/n:.1f}%)")


if __name__ == "__main__":

    # Label gold test set; teacher labels for agreement measurement
    label_file(
        input_path="data/gold/gold.jsonl",
        output_path="data/gold/gold_teacher_labeled.jsonl",
        desc="gold test emails",
    )

    # Label training set
    label_file(
        input_path="data/train.jsonl",
        output_path="data/train_labeled.jsonl",
        desc="training emails",
    )

    # Print distribution stats
    print_label_distribution("data/train_labeled.jsonl")
    print_label_distribution("data/gold/gold_teacher_labeled.jsonl")
