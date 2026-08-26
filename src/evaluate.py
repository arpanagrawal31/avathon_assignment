"""
Evaluation harness for email triage extraction - Simple counts + exact match

Usage:
    python src/evaluate.py \
        --predictions results/baseline_zeroshot_preds.jsonl \
        --gold data/gold/gold_teacher_labeled.jsonl \
        --output results/baseline_zeroshot.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.schema import validate_json_string, check_grounding


def evaluate(predictions: list[dict], gold_labels: list[dict]) -> dict:
    """Evaluate predictions against gold labels using counts and exact match"""

    gold_by_idx = {g["orig_idx"]: g for g in gold_labels}
    per_example = []

    n = 0
    schema_valid   = 0
    intent_match   = 0
    urgency_match  = 0
    response_match = 0
    escalation_match = 0

    # Abstention: gold has empty action_items
    abst_total = 0          # how many gold examples have []
    abst_correct = 0        # model also predicted []

    # Grounding
    span_total = 0
    span_unsupported = 0

    # Action item count
    ai_count_match = 0      # predicted count == gold count

    for pred_rec in predictions:
        idx = pred_rec["orig_idx"]
        gold_rec = gold_by_idx.get(idx)
        if gold_rec is None or gold_rec.get("label") is None:
            continue

        n += 1
        raw   = pred_rec["raw_output"]
        body  = pred_rec.get("body", gold_rec.get("body", ""))
        g     = gold_rec["label"]

        parsed, err = validate_json_string(raw)
        valid = parsed is not None

        ex = {"orig_idx": idx, "schema_valid": valid, "error": err}

        if valid:
            schema_valid += 1
            p = parsed.model_dump()

            ex["intent_match"]    = p["intent"] == g["intent"]
            ex["urgency_match"]   = p["urgency"] == g["urgency"]
            ex["response_match"]  = p["requires_response"] == g["requires_response"]
            ex["escalation_match"] = p["escalation_flag"] == g["escalation_flag"]

            intent_match    += int(ex["intent_match"])
            urgency_match   += int(ex["urgency_match"])
            response_match  += int(ex["response_match"])
            escalation_match += int(ex["escalation_match"])

            # Action item count
            ex["ai_count_pred"] = len(p["action_items"])
            ex["ai_count_gold"] = len(g["action_items"])
            if len(p["action_items"]) == len(g["action_items"]):
                ai_count_match += 1

            # Abstention
            gold_empty = len(g["action_items"]) == 0
            pred_empty = len(p["action_items"]) == 0
            if gold_empty:
                abst_total += 1
                if pred_empty:
                    abst_correct += 1

            # Grounding
            gr = check_grounding(parsed, body)
            span_total       += gr["total_spans"]
            span_unsupported += gr["unsupported"]
            ex["grounding"] = {
                "total": gr["total_spans"],
                "unsupported": gr["unsupported"],
            }

        per_example.append(ex)

    def rate(num, den):
        return round(num / den, 4) if den else None

    metrics = {
        "total":               n,
        "schema_valid":        schema_valid,
        "schema_validity_rate": rate(schema_valid, n),
        # All below are out of schema_valid, not n
        "intent_accuracy":      rate(intent_match, schema_valid),
        "urgency_accuracy":     rate(urgency_match, schema_valid),
        "requires_response_accuracy": rate(response_match, schema_valid),
        "escalation_flag_accuracy":   rate(escalation_match, schema_valid),
        "action_item_count_accuracy": rate(ai_count_match, schema_valid),
        "abstention_recall":    rate(abst_correct, abst_total),
        "abstention_total":     abst_total,
        "unsupported_span_rate": rate(span_unsupported, span_total),
        "unsupported_spans":    span_unsupported,
        "total_spans":          span_total,
    }

    return {"metrics": metrics, "per_example": per_example}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def print_summary(m: dict):
    v = m["schema_valid"]
    n = m["total"]
    print(f"\n{'='*50}")
    print(f"  ({n} examples)")
    print(f"{'='*50}")
    print(f"  Schema validity:        {m['schema_validity_rate']:.0%}  ({v}/{n})")
    if v == 0:
        print("  (no valid outputs — skipping field metrics)")
        return
    print(f"  Intent accuracy:        {m['intent_accuracy']:.0%}")
    print(f"  Urgency accuracy:       {m['urgency_accuracy']:.0%}")
    print(f"  requires_response acc:  {m['requires_response_accuracy']:.0%}")
    print(f"  escalation_flag acc:    {m['escalation_flag_accuracy']:.0%}")
    print(f"  Action-item count acc:  {m['action_item_count_accuracy']:.0%}")
    if m["abstention_total"] > 0:
        print(f"  Abstention recall:      {m['abstention_recall']:.0%}"
              f"  ({m['abstention_total']} gold-empty examples)")
    if m["total_spans"] > 0:
        print(f"  Unsupported-span rate:  {m['unsupported_span_rate']:.0%}"
              f"  ({m['unsupported_spans']}/{m['total_spans']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gold",        required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--model",       default="unknown")
    ap.add_argument("--mode",        default="unknown")
    args = ap.parse_args()

    results = evaluate(load_jsonl(args.predictions), load_jsonl(args.gold))
    results["metadata"] = {"model": args.model}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results["metrics"])
    print(f"\n  Saved → {args.output}")


if __name__ == "__main__":
    main()
