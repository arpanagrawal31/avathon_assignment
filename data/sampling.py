import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from sampling_utils import clean_one, trigrams, near_dup

SEED = 42
N_TRAIN = 1000
N_TEST = 50
N_TOTAL = N_TRAIN + N_TEST
SAMPLE_BUF = 2 # Take double the number of samples. Stop early if reached the required number of samples
KAGGLE_CSV = "/kaggle/input/datasets/wcukierski/enron-email-dataset/emails.csv" # Or give path to csv
OUT_DIR    = Path("data");  GOLD_DIR = OUT_DIR / "gold"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GOLD_DIR.mkdir(parents=True, exist_ok=True)

# Read dataset
df = pd.read_csv(KAGGLE_CSV)
print(f"  {len(df):,} raw emails loaded")

pool = []
seen_fps = set()
stats = Counter()

for idx in range(len(df)):
    stats["total"] += 1
    if stats["total"] % 50_000 == 0:
        print(f"  processed {stats['total']:,}, pool {len(pool):,}")

    raw = df.iloc[idx]["message"]
    if not isinstance(raw, str):
        stats["filtered"] += 1
        continue

    c = clean_one(raw)

    if c is None:
        stats["filtered"] += 1
        continue
    if c["fingerprint"] in seen_fps:
        stats["exact_dup"] += 1
        continue
    seen_fps.add(c["fingerprint"])
    pool.append((idx, c))
    stats["clean"] += 1

    # print(c)
    # break

del df  # free ~1.4 GB

print(f"CLEAN POOL STATS")
for k, v in sorted(stats.items()):
    print(f"  {k:>12s}: {v:>8,}")
print(f"  {'pool_size':>12s}: {len(pool):>8,}")
print()


n_draw = min(int(N_TOTAL * SAMPLE_BUF), len(pool))
print(f"Drawing {n_draw} candidates from pool of {len(pool):,} (seed={SEED})")

random.seed(SEED)
candidates = random.sample(pool, n_draw)

# using trigrams and jaccard similarity to identify near duplicates
final, seen_tg = [], []
final_count, n_near = 0, 0
for idx, c in candidates:
    tg = trigrams(c["body"])
    if near_dup(tg, seen_tg):
        n_near += 1
        continue
    seen_tg.append(tg)
    final.append((idx, c))
    final_count += 1
    if final_count >= N_TOTAL:
        break

print(f"  Near-dups removed: {n_near}")

# Reproducible shuffle for train/test split
random.seed(SEED + 1)
random.shuffle(final)

test_set  = final[:N_TEST]
train_set = final[N_TEST : N_TEST + N_TRAIN]
print(f"  Train: {len(train_set)},  Gold test: {len(test_set)}")

# Write the config for reproducibility
manifest = {
    "seed": SEED,
    "dataset": "kaggle:wcukierski/enron-email-dataset",
    "n_train": len(train_set),
    "n_test":  len(test_set),
    "train_ids": [i for i, _ in train_set],
    "test_ids":  [i for i, _ in test_set],
}

with open(OUT_DIR / "sample_ids.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nFrozen IDs -> {OUT_DIR / 'sample_ids.json'}")

def _write_jsonl(path, items):
    with open(path, "w") as f:
        for idx, c in items:
            f.write(json.dumps({
                "orig_idx": idx,
                "subject":  c["subject"],
                "body":     c["body"],
                "from":     c["from"],
                "to":       c["to"],
                "date":     c["date"],
            }) + "\n")

_write_jsonl(OUT_DIR / "train.jsonl", train_set)
print(f"Train set -> {OUT_DIR / 'train.jsonl'}")

_write_jsonl(GOLD_DIR / "gold.jsonl", test_set)
print(f"Gold test set -> {GOLD_DIR / 'gold.jsonl'}")

with open(OUT_DIR / "clean_pool_stats.json", "w") as f:
    json.dump(dict(stats), f, indent=2)
print(f"Pool stats -> {OUT_DIR / 'clean_pool_stats.json'}")


# Check token lengths of the samples and how many have been truncated.
wcs = [len(c["body"].split()) for _, c in train_set + test_set]
toks = [int(w * 1.3) for w in wcs]
trunc = sum(1 for t in toks if t > 768)
pct = 100 * trunc / len(toks)

print(f"TOKEN-LENGTH CHECK (max_seq=768)")
print(f"  Words:     min={min(wcs):>4d}  med={sorted(wcs)[len(wcs)//2]:>4d}  max={max(wcs):>4d}")
print(f"  Est.tokens:min={min(toks):>4d}  med={sorted(toks)[len(toks)//2]:>4d}  max={max(toks):>4d}")
print(f"  Truncated: {trunc}/{len(toks)} ({pct:.1f}%)")
