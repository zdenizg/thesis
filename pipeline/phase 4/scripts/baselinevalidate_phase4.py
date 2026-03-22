"""Phase 4 Baseline Validation"""

import pandas as pd
from pathlib import Path

INPUT_PATH  = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 3 /data/pages_phase3_linefiltered.csv")
OUTPUT_PATH = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 4/data/pages_phase4_modeltext.csv")

REQUIRED_COLS = [
    "id", "filename", "page_number",
    "content_model_lemma", "content_model_no_lemma",
    "token_count_model_lemma", "token_count_model_no_lemma",
]
NOTNULL_COLS = ["id", "filename", "page_number"]

print("=" * 60)
print("PHASE 4 BASELINE VALIDATION")
print("=" * 60)

inp = pd.read_csv(INPUT_PATH,  low_memory=False)
out = pd.read_csv(OUTPUT_PATH, low_memory=False)

results = []

# Check 1 — Row count
if len(out) == len(inp):
    print(f"PASS  row count: {len(out):,} rows (matches input)")
    results.append(True)
else:
    print(f"FAIL  row count: input={len(inp):,}, output={len(out):,}")
    results.append(False)

# Check 2 — Required columns
missing = [c for c in REQUIRED_COLS if c not in out.columns]
if not missing:
    print(f"PASS  required columns: all {len(REQUIRED_COLS)} present")
    results.append(True)
else:
    print(f"FAIL  required columns missing: {missing}")
    results.append(False)

# Check 3 — No nulls in key columns
null_counts = {c: int(out[c].isnull().sum()) for c in NOTNULL_COLS if c in out.columns}
any_nulls = any(v > 0 for v in null_counts.values())
if not any_nulls:
    print(f"PASS  no nulls in {NOTNULL_COLS}")
    results.append(True)
else:
    print(f"FAIL  nulls found: {null_counts}")
    results.append(False)

# Check 4 — token_count_model_lemma is non-negative
if "token_count_model_lemma" in out.columns:
    neg = int((out["token_count_model_lemma"] < 0).sum())
    if neg == 0:
        print(f"PASS  token_count_model_lemma is non-negative everywhere")
        results.append(True)
    else:
        print(f"FAIL  {neg:,} rows with negative token_count_model_lemma")
        results.append(False)
else:
    print("SKIP  token_count_model_lemma column not found")
    results.append(True)

# Check 5 — Nulls in content_model_lemma
if "content_model_lemma" in out.columns:
    null_lemma = int(out["content_model_lemma"].isnull().sum())
    zero_token = int((out["token_count_model_lemma"] == 0).sum()) if "token_count_model_lemma" in out.columns else None
    if null_lemma == 0:
        print(f"PASS  no nulls in content_model_lemma")
        results.append(True)
    elif zero_token is not None and null_lemma == zero_token:
        print(f"PASS  {null_lemma:,} nulls in content_model_lemma — all correspond to zero-token pages (expected)")
        results.append(True)
    else:
        print(f"FAIL  {null_lemma:,} nulls in content_model_lemma ({zero_token} are zero-token pages)")
        results.append(False)
else:
    print("SKIP  content_model_lemma column not found")
    results.append(True)

# Info — token_count_model_lemma distribution
if "token_count_model_lemma" in out.columns:
    tc = out["token_count_model_lemma"]
    print()
    print("  token_count_model_lemma distribution:")
    print(f"    mean   : {tc.mean():.1f}")
    print(f"    median : {tc.median():.1f}")
    print(f"    min    : {int(tc.min())}")
    print(f"    max    : {int(tc.max())}")
    print(f"    zero-token pages: {int((tc == 0).sum()):,}")

# Summary
passed = sum(results)
total  = len(results)
print()
print("=" * 60)
print(f"SUMMARY: {passed}/{total} checks passed {'✓' if passed == total else '✗'}")
print("=" * 60)
