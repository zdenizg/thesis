"""Phase 2 Baseline Validation"""

import pandas as pd
from pathlib import Path

INPUT_PATH  = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 1 /data/pages_phase1_structural.csv")
OUTPUT_PATH = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 2/data/pages_phase2_cleaned.csv")

REQUIRED_COLS = [
    "id", "filename", "page_number", "content",
    "content_clean_ocr", "word_count_clean", "char_count_clean",
]
NOTNULL_COLS = ["id", "filename", "page_number"]

print("=" * 60)
print("PHASE 2 BASELINE VALIDATION")
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

# Check 4 — Non-negative word_count_clean and char_count_clean
neg_word = int((out["word_count_clean"] < 0).sum()) if "word_count_clean" in out.columns else -1
neg_char = int((out["char_count_clean"] < 0).sum()) if "char_count_clean" in out.columns else -1
if neg_word == 0 and neg_char == 0:
    print(f"PASS  word_count_clean and char_count_clean are non-negative everywhere")
    results.append(True)
else:
    print(f"FAIL  negative counts: word_count_clean={neg_word}, char_count_clean={neg_char}")
    results.append(False)

# Check 5 — char_count_clean <= char_count from Phase 1
if "char_count" in inp.columns and "char_count_clean" in out.columns and "id" in inp.columns and "id" in out.columns:
    merged = out[["id", "char_count_clean"]].merge(
        inp[["id", "char_count"]], on="id", how="inner"
    )
    violations = int((merged["char_count_clean"] > merged["char_count"]).sum())
    if violations == 0:
        print(f"PASS  char_count_clean <= char_count (Phase 1) for all {len(merged):,} matched rows")
        results.append(True)
    else:
        print(f"FAIL  {violations:,} rows where char_count_clean > char_count (Phase 1)")
        results.append(False)
else:
    print("SKIP  char_count comparison — required columns not available")
    results.append(True)

# Info — word_count_clean distribution
if "word_count_clean" in out.columns:
    print()
    print("  word_count_clean distribution:")
    print(f"    mean   : {out['word_count_clean'].mean():.1f}")
    print(f"    median : {out['word_count_clean'].median():.1f}")

# Summary
passed = sum(results)
total  = len(results)
print()
print("=" * 60)
print(f"SUMMARY: {passed}/{total} checks passed {'✓' if passed == total else '✗'}")
print("=" * 60)
