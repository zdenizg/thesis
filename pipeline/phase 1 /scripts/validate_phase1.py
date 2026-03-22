"""Phase 1 Baseline Validation"""

import pandas as pd
from pathlib import Path

INPUT_PATH  = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 1 /JFK_Pages_Merged.csv")
OUTPUT_PATH = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 1 /data/pages_phase1_structural.csv")

REQUIRED_COLS = [
    "id", "file_id", "filename", "page_number", "word_count", "char_count",
    "is_low_content_page", "is_likely_cover_page", "is_likely_distribution_page",
]
BOOL_COLS    = ["is_low_content_page", "is_likely_cover_page", "is_likely_distribution_page"]
NOTNULL_COLS = ["id", "filename", "page_number"]

print("=" * 60)
print("PHASE 1 BASELINE VALIDATION")
print("=" * 60)

inp = pd.read_csv(INPUT_PATH,  low_memory=False)
out = pd.read_csv(OUTPUT_PATH, low_memory=False)

results = []

# Check 1 — Output has no more rows than input
# (Phase 1 removes empty-content rows, so output <= input is expected)
if len(out) <= len(inp):
    print(f"PASS  row count: output={len(out):,} <= input={len(inp):,} (diff={len(inp)-len(out):,} empty rows removed)")
    results.append(True)
else:
    print(f"FAIL  row count: output={len(out):,} > input={len(inp):,}")
    results.append(False)

# Check 2 — Required columns present
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

# Check 4 — Non-negative word_count and char_count
neg_word = int((out["word_count"] < 0).sum()) if "word_count" in out.columns else -1
neg_char = int((out["char_count"] < 0).sum()) if "char_count" in out.columns else -1
if neg_word == 0 and neg_char == 0:
    print(f"PASS  word_count and char_count are non-negative everywhere")
    results.append(True)
else:
    print(f"FAIL  negative counts: word_count={neg_word}, char_count={neg_char}")
    results.append(False)

# Check 5 — Boolean columns contain only True/False
bool_ok = True
for col in BOOL_COLS:
    if col not in out.columns:
        continue
    unique_vals = set(out[col].dropna().unique())
    if not unique_vals.issubset({True, False}):
        print(f"FAIL  {col} contains non-boolean values: {unique_vals}")
        bool_ok = False
if bool_ok:
    print(f"PASS  boolean flag columns contain only True/False")
results.append(bool_ok)

# Info — value counts for boolean flags
print()
print("  Boolean flag value counts:")
for col in BOOL_COLS:
    if col in out.columns:
        vc = out[col].value_counts().to_dict()
        print(f"    {col}: {vc}")

# Summary
passed = sum(results)
total  = len(results)
print()
print("=" * 60)
print(f"SUMMARY: {passed}/{total} checks passed {'✓' if passed == total else '✗'}")
print("=" * 60)
