"""Phase 3 Baseline Validation"""

import pandas as pd
from pathlib import Path

INPUT_PATH  = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 2/data/pages_phase2_cleaned.csv")
OUTPUT_PATH = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 3 /data/pages_phase3_linefiltered.csv")

REQUIRED_COLS = [
    "id", "filename", "page_number",
    "content_clean_lines", "lines_before", "lines_after",
    "lines_removed", "line_removal_ratio",
]
NOTNULL_COLS = ["id", "filename", "page_number"]

print("=" * 60)
print("PHASE 3 BASELINE VALIDATION")
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

# Check 4 — lines_after <= lines_before
if "lines_before" in out.columns and "lines_after" in out.columns:
    violations = int((out["lines_after"] > out["lines_before"]).sum())
    if violations == 0:
        print(f"PASS  lines_after <= lines_before everywhere")
        results.append(True)
    else:
        print(f"FAIL  {violations:,} rows where lines_after > lines_before")
        results.append(False)
else:
    print("SKIP  lines_after/lines_before columns not found")
    results.append(True)

# Check 5 — line_removal_ratio between 0 and 1
if "line_removal_ratio" in out.columns:
    below = int((out["line_removal_ratio"] < 0).sum())
    above = int((out["line_removal_ratio"] > 1).sum())
    if below == 0 and above == 0:
        print(f"PASS  line_removal_ratio is between 0 and 1 everywhere")
        results.append(True)
    else:
        print(f"FAIL  line_removal_ratio out of range: <0={below}, >1={above}")
        results.append(False)
else:
    print("SKIP  line_removal_ratio column not found")
    results.append(True)

# Info — line_removal_ratio distribution
if "line_removal_ratio" in out.columns:
    print()
    print("  line_removal_ratio distribution:")
    print(f"    mean   : {out['line_removal_ratio'].mean():.3f}")
    print(f"    median : {out['line_removal_ratio'].median():.3f}")

# Summary
passed = sum(results)
total  = len(results)
print()
print("=" * 60)
print(f"SUMMARY: {passed}/{total} checks passed {'✓' if passed == total else '✗'}")
print("=" * 60)
