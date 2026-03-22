"""Phase 5 Baseline Validation"""

import json
import pandas as pd
from pathlib import Path

INPUT_PATH    = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 4/data/pages_phase4_modeltext.csv")
RETAINED_PATH = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase5/data/pages_for_modeling.csv")
EXCLUDED_PATH = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase5/data/pages_excluded.csv")
SUMMARY_PATH  = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase5/data/phase5_summary.json")

VALID_LABELS = {"sparse", "low_content", "cover", "non_english"}
NOTNULL_COLS = ["id", "filename", "page_number"]

print("=" * 60)
print("PHASE 5 BASELINE VALIDATION")
print("=" * 60)

inp      = pd.read_csv(INPUT_PATH,    low_memory=False)
retained = pd.read_csv(RETAINED_PATH, low_memory=False)
excluded = pd.read_csv(EXCLUDED_PATH, low_memory=False)

with open(SUMMARY_PATH) as f:
    summary = json.load(f)

results = []

# Check 1 — retained + excluded == input
total_out = len(retained) + len(excluded)
if total_out == len(inp):
    print(f"PASS  retained ({len(retained):,}) + excluded ({len(excluded):,}) = input ({len(inp):,})")
    results.append(True)
else:
    print(f"FAIL  retained + excluded = {total_out:,}, expected {len(inp):,}")
    results.append(False)

# Check 2 — No nulls in key columns in both files
all_null_ok = True
for label, df in [("retained", retained), ("excluded", excluded)]:
    for col in NOTNULL_COLS:
        if col not in df.columns:
            continue
        n = int(df[col].isnull().sum())
        if n > 0:
            print(f"FAIL  {n:,} nulls in {col} ({label})")
            all_null_ok = False
if all_null_ok:
    print(f"PASS  no nulls in {NOTNULL_COLS} in either output file")
results.append(all_null_ok)

# Check 3 — exclusion_reason has no nulls in excluded file
if "exclusion_reason" in excluded.columns:
    null_reason = int(excluded["exclusion_reason"].isnull().sum())
    if null_reason == 0:
        print(f"PASS  no nulls in exclusion_reason")
        results.append(True)
    else:
        print(f"FAIL  {null_reason:,} nulls in exclusion_reason")
        results.append(False)
else:
    print("FAIL  exclusion_reason column missing from excluded file")
    results.append(False)

# Check 4 — All labels in exclusion_reason are valid
if "exclusion_reason" in excluded.columns:
    all_labels_ok = True
    for reason in excluded["exclusion_reason"].dropna().unique():
        invalid = set(str(reason).split("|")) - VALID_LABELS
        if invalid:
            print(f"FAIL  invalid label(s) in exclusion_reason: {invalid}")
            all_labels_ok = False
            break
    if all_labels_ok:
        print(f"PASS  all exclusion_reason values use valid labels")
    results.append(all_labels_ok)
else:
    results.append(False)

# Check 5 — Summary JSON matches actual counts
json_ok = (
    summary.get("total_retained")    == len(retained) and
    summary.get("total_excluded")    == len(excluded) and
    summary.get("total_input_pages") == len(inp)
)
if json_ok:
    print(f"PASS  phase5_summary.json counts match actual file row counts")
    results.append(True)
else:
    print(f"FAIL  summary mismatch")
    print(f"       json:   retained={summary.get('total_retained')}, excluded={summary.get('total_excluded')}, total={summary.get('total_input_pages')}")
    print(f"       actual: retained={len(retained)}, excluded={len(excluded)}, total={len(inp)}")
    results.append(False)

# Check 6 — No id in both retained and excluded
if "id" in retained.columns and "id" in excluded.columns:
    overlap = set(retained["id"]).intersection(set(excluded["id"]))
    if not overlap:
        print(f"PASS  no id appears in both retained and excluded")
        results.append(True)
    else:
        print(f"FAIL  {len(overlap):,} id(s) appear in both files: {list(overlap)[:5]}")
        results.append(False)
else:
    print("SKIP  id column not available for overlap check")
    results.append(True)

# Info — retention rate and exclusion breakdown
total = len(inp)
print()
print(f"  Retention rate : {len(retained)/total*100:.1f}%  ({len(retained):,} of {total:,})")
print(f"  Exclusion breakdown by criterion:")
for label in sorted(VALID_LABELS):
    if "exclusion_reason" in excluded.columns:
        count = excluded["exclusion_reason"].str.contains(label).sum()
        print(f"    {label:<14}: {count:,}  ({count/total*100:.1f}%)")

# Summary
passed = sum(results)
total_checks = len(results)
print()
print("=" * 60)
print(f"SUMMARY: {passed}/{total_checks} checks passed {'✓' if passed == total_checks else '✗'}")
print("=" * 60)
