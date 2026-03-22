"""
Phase 5 Validation Script
Checks retained pages, exclusion overlap, and column integrity.
Read-only — no modifications to any dataset.
"""

import pandas as pd
from pathlib import Path

PHASE4_INPUT  = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 4/data/pages_phase4_modeltext.csv")
RETAINED_PATH = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase5/data/pages_for_modeling.csv")
EXCLUDED_PATH = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase5/data/pages_excluded.csv")

retained = pd.read_csv(RETAINED_PATH, low_memory=False)
excluded = pd.read_csv(EXCLUDED_PATH, low_memory=False)

# ---------------------------------------------------------------------------
# Check 1 — Spot-check retained pages
# ---------------------------------------------------------------------------
print("=" * 60)
print("CHECK 1 — SPOT-CHECK RETAINED PAGES (n=20, random_state=42)")
print("=" * 60)

sample = retained.sample(20, random_state=42)
for i, (_, row) in enumerate(sample.iterrows(), 1):
    print(f"\n--- {i} ---")
    print(f"filename    : {row['filename']}")
    print(f"page_number : {row['page_number']}")
    print(f"content     :")
    print(str(row.get("content", ""))[:300])

# ---------------------------------------------------------------------------
# Check 2 — Overlap between exclusion criteria
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("CHECK 2 — EXCLUSION CRITERIA OVERLAP")
print("=" * 60)

reasons = excluded["exclusion_reason"].fillna("")

single_flag = (reasons.str.count(r"\|") == 0) & (reasons != "")
multi_flag  = reasons.str.count(r"\|") >= 1

print(f"\n  Flagged by exactly one criterion : {single_flag.sum():,}")
print(f"  Flagged by more than one         : {multi_flag.sum():,}")

print("\n  Per-criterion counts:")
for label in ["sparse", "low_content", "cover"]:
    count = reasons.str.contains(label).sum()
    print(f"    {label:<14}: {count:,}")

# ---------------------------------------------------------------------------
# Check 3 — Column integrity
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("CHECK 3 — COLUMN INTEGRITY")
print("=" * 60)

phase4_cols   = pd.read_csv(PHASE4_INPUT, low_memory=False, nrows=0).columns.tolist()
retained_cols = retained.columns.tolist()

missing = [c for c in phase4_cols if c not in retained_cols]
extra   = [c for c in retained_cols if c not in phase4_cols]

if not missing and not extra:
    print("\n  PASS — retained CSV has exactly the same columns as Phase 4 input.")
else:
    if missing:
        print(f"\n  FAIL — missing columns: {missing}")
    if extra:
        print(f"  NOTE — extra columns not in Phase 4: {extra}")
