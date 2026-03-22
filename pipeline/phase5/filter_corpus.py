"""
Phase 5: Corpus Filtering
Filters pages into retained vs excluded based on page-level quality criteria.
Read from Phase 4 output; no text transformation applied.
"""

import json
import pandas as pd
from pathlib import Path

INPUT_PATH  = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase 4/data/pages_phase4_modeltext.csv")
OUTPUT_DIR  = Path("/Users/denizguvenol/Desktop/thesis/cleaning/phase5/data")
RETAINED_PATH = OUTPUT_DIR / "pages_for_modeling.csv"
EXCLUDED_PATH = OUTPUT_DIR / "pages_excluded.csv"
SUMMARY_PATH  = OUTPUT_DIR / "phase5_summary.json"

MIN_TOKENS       = 15
NON_ASCII_THRESH = 0.05

def non_ascii_ratio(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading Phase 4 output …")
df = pd.read_csv(INPUT_PATH, low_memory=False)
total = len(df)
print(f"  {total:,} pages loaded")

# ---------------------------------------------------------------------------
# Build per-criterion boolean masks
# ---------------------------------------------------------------------------
mask_sparse      = df["token_count_model_lemma"] < MIN_TOKENS
mask_low_content = df["is_low_content_page"] == True
mask_cover       = df["is_likely_cover_page"] == True
mask_non_english = df["content"].apply(non_ascii_ratio) > NON_ASCII_THRESH

criteria = {
    "sparse":      mask_sparse,
    "low_content": mask_low_content,
    "cover":       mask_cover,
    "non_english": mask_non_english,
}

# ---------------------------------------------------------------------------
# Build exclusion_reason column (pipe-delimited, empty string if retained)
# ---------------------------------------------------------------------------
def build_reason(row):
    reasons = [label for label, mask in criteria.items() if mask.loc[row.name]]
    return "|".join(reasons)

print("Evaluating exclusion criteria …")
excluded_mask = mask_sparse | mask_low_content | mask_cover | mask_non_english

df["exclusion_reason"] = ""
df.loc[excluded_mask, "exclusion_reason"] = (
    df[excluded_mask].apply(build_reason, axis=1)
)

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
retained = df[~excluded_mask].drop(columns=["exclusion_reason"]).copy()
excluded = df[excluded_mask].copy()

# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------
n_retained = len(retained)
n_excluded = len(excluded)

per_criterion = {label: int(mask.sum()) for label, mask in criteria.items()}

multi_flag = int((sum(criteria.values()) > 1).sum())

# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Writing retained pages → {RETAINED_PATH.name} …")
retained.to_csv(RETAINED_PATH, index=False)

print(f"Writing excluded pages → {EXCLUDED_PATH.name} …")
excluded.to_csv(EXCLUDED_PATH, index=False)

# ---------------------------------------------------------------------------
# Save summary JSON
# ---------------------------------------------------------------------------
summary = {
    "total_input_pages": total,
    "total_retained":    n_retained,
    "total_excluded":    n_excluded,
    "exclusion_count_per_criterion": per_criterion,
    "multi_flag_pages":  multi_flag,
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Summary JSON → {SUMMARY_PATH.name}")

# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------
print()
print("=" * 52)
print("PHASE 5 SUMMARY")
print("=" * 52)
print(f"  Input pages   : {total:>8,}")
print(f"  Retained      : {n_retained:>8,}  ({n_retained/total*100:.1f}%)")
print(f"  Excluded      : {n_excluded:>8,}  ({n_excluded/total*100:.1f}%)")
print()
print("  Exclusion counts per criterion:")
for label, count in per_criterion.items():
    print(f"    {label:<14}: {count:>6,}  ({count/total*100:.1f}%)")
print()
print(f"  Pages flagged by >1 criterion: {multi_flag:,}")
print("=" * 52)
