"""
Phase 2: Dataset Inspection & Validation
=========================================
Loads the Phase 2 cleaned dataset and performs validation checks to confirm
that boilerplate removal and OCR normalisation behaved correctly.

Checks performed
----------------
  1. Dataset shape and column names
  2. Descriptive statistics for char_count_clean and word_count_clean
  3. Ratio of cleaned length vs original length (word-level)
  4. Pages where cleaning removed more than 50% of text
  5. Examples of those heavily-cleaned pages
  6. Top-30 most frequent tokens before and after cleaning
  7. 10 random original vs cleaned pages for manual inspection

No data is modified. The dataset is opened read-only.
"""

import re
import numpy as np
import pandas as pd
from collections import Counter

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PHASE2_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH  = os.path.join(PHASE2_ROOT, "data", "pages_phase2_cleaned.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SEP = "=" * 70

def word_tokens(series, lowercase=True):
    """Return a flat list of all alphabetic tokens (length >= 2) from a Series."""
    tokens = []
    for text in series.dropna():
        if lowercase:
            text = text.lower()
        tokens.extend(re.findall(r"[a-zA-Z]{2,}", text))
    return tokens

def preview(text, n=500):
    """Return a single-line preview of up to n characters."""
    if not isinstance(text, str):
        return "<NaN>"
    return text[:n].replace("\n", "↵ ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():

    # -- Load (read-only) -----------------------------------------------------
    print(f"\nLoading: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Dataset not found:\n  {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    # Compute retention ratio (used in checks 3-5)
    orig_wc = df["word_count"].replace(0, np.nan)
    df["_ratio_kept"] = (df["word_count_clean"] / orig_wc).clip(upper=1.0)

    # =========================================================================
    # 1. Shape and column names
    # =========================================================================
    print(f"\n{SEP}")
    print("1. DATASET SHAPE & COLUMNS")
    print(SEP)
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {df.shape[1]}")
    print()
    for i, col in enumerate(df.columns, 1):
        # mark the new Phase-2 columns for clarity
        tag = "  ← Phase 2" if col in {
            "content_clean_boilerplate",
            "content_clean_ocr", "word_count_clean", "char_count_clean"
        } else ""
        print(f"  {i:>2}. {col}{tag}")

    # =========================================================================
    # 2. Descriptive statistics
    # =========================================================================
    print(f"\n{SEP}")
    print("2. DESCRIPTIVE STATISTICS  (char_count & word_count — before vs after)")
    print(SEP)
    stats = df[[
        "char_count", "char_count_clean",
        "word_count", "word_count_clean"
    ]].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(stats.to_string())

    # =========================================================================
    # 3. Retention ratio
    # =========================================================================
    print(f"\n{SEP}")
    print("3. RETENTION RATIO  (word_count_clean / word_count)")
    print(SEP)

    ratio_stats = df["_ratio_kept"].describe(
        percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )
    print(ratio_stats.to_string())

    print()
    buckets = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.0]
    labels  = ["100% kept", "90–99%", "80–89%", "70–79%", "60–69%", "50–59%", "< 50%"]
    for lo, hi, lbl in zip(buckets[1:], buckets[:-1], labels):
        n   = ((df["_ratio_kept"] > lo) & (df["_ratio_kept"] <= hi)).sum()
        pct = 100 * n / len(df)
        bar = "█" * int(pct)
        print(f"  {lbl:>12}  {n:>7,}  ({pct:5.1f}%)  {bar}")

    # =========================================================================
    # 4. Pages where cleaning removed > 50 % of text
    # =========================================================================
    print(f"\n{SEP}")
    print("4. PAGES WHERE CLEANING REMOVED > 50% OF TEXT")
    print(SEP)

    heavy = df[df["_ratio_kept"] < 0.50].copy()
    print(f"  Count : {len(heavy):,}  ({100*len(heavy)/len(df):.2f}% of all pages)")

    print(f"\n  Overlap with Phase-1 heuristic flags:")
    flag_cols = [
        "is_low_content_page",
        "is_likely_distribution_page",
        "is_likely_cover_page",
    ]
    for col in flag_cols:
        n   = heavy[col].sum()
        pct = 100 * n / max(len(heavy), 1)
        print(f"    {col}: {n:,}  ({pct:.1f}% of heavily-cleaned pages)")

    # =========================================================================
    # 5. Examples of heavily-cleaned pages
    # =========================================================================
    print(f"\n{SEP}")
    print("5. EXAMPLES: PAGES WHERE > 50% TEXT REMOVED  (up to 5)")
    print(SEP)

    sample_heavy = heavy.sample(n=min(5, len(heavy)), random_state=7)
    for i, (_, row) in enumerate(sample_heavy.iterrows(), 1):
        kept_pct = 100 * row["_ratio_kept"] if not np.isnan(row["_ratio_kept"]) else 0
        print(f"\n  [{i}] file_id={row['file_id']}  page={row['page_number']}"
              f"  words_before={int(row['word_count'])}  words_after={row['word_count_clean']}"
              f"  kept={kept_pct:.1f}%")
        print(f"  ORIGINAL : {preview(row['content'], 400)}")
        print(f"  CLEANED  : {preview(row['content_clean_ocr'], 400)}")

    # =========================================================================
    # 6. Top-30 tokens before and after cleaning
    # =========================================================================
    print(f"\n{SEP}")
    print("6. TOP 30 TOKENS  (before cleaning vs after cleaning)")
    print(SEP)

    print("  Counting tokens in content …")
    before_counts = Counter(word_tokens(df["content"]))

    print("  Counting tokens in content_clean_ocr …")
    after_counts  = Counter(word_tokens(df["content_clean_ocr"]))

    print()
    print(f"  {'RANK':>4}  {'TOKEN (before)':>22} {'COUNT':>9}"
          f"    {'TOKEN (after)':>22} {'COUNT':>9}")
    print(f"  {'----':>4}  {'-------------':>22} {'-----':>9}"
          f"    {'------------':>22} {'-----':>9}")

    top_before = before_counts.most_common(30)
    top_after  = after_counts.most_common(30)
    for rank, ((wb, cb), (wa, ca)) in enumerate(zip(top_before, top_after), 1):
        print(f"  {rank:>4}  {wb:>22} {cb:>9,}    {wa:>22} {ca:>9,}")

    # Tokens present before but absent from top-30 after (removed by cleaning)
    before_top_set = {t for t, _ in top_before}
    after_top_set  = {t for t, _ in top_after}
    dropped = before_top_set - after_top_set
    gained  = after_top_set  - before_top_set
    if dropped:
        print(f"\n  Tokens that LEFT the top-30 after cleaning : {sorted(dropped)}")
    if gained:
        print(f"  Tokens that JOINED the top-30 after cleaning: {sorted(gained)}")

    # =========================================================================
    # 7. 10 random original vs cleaned pages
    # =========================================================================
    print(f"\n{SEP}")
    print("7. 10 RANDOM ORIGINAL vs CLEANED PAGES  (manual inspection)")
    print(SEP)

    rand_sample = df[df["content"].notna()].sample(n=10, random_state=99)
    for i, (_, row) in enumerate(rand_sample.iterrows(), 1):
        ratio_val = row["_ratio_kept"]
        kept_pct  = f"{100*ratio_val:.1f}%" if not np.isnan(ratio_val) else "n/a"
        print(f"\n  [{i:02d}] file_id={row['file_id']}  page={row['page_number']}"
              f"  words={int(row['word_count'])}→{row['word_count_clean']}  kept={kept_pct}")
        print(f"       ORIGINAL : {preview(row['content'])}")
        print(f"       CLEANED  : {preview(row['content_clean_ocr'])}")

    # =========================================================================
    # 8. Substantive pages with heavy cleaning (>20% removed, >100 words)
    # =========================================================================
    print(f"\n{SEP}")
    print("8. SUBSTANTIVE PAGES WITH HEAVY CLEANING  (>100 words, >20% removed)")
    print(SEP)

    orig_wc = df["word_count"].replace(0, np.nan)
    df["_frac_removed"] = (
        (df["word_count"] - df["word_count_clean"])
        .clip(lower=0)
        .div(orig_wc)
        .fillna(0)
    )

    substantive_heavy = df[
        (df["_frac_removed"] > 0.20) & (df["word_count"] > 100)
    ].sort_values("_frac_removed", ascending=False)

    print(f"  Count : {len(substantive_heavy):,}  ({100*len(substantive_heavy)/len(df):.1f}% of all pages)")
    print(f"\n  Top 10 most heavily cleaned substantive pages:")
    for i, (_, row) in enumerate(substantive_heavy.head(10).iterrows(), 1):
        print(f"\n  [{i}] file_id={row['file_id']}  page={row['page_number']}"
              f"  words: {int(row['word_count'])} → {int(row['word_count_clean'])}"
              f"  ({row['_frac_removed']*100:.1f}% removed)")
        print(f"  ORIGINAL : {preview(row['content'], 300)}")
        print(f"  CLEANED  : {preview(row['content_clean_ocr'], 300)}")

    # =========================================================================
    # Done
    # =========================================================================
    print(f"\n{SEP}")
    print("Inspection complete.  Dataset was NOT modified.")
    print(SEP)


if __name__ == "__main__":
    main()
