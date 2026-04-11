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
  8. Substantive pages with heavy cleaning (>20% removed, >100 words)

No data is modified. The dataset is opened read-only.
"""

from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE2_ROOT = SCRIPT_DIR.parent
INPUT_PATH = PHASE2_ROOT / "data" / "pages_phase2_cleaned.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SEP = "=" * 70
REQUIRED_COLUMNS = {
    "content",
    "content_clean_ocr",
    "char_count",
    "char_count_clean",
    "word_count",
    "word_count_clean",
    "file_id",
    "page_number",
    "is_low_content_page",
    "is_likely_distribution_page",
    "is_likely_cover_page",
}


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a clear error if an expected dataset column is missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def safe_percent(part: int, whole: int, decimals: int = 1) -> str:
    """Format a percentage while handling zero denominators."""
    if whole == 0:
        return "n/a"
    return f"{(100 * part / whole):.{decimals}f}%"


def word_tokens(series: pd.Series, lowercase: bool = True, *, desc: str) -> list[str]:
    """Return alphabetic tokens (length >= 2) from a Series."""
    tokens: list[str] = []
    for text in tqdm(series.dropna(), desc=desc, unit="row"):
        if not isinstance(text, str):
            text = str(text)
        if lowercase:
            text = text.lower()
        tokens.extend(re.findall(r"[a-zA-Z]{2,}", text))
    return tokens


def preview(text: object, n: int = 500) -> str:
    """Return a single-line preview of up to n characters."""
    if not isinstance(text, str):
        return "<NaN>"
    return text[:n].replace("\n", "↵ ")


def print_section(title: str) -> None:
    """Print a consistently formatted section header."""
    print(f"\n{SEP}")
    print(title)
    print(SEP)


def main() -> None:
    """Run the Phase 2 inspection report."""
    print(f"\nLoading: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found:\n  {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    validate_columns(df, REQUIRED_COLUMNS, INPUT_PATH.name)

    # Compute retention ratio (used in checks 3-5)
    orig_wc = df["word_count"].replace(0, np.nan)
    df["_ratio_kept"] = (df["word_count_clean"] / orig_wc).clip(upper=1.0)

    # =========================================================================
    # 1. Shape and column names
    # =========================================================================
    print_section("1. DATASET SHAPE & COLUMNS")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {df.shape[1]}")
    print()
    for index, col in enumerate(df.columns, 1):
        # Mark the new Phase-2 columns for clarity.
        tag = (
            "  <- Phase 2"
            if col in {
                "content_clean_boilerplate",
                "content_clean_ocr",
                "word_count_clean",
                "char_count_clean",
            }
            else ""
        )
        print(f"  {index:>2}. {col}{tag}")

    # =========================================================================
    # 2. Descriptive statistics
    # =========================================================================
    print_section("2. DESCRIPTIVE STATISTICS  (char_count & word_count - before vs after)")
    stats = df[
        ["char_count", "char_count_clean", "word_count", "word_count_clean"]
    ].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(stats.to_string())

    # =========================================================================
    # 3. Retention ratio
    # =========================================================================
    print_section("3. RETENTION RATIO  (word_count_clean / word_count)")

    ratio_stats = df["_ratio_kept"].describe(
        percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )
    print(ratio_stats.to_string())

    print()
    bucket_edges = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.0]
    bucket_labels = [
        ">90% to 100%",
        ">80% to 90%",
        ">70% to 80%",
        ">60% to 70%",
        ">50% to 60%",
        ">0% to 50%",
    ]
    total_rows = len(df)
    for lower, upper, label in zip(bucket_edges[1:], bucket_edges[:-1], bucket_labels):
        count = int(((df["_ratio_kept"] > lower) & (df["_ratio_kept"] <= upper)).sum())
        pct_value = 0.0 if total_rows == 0 else (100 * count / total_rows)
        bar = "#" * int(pct_value)
        print(f"  {label:>12}  {count:>7,}  ({pct_value:5.1f}%)  {bar}")

    # =========================================================================
    # 4. Pages where cleaning removed > 50 % of text
    # =========================================================================
    print_section("4. PAGES WHERE CLEANING REMOVED > 50% OF TEXT")

    heavy = df[df["_ratio_kept"] < 0.50].copy()
    print(f"  Count : {len(heavy):,}  ({safe_percent(len(heavy), len(df), 2)} of all pages)")

    print("\n  Overlap with Phase-1 heuristic flags:")
    flag_cols = [
        "is_low_content_page",
        "is_likely_distribution_page",
        "is_likely_cover_page",
    ]
    for col in flag_cols:
        count = int(heavy[col].sum())
        pct = safe_percent(count, len(heavy), 1)
        print(f"    {col}: {count:,}  ({pct} of heavily-cleaned pages)")

    # =========================================================================
    # 5. Examples of heavily-cleaned pages
    # =========================================================================
    print_section("5. EXAMPLES: PAGES WHERE > 50% TEXT REMOVED  (up to 5)")

    if heavy.empty:
        print("  No pages met the >50% removal threshold.")
    else:
        sample_heavy = heavy.sample(n=min(5, len(heavy)), random_state=7)
        for index, (_, row) in enumerate(sample_heavy.iterrows(), 1):
            kept_pct = 100 * row["_ratio_kept"] if not np.isnan(row["_ratio_kept"]) else 0
            print(
                f"\n  [{index}] file_id={row['file_id']}  page={row['page_number']}"
                f"  words_before={int(row['word_count'])}  words_after={row['word_count_clean']}"
                f"  kept={kept_pct:.1f}%"
            )
            print(f"  ORIGINAL : {preview(row['content'], 400)}")
            print(f"  CLEANED  : {preview(row['content_clean_ocr'], 400)}")

    # =========================================================================
    # 6. Top-30 tokens before and after cleaning
    # =========================================================================
    print_section("6. TOP 30 TOKENS  (before cleaning vs after cleaning)")

    print("  Counting tokens in content...")
    before_counts = Counter(word_tokens(df["content"], desc="Tokens before"))

    print("  Counting tokens in content_clean_ocr...")
    after_counts = Counter(word_tokens(df["content_clean_ocr"], desc="Tokens after"))

    print()
    print(
        f"  {'RANK':>4}  {'TOKEN (before)':>22} {'COUNT':>9}"
        f"    {'TOKEN (after)':>22} {'COUNT':>9}"
    )
    print(
        f"  {'----':>4}  {'-------------':>22} {'-----':>9}"
        f"    {'------------':>22} {'-----':>9}"
    )

    top_before = before_counts.most_common(30)
    top_after = after_counts.most_common(30)
    max_rows = max(len(top_before), len(top_after))
    for rank in range(max_rows):
        before_token, before_count = top_before[rank] if rank < len(top_before) else ("", 0)
        after_token, after_count = top_after[rank] if rank < len(top_after) else ("", 0)
        print(
            f"  {rank + 1:>4}  {before_token:>22} {before_count:>9,}"
            f"    {after_token:>22} {after_count:>9,}"
        )

    # Tokens present before but absent from top-30 after (removed by cleaning)
    before_top_set = {token for token, _ in top_before}
    after_top_set = {token for token, _ in top_after}
    dropped = before_top_set - after_top_set
    gained = after_top_set - before_top_set
    if dropped:
        print(f"\n  Tokens that LEFT the top-30 after cleaning : {sorted(dropped)}")
    if gained:
        print(f"  Tokens that JOINED the top-30 after cleaning: {sorted(gained)}")

    # =========================================================================
    # 7. 10 random original vs cleaned pages
    # =========================================================================
    print_section("7. 10 RANDOM ORIGINAL vs CLEANED PAGES  (manual inspection)")

    non_null_content = df[df["content"].notna()]
    sample_size = min(10, len(non_null_content))
    if sample_size == 0:
        print("  No non-null content rows available for random inspection.")
    else:
        rand_sample = non_null_content.sample(n=sample_size, random_state=99)
        for index, (_, row) in enumerate(rand_sample.iterrows(), 1):
            ratio_val = row["_ratio_kept"]
            kept_pct = f"{100 * ratio_val:.1f}%" if not np.isnan(ratio_val) else "n/a"
            print(
                f"\n  [{index:02d}] file_id={row['file_id']}  page={row['page_number']}"
                f"  words={int(row['word_count'])}->{row['word_count_clean']}  kept={kept_pct}"
            )
            print(f"       ORIGINAL : {preview(row['content'])}")
            print(f"       CLEANED  : {preview(row['content_clean_ocr'])}")

    # =========================================================================
    # 8. Substantive pages with heavy cleaning (>20% removed, >100 words)
    # =========================================================================
    print_section("8. SUBSTANTIVE PAGES WITH HEAVY CLEANING  (>100 words, >20% removed)")

    orig_wc = df["word_count"].replace(0, np.nan)
    df["_frac_removed"] = (
        (df["word_count"] - df["word_count_clean"]).clip(lower=0).div(orig_wc).fillna(0)
    )

    substantive_heavy = df[
        (df["_frac_removed"] > 0.20) & (df["word_count"] > 100)
    ].sort_values("_frac_removed", ascending=False)

    print(
        f"  Count : {len(substantive_heavy):,}  "
        f"({safe_percent(len(substantive_heavy), len(df), 1)} of all pages)"
    )
    print("\n  Top 10 most heavily cleaned substantive pages:")
    for index, (_, row) in enumerate(substantive_heavy.head(10).iterrows(), 1):
        print(
            f"\n  [{index}] file_id={row['file_id']}  page={row['page_number']}"
            f"  words: {int(row['word_count'])} -> {int(row['word_count_clean'])}"
            f"  ({row['_frac_removed'] * 100:.1f}% removed)"
        )
        print(f"  ORIGINAL : {preview(row['content'], 300)}")
        print(f"  CLEANED  : {preview(row['content_clean_ocr'], 300)}")

    # =========================================================================
    # Done
    # =========================================================================
    print(f"\n{SEP}")
    print("Inspection complete. Dataset was NOT modified.")
    print(SEP)


if __name__ == "__main__":
    main()
