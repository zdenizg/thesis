"""
Phase 3 Validation Script
=========================
Read-only validation of pages_phase3_linefiltered.csv.
Checks that line-level metadata filtering was conservative and correct.
"""

from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
INPUT_PATH = BASE_DIR / "data" / "pages_phase3_linefiltered.csv"

SEPARATOR = "=" * 70
REQUIRED_COLUMNS = {
    "file_id",
    "page_number",
    "content_clean_ocr",
    "content_clean_lines",
    "lines_before",
    "lines_after",
    "lines_removed",
    "line_removal_ratio",
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


def preview_text(text: object, limit: int = 500) -> str:
    """Return a single-line preview suitable for terminal output."""
    if not isinstance(text, str):
        return "<NaN>"
    return text[:limit].replace("\n", " <-> ")


def top_tokens(series: pd.Series, n: int = 30, *, desc: str) -> list[tuple[str, int]]:
    """Count word-like tokens across all texts in a Series."""
    counter: Counter[str] = Counter()
    for text in tqdm(series.dropna(), desc=desc, unit="row"):
        counter.update(re.findall(r"[a-z]{2,}", str(text)))
    return counter.most_common(n)


def count_token_in_series(series: pd.Series, token: str) -> int:
    """Count a whole-word token across the full Series."""
    pattern = rf"\b{re.escape(token)}\b"
    return int(series.fillna("").astype(str).str.count(pattern).sum())


def main() -> None:
    """Run the full Phase 3 validation report."""
    print(SEPARATOR)
    print("PHASE 3 VALIDATION")
    print(SEPARATOR)

    print("\n[1] Loading dataset...")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found:\n  {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    validate_columns(df, REQUIRED_COLUMNS, INPUT_PATH.name)

    # -----------------------------------------------------------------------
    # 2. Basic dataset information
    # -----------------------------------------------------------------------
    print("\n[2] BASIC DATASET INFORMATION")
    print(f"  Rows              : {len(df):,}")
    print(f"  Unique file_ids   : {df['file_id'].nunique():,}")
    print(f"  Columns ({len(df.columns)})    : {df.columns.tolist()}")
    print("\n  Missing values per column:")
    missing = df.isnull().sum()
    missing_rows = missing[missing > 0]
    if missing_rows.empty:
        print("    None")
    else:
        for col, count in missing_rows.items():
            pct = safe_percent(int(count), len(df), 1)
            print(f"    {col:<35} {int(count):>8,}  ({pct})")

    # -----------------------------------------------------------------------
    # 3. Descriptive statistics for line-removal metrics
    # -----------------------------------------------------------------------
    print("\n[3] LINE-REMOVAL DESCRIPTIVE STATISTICS")
    metrics = ["lines_before", "lines_after", "lines_removed", "line_removal_ratio"]
    print(df[metrics].describe().round(3).to_string())

    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.01]
    labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-75%", "75-100%"]
    has_text = df[df["lines_before"] > 0].copy()
    print("\n  Removal ratio distribution (pages with text only):")
    if has_text.empty:
        print("    No pages with text were available.")
    else:
        removal_bins = pd.cut(
            has_text["line_removal_ratio"],
            bins=bins,
            labels=labels,
            right=False,
        )
        for label, count in removal_bins.value_counts().sort_index().items():
            pct = safe_percent(int(count), len(has_text), 1)
            print(f"    {label:<12} {int(count):>7,}  ({pct})")

    # -----------------------------------------------------------------------
    # 4. High-removal pages (ratio > 0.4)  - 10 random examples
    # -----------------------------------------------------------------------
    print("\n[4] HIGH-REMOVAL PAGES  (line_removal_ratio > 0.40)  - 10 random examples")
    high = df[df["line_removal_ratio"] > 0.40].copy()
    print(f"  Total high-removal pages: {len(high):,}")

    sample_high = high.sample(min(10, len(high)), random_state=42)
    if sample_high.empty:
        print("  No high-removal pages met the threshold.")
    else:
        for _, row in sample_high.iterrows():
            before = preview_text(row["content_clean_ocr"], 500)
            after = preview_text(row["content_clean_lines"], 500)
            print(
                f"\n  file_id={row['file_id']}  page={row['page_number']}  "
                f"lines_before={row['lines_before']}  lines_after={row['lines_after']}  "
                f"ratio={row['line_removal_ratio']:.2f}"
            )
            print(f"  BEFORE: {before}")
            print(f"  AFTER : {after}")

    # -----------------------------------------------------------------------
    # 5. Low-removal pages (ratio < 0.1)  - 10 random examples
    # -----------------------------------------------------------------------
    print("\n[5] LOW-REMOVAL PAGES  (line_removal_ratio < 0.10)  - 10 random examples")
    low = df[(df["line_removal_ratio"] < 0.10) & (df["lines_before"] > 5)].copy()
    print(f"  Total low-removal pages: {len(low):,}")

    sample_low = low.sample(min(10, len(low)), random_state=99)
    if sample_low.empty:
        print("  No low-removal pages met the threshold.")
    else:
        for _, row in sample_low.iterrows():
            before = preview_text(row["content_clean_ocr"], 500)
            after = preview_text(row["content_clean_lines"], 500)
            print(
                f"\n  file_id={row['file_id']}  page={row['page_number']}  "
                f"lines_before={row['lines_before']}  lines_after={row['lines_after']}  "
                f"ratio={row['line_removal_ratio']:.2f}"
            )
            print(f"  BEFORE: {before}")
            print(f"  AFTER : {after}")

    # -----------------------------------------------------------------------
    # 6. Vocabulary comparison: top-30 tokens before vs after
    # -----------------------------------------------------------------------
    print("\n[6] TOP-30 TOKEN COMPARISON  (content_clean_ocr  vs  content_clean_lines)")
    print("  Counting top tokens in content_clean_ocr...")
    tok_before = top_tokens(df["content_clean_ocr"], desc="Tokens before")
    print("  Counting top tokens in content_clean_lines...")
    tok_after = top_tokens(df["content_clean_lines"], desc="Tokens after")

    after_dict = dict(tok_after)

    routing_tokens = {
        "division",
        "station",
        "distribution",
        "dissem",
        "rybat",
        "nodis",
        "limdis",
        "exdis",
        "priority",
        "immediate",
        "secret",
        "confidential",
        "unclassified",
        "film",
    }

    print(f"\n  {'TOKEN':<20} {'BEFORE':>12} {'AFTER':>12} {'CHANGE':>10}")
    print("  " + "-" * 56)

    for token, count_before in tok_before:
        count_after = after_dict.get(token, 0)
        change = count_after - count_before
        flag = "  <- ROUTING" if token in routing_tokens else ""
        print(
            f"  {token:<20} {count_before:>12,} {count_after:>12,} "
            f"{change:>+10,}{flag}"
        )

    print("\n  Routing token counts (selected):")
    print(f"  {'TOKEN':<20} {'BEFORE':>12} {'AFTER':>12} {'% CHANGE':>10}")
    print("  " + "-" * 56)

    routing_counts: list[tuple[str, int, int, float]] = []
    for token in sorted(routing_tokens):
        count_before = count_token_in_series(df["content_clean_ocr"], token)
        count_after = count_token_in_series(df["content_clean_lines"], token)
        pct_change = (100 * (count_after - count_before) / count_before) if count_before > 0 else 0
        routing_counts.append((token, count_before, count_after, pct_change))
        print(
            f"  {token:<20} {count_before:>12,} {count_after:>12,} "
            f"{pct_change:>+9.1f}%"
        )

    # -----------------------------------------------------------------------
    # 7. Character retention ratio
    # -----------------------------------------------------------------------
    print("\n[7] CHARACTER RETENTION RATIO  (char_ratio = len(after) / len(before))")

    has_both = df[df["content_clean_ocr"].notna() & df["content_clean_lines"].notna()].copy()
    before_len = has_both["content_clean_ocr"].astype(str).str.len()
    has_both = has_both[before_len > 0].copy()
    if has_both.empty:
        has_both["char_ratio"] = pd.Series(dtype=float)
        print("No pages with text were available for character-retention analysis.")
    else:
        before_len = has_both["content_clean_ocr"].astype(str).str.len().replace(0, np.nan)
        after_len = has_both["content_clean_lines"].astype(str).str.len()
        has_both["char_ratio"] = after_len / before_len
        print(has_both["char_ratio"].describe().round(4).to_string())

    low_retention = has_both[has_both["char_ratio"] < 0.5]
    pct_low_retention = safe_percent(len(low_retention), len(has_both), 1)
    print(
        f"\n  Pages with char_ratio < 0.50: {len(low_retention):,}  "
        f"({pct_low_retention} of pages with text)"
    )

    print("\n  Sample low-retention pages:")
    sample_low_retention = low_retention.sample(min(5, len(low_retention)), random_state=7)
    if sample_low_retention.empty:
        print("  None")
    else:
        for _, row in sample_low_retention.iterrows():
            before = preview_text(row["content_clean_ocr"], 400)
            after = preview_text(row["content_clean_lines"], 400)
            print(
                f"\n  file_id={row['file_id']}  page={row['page_number']}  "
                f"char_ratio={row['char_ratio']:.3f}  "
                f"line_removal_ratio={row['line_removal_ratio']:.2f}"
            )
            print(f"  BEFORE: {before}")
            print(f"  AFTER : {after}")

    # -----------------------------------------------------------------------
    # 8. 10 completely random page comparisons
    # -----------------------------------------------------------------------
    print("\n[8] 10 RANDOM PAGE COMPARISONS (visual narrative check)")

    random_pool = df[df["content_clean_ocr"].notna()]
    sample_size = min(10, len(random_pool))
    if sample_size == 0:
        print("  No non-null pages were available for random comparison.")
    else:
        rand_pages = random_pool.sample(sample_size, random_state=55)
        for index, (_, row) in enumerate(rand_pages.iterrows(), 1):
            before = preview_text(row["content_clean_ocr"], 600)
            after = preview_text(row["content_clean_lines"], 600)
            print(
                f"\n  [{index}] file_id={row['file_id']}  page={row['page_number']}  "
                f"lines_before={row['lines_before']}  lines_after={row['lines_after']}  "
                f"ratio={row['line_removal_ratio']:.2f}"
            )
            print(f"  BEFORE: {before}")
            print(f"  AFTER : {after}")

    # -----------------------------------------------------------------------
    # 9. Diagnostic summary
    # -----------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("[9] DIAGNOSTIC SUMMARY")
    print(SEPARATOR)

    if has_both.empty:
        avg_ratio = np.nan
        med_ratio = np.nan
        pct_zero = np.nan
        pct_gt30 = np.nan
        pct_gt50 = np.nan
        avg_char = np.nan
    else:
        avg_ratio = has_both["line_removal_ratio"].mean()
        med_ratio = has_both["line_removal_ratio"].median()
        pct_zero = 100 * (has_both["line_removal_ratio"] == 0).sum() / len(has_both)
        pct_gt30 = 100 * (has_both["line_removal_ratio"] > 0.30).sum() / len(has_both)
        pct_gt50 = 100 * (has_both["line_removal_ratio"] > 0.50).sum() / len(has_both)
        avg_char = has_both["char_ratio"].mean()

    def fmt_metric(value: float, *, suffix: str = "") -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:.3f}{suffix}"

    def fmt_percent_metric(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:.1f}%"

    print(f"  Average line_removal_ratio   : {fmt_metric(avg_ratio)}  ({fmt_percent_metric(avg_ratio * 100 if pd.notna(avg_ratio) else np.nan)})")
    print(f"  Median  line_removal_ratio   : {fmt_metric(med_ratio)}  ({fmt_percent_metric(med_ratio * 100 if pd.notna(med_ratio) else np.nan)})")
    print(f"  Pages with 0 % removed       : {fmt_percent_metric(pct_zero)}")
    print(f"  Pages with >30% removed      : {fmt_percent_metric(pct_gt30)}")
    print(f"  Pages with >50% removed      : {fmt_percent_metric(pct_gt50)}")
    print(f"  Average character retention  : {fmt_metric(avg_char)}  ({fmt_percent_metric(avg_char * 100 if pd.notna(avg_char) else np.nan)})")

    print()
    if pd.isna(avg_ratio) or pd.isna(pct_gt50):
        verdict = "N/A  -  No pages with text were available."
    elif avg_ratio < 0.15 and pct_gt50 < 5:
        verdict = "CONSERVATIVE ✓  Average removal is low; most pages retain >85% of lines."
    elif avg_ratio < 0.25 and pct_gt50 < 10:
        verdict = "MODERATE  -  Removal is moderate. Spot-check high-removal pages."
    else:
        verdict = "AGGRESSIVE ✗  High average removal. Review filtering rules."

    print(f"  Verdict: {verdict}")

    print()
    routing_decreases = [
        (token, count_before, count_after, (count_after - count_before) / count_before)
        for token, count_before, count_after, _ in routing_counts
        if count_before > 0
    ]

    if not routing_decreases:
        print("  Routing token reduction: no monitored routing tokens were present.")
    elif all(change <= 0 for _, _, _, change in routing_decreases):
        print("  Routing token reduction: ALL monitored tokens decreased or stayed unchanged ✓")
    else:
        increased = [(token, change) for token, _, _, change in routing_decreases if change > 0]
        print(
            "  Routing tokens that INCREASED after filtering: "
            f"{[(token, f'{change * 100:+.1f}%') for token, change in increased]}"
        )

    print(f"  Phase 3 status: script ran on {len(df):,} pages; dataset loaded from")
    print(f"  {INPUT_PATH}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
