"""
Phase 4: Token-Level Stopword Discovery
=======================================
Empirically identifies the most frequent tokens in the Phase 3 corpus
BEFORE archive stopword removal, to justify and validate ARCHIVE_STOPWORDS.

For each high-frequency token the script reports:
  - Raw frequency in the corpus
  - Which filter catches it:
      nltk          : removed by NLTK English stopwords
      archive       : removed by ARCHIVE_STOPWORDS
      redundant     : in both NLTK and ARCHIVE (duplicate - can clean up)
      UNCAUGHT      : passes all filters -> reaches the topic model

Run this before finalising ARCHIVE_STOPWORDS to ensure high-frequency
non-substantive tokens are covered, and to confirm substantive tokens
are not being over-removed.
"""

from collections import Counter
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE4_ROOT = SCRIPT_DIR.parent
PHASE3_DATA = PHASE4_ROOT.parent / "phase3" / "data" / "pages_phase3_linefiltered.csv"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TOP_N = 150       # how many top tokens to report
MIN_COUNT = 200   # ignore tokens appearing fewer than this many times
SEPARATOR = "=" * 72
REQUIRED_COLUMNS = {"content_clean_lines"}

# ---------------------------------------------------------------------------
# Import current filters from phase4_modeltext.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(SCRIPT_DIR))
from phase4_modeltext import (
    ARCHIVE_STOPWORDS,
    ENGLISH_STOPWORDS,
    is_valid_token,
    word_tokenize,
)


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a clear error if an expected dataset column is missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def which_filter(token: str) -> str:
    """Return which stopword filter catches this token, or UNCAUGHT."""
    in_nltk = token in ENGLISH_STOPWORDS
    in_archive = token in ARCHIVE_STOPWORDS
    if in_nltk and in_archive:
        return "redundant"
    if in_nltk:
        return "nltk"
    if in_archive:
        return "archive"
    return "UNCAUGHT"


def main() -> None:
    """Run the Phase 4 token-discovery report."""
    print(f"Loading: {PHASE3_DATA}")
    if not PHASE3_DATA.exists():
        sys.exit(f"ERROR: Input file not found:\n  {PHASE3_DATA}")

    df = pd.read_csv(PHASE3_DATA, low_memory=False)
    validate_columns(df, REQUIRED_COLUMNS, PHASE3_DATA.name)
    print(f"  Loaded {len(df):,} rows\n")

    # -----------------------------------------------------------------------
    # Count all valid tokens (no stopword removal)
    # -----------------------------------------------------------------------
    print("Counting tokens...")
    token_counter: Counter[str] = Counter()

    for text in tqdm(df["content_clean_lines"].dropna(), desc="Scanning pages", unit="page"):
        for token in word_tokenize(str(text).lower()):
            token_counter[token] += 1

    print(f"  Total unique raw tokens : {len(token_counter):,}")

    valid_counter = Counter(
        {token: count for token, count in token_counter.items() if is_valid_token(token)}
    )
    print(f"  After is_valid_token    : {len(valid_counter):,} unique tokens\n")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print(SEPARATOR)
    print(f"TOP {TOP_N} VALID TOKENS  (min {MIN_COUNT} occurrences)")
    print(f"{'COUNT':>9}  {'FILTER':>12}  TOKEN")
    print(f"{'-----':>9}  {'------':>12}  {'-----'}")
    print(SEPARATOR)

    filter_counts: Counter[str] = Counter()
    uncaught_tokens: list[tuple[str, int]] = []
    reviewed = 0

    for token, count in valid_counter.most_common(TOP_N):
        if count < MIN_COUNT:
            break
        reviewed += 1
        filter_name = which_filter(token)
        filter_counts[filter_name] += 1
        if filter_name == "UNCAUGHT":
            uncaught_tokens.append((token, count))
        print(f"  {count:>8,}  {filter_name:>12}  {token}")

    print(SEPARATOR)
    print(f"\nSummary (reviewed {reviewed} tokens, min {MIN_COUNT} occurrences):")
    for filter_name, count in sorted(filter_counts.items()):
        print(f"  {filter_name:<14}: {count}")

    print("\nUncaught tokens (review - not automatically noise):")
    if uncaught_tokens:
        for token, count in uncaught_tokens:
            print(f"  {count:>8,}  {token}")
    else:
        print("  None - all high-frequency tokens are caught by a filter.")

    if filter_counts.get("redundant", 0) > 0:
        print("\nRedundant entries in ARCHIVE_STOPWORDS (also in NLTK - safe to remove):")
        for token, count in valid_counter.most_common(TOP_N):
            if count < MIN_COUNT:
                break
            if which_filter(token) == "redundant":
                print(f"  {count:>8,}  {token}")

    print(
        "\nNote: 'UNCAUGHT' tokens are not automatically noise - "
        "review each one before adding to ARCHIVE_STOPWORDS."
    )


if __name__ == "__main__":
    main()
