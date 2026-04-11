"""
Phase 3: Line-Level Metadata Discovery
======================================
Empirically identifies the most frequent short standalone lines in the
Phase 2 cleaned corpus. Used to justify and validate the metadata-removal
patterns in phase3_line_filtering.py.

For each frequent line, the script reports:
  - How many times it appears across all pages
  - Which Phase 3 rule already catches it (or NONE if uncaught)

Run this script before finalising phase3_line_filtering.py to ensure all
high-frequency metadata lines are covered.
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
PHASE3_ROOT = SCRIPT_DIR.parent
PHASE2_ROOT = PHASE3_ROOT.parent / "phase2"

INPUT_PATH = PHASE2_ROOT / "data" / "pages_phase2_cleaned.csv"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MAX_LINE_LEN = 60   # only consider lines shorter than this
TOP_N = 100         # how many top lines to report
MIN_COUNT = 50      # ignore lines appearing fewer than this many times
SEPARATOR = "=" * 70
REQUIRED_COLUMNS = {"content_clean_ocr"}

# ---------------------------------------------------------------------------
# Import current filtering rules from phase3_line_filtering.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(SCRIPT_DIR))
from phase3_line_filtering import (
    _line_is_code_like,
    _line_is_filing_action,
    _line_is_low_content,
    _line_is_metadata_phrase,
    _line_is_standalone_keyword,
)


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a clear error if an expected dataset column is missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def which_rule_catches(line: str) -> str:
    """Return the first Phase 3 rule that catches this line, or NONE."""
    line_lower = line.lower().strip()
    if _line_is_code_like(line_lower):
        return "code_like"
    if _line_is_metadata_phrase(line_lower):
        return "metadata_phrase"
    if _line_is_standalone_keyword(line_lower):
        return "standalone_keyword"
    if _line_is_filing_action(line_lower):
        return "filing_action"
    if _line_is_low_content(line_lower):
        return "low_content"
    return "NONE"


def main() -> None:
    """Run the Phase 3 line-discovery report."""
    print(f"Loading: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        sys.exit(f"ERROR: Input file not found:\n  {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    validate_columns(df, REQUIRED_COLUMNS, INPUT_PATH.name)
    print(f"  Loaded {len(df):,} rows\n")

    # -----------------------------------------------------------------------
    # Count all short standalone lines across the corpus
    # -----------------------------------------------------------------------
    print("Counting short lines...")
    line_counter: Counter[str] = Counter()

    for text in tqdm(df["content_clean_ocr"].dropna(), desc="Scanning pages", unit="page"):
        if not isinstance(text, str):
            text = str(text)
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if 1 <= len(stripped) <= MAX_LINE_LEN:
                line_counter[stripped.upper()] += 1

    print(f"  Found {len(line_counter):,} unique short lines\n")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print(SEPARATOR)
    print(
        f"TOP {TOP_N} SHORT LINES  (max {MAX_LINE_LEN} chars, min {MIN_COUNT} occurrences)"
    )
    print(f"{'COUNT':>8}  {'RULE':>18}  LINE")
    print(f"{'-----':>8}  {'----':>18}  {'----'}")
    print(SEPARATOR)

    already_caught: Counter[str] = Counter()
    not_caught = 0
    reviewed = 0

    for line, count in line_counter.most_common(TOP_N):
        if count < MIN_COUNT:
            break
        reviewed += 1
        rule = which_rule_catches(line)
        if rule != "NONE":
            already_caught[rule] += 1
        else:
            not_caught += 1
        print(f"  {count:>7,}  {rule:>18}  {line}")

    print(SEPARATOR)
    print(f"\nSummary (reviewed {reviewed} lines, min {MIN_COUNT} occurrences):")
    for rule, count in sorted(already_caught.items()):
        print(f"  Caught by {rule:<22}: {count}")
    print(f"  Not caught (review manually)  : {not_caught}")
    print(
        "\nNote: 'not caught' lines are not automatically metadata - "
        "review each one manually before adding a new pattern."
    )


if __name__ == "__main__":
    main()
