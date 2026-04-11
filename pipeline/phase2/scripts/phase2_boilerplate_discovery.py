"""
Phase 2: Boilerplate Discovery
==============================
Empirically identifies the most frequent short standalone lines in the
raw OCR corpus (Phase 1 output). Used to justify and validate the
boilerplate removal patterns in phase2_cleaning.py.

For each frequent line, the script reports:
  - How many times it appears across all pages
  - Whether it is already caught by the current boilerplate patterns

Run this script before finalising phase2_cleaning.py to ensure all
high-frequency boilerplate is covered.
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
PHASE2_ROOT = SCRIPT_DIR.parent
PHASE1_ROOT = PHASE2_ROOT.parent / "phase1"

INPUT_PATH = PHASE1_ROOT / "data" / "pages_phase1_structural.csv"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MAX_LINE_LEN = 60   # only consider lines shorter than this
TOP_N = 100         # how many top lines to report
MIN_COUNT = 100     # ignore lines appearing fewer than this many times
SEPARATOR = "=" * 70
REQUIRED_COLUMNS = {"content"}

# ---------------------------------------------------------------------------
# Import current boilerplate patterns from phase2_cleaning.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_cleaning import BOILERPLATE_RE


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a clear error if an expected dataset column is missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def main() -> None:
    """Run the boilerplate discovery report."""
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

    for text in tqdm(df["content"].dropna(), desc="Scanning pages", unit="page"):
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
    # Report: for each top line, show whether it is already caught
    # -----------------------------------------------------------------------
    print(SEPARATOR)
    print(
        f"TOP {TOP_N} SHORT LINES  (max {MAX_LINE_LEN} chars, min {MIN_COUNT} occurrences)"
    )
    print(f"{'COUNT':>8}  {'CAUGHT':>6}  LINE")
    print(f"{'-----':>8}  {'------':>6}  {'----'}")
    print(SEPARATOR)

    already_caught = 0
    not_caught = 0
    reviewed = 0

    for line, count in line_counter.most_common(TOP_N):
        if count < MIN_COUNT:
            break
        reviewed += 1
        caught = bool(BOILERPLATE_RE.fullmatch(line))
        status = "YES" if caught else "NO"
        if caught:
            already_caught += 1
        else:
            not_caught += 1
        print(f"  {count:>7,}  {status:>6}  {line}")

    print(SEPARATOR)
    print(f"\nSummary (reviewed {reviewed} lines):")
    print(f"  Already caught by current patterns : {already_caught}")
    print(f"  Not caught (review manually)       : {not_caught}")
    print(
        "\nNote: 'not caught' lines are not automatically boilerplate - "
        "review each one manually before adding a new pattern."
    )


if __name__ == "__main__":
    main()
