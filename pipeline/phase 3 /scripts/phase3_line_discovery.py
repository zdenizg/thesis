"""
Phase 3: Line-Level Metadata Discovery
=======================================
Empirically identifies the most frequent short standalone lines in the
Phase 2 cleaned corpus. Used to justify and validate the metadata-removal
patterns in phase3_line_filtering.py.

For each frequent line, the script reports:
  - How many times it appears across all pages
  - Which Phase 3 rule already catches it (or NONE if uncaught)
  - A suggested action

Run this script before finalising phase3_line_filtering.py to ensure all
high-frequency metadata lines are covered.
"""

import os
import sys
from collections import Counter
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PHASE3_ROOT = os.path.dirname(SCRIPT_DIR)
PHASE2_ROOT = os.path.join(os.path.dirname(PHASE3_ROOT), "phase 2")

INPUT_PATH = os.path.join(PHASE2_ROOT, "data", "pages_phase2_cleaned.csv")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MAX_LINE_LEN = 60    # only consider lines shorter than this
TOP_N        = 100   # how many top lines to report
MIN_COUNT    = 50    # ignore lines appearing fewer than this many times

# ---------------------------------------------------------------------------
# Import current filtering rules from phase3_line_filtering.py
# ---------------------------------------------------------------------------
sys.path.insert(0, SCRIPT_DIR)
from phase3_line_filtering import (
    _line_is_code_like,
    _line_is_metadata_phrase,
    _line_is_standalone_keyword,
    _line_is_filing_action,
    _line_is_low_content,
)

def which_rule_catches(line: str) -> str:
    """Return the name of the first Phase 3 rule that catches this line, or NONE."""
    lc = line.lower().strip()
    if _line_is_code_like(lc):
        return "code_like"
    if _line_is_metadata_phrase(lc):
        return "metadata_phrase"
    if _line_is_standalone_keyword(lc):
        return "standalone_keyword"
    if _line_is_filing_action(lc):
        return "filing_action"
    if _line_is_low_content(lc):
        return "low_content"
    return "NONE"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Loading: {INPUT_PATH}")
if not os.path.exists(INPUT_PATH):
    sys.exit(f"ERROR: Input file not found:\n  {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"  Loaded {len(df):,} rows\n")

# ---------------------------------------------------------------------------
# Count all short standalone lines across the corpus
# ---------------------------------------------------------------------------
print("Counting short lines …")
line_counter: Counter = Counter()

for text in df["content_clean_ocr"].dropna():
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) == 0:
            continue
        if 1 <= len(stripped) <= MAX_LINE_LEN:
            line_counter[stripped.upper()] += 1

print(f"  Found {len(line_counter):,} unique short lines\n")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
SEP = "=" * 70

print(SEP)
print(f"TOP {TOP_N} SHORT LINES  (max {MAX_LINE_LEN} chars, min {MIN_COUNT} occurrences)")
print(f"{'COUNT':>8}  {'RULE':>18}  LINE")
print(f"{'-----':>8}  {'----':>18}  {'----'}")
print(SEP)

already_caught = Counter()
not_caught     = 0

for line, count in line_counter.most_common(TOP_N):
    if count < MIN_COUNT:
        break
    rule = which_rule_catches(line)
    if rule != "NONE":
        already_caught[rule] += 1
    else:
        not_caught += 1
    print(f"  {count:>7,}  {rule:>18}  {line}")

print(SEP)
print(f"\nSummary (of top {TOP_N}, min {MIN_COUNT} occurrences):")
for rule, n in sorted(already_caught.items()):
    print(f"  Caught by {rule:<22}: {n}")
print(f"  Not caught (review manually)  : {not_caught}")
print(
    "\nNote: 'not caught' lines are not automatically metadata — "
    "review each one manually before adding a new pattern."
)
