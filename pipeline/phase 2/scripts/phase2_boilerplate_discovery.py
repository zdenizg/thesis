"""
Phase 2: Boilerplate Discovery
==============================
Empirically identifies the most frequent short standalone lines in the
raw OCR corpus (Phase 1 output). Used to justify and validate the
boilerplate removal patterns in phase2_cleaning.py.

For each frequent line, the script reports:
  - How many times it appears across all pages
  - Whether it is already caught by the current boilerplate patterns
  - A suggested action (already caught / consider adding / keep as content)

Run this script before finalising phase2_cleaning.py to ensure all
high-frequency boilerplate is covered.
"""

import os
import sys
from collections import Counter
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PHASE2_ROOT = os.path.dirname(SCRIPT_DIR)
PHASE1_ROOT = os.path.join(os.path.dirname(PHASE2_ROOT), "phase 1 ")

INPUT_PATH = os.path.join(PHASE1_ROOT, "data", "pages_phase1_structural.csv")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MAX_LINE_LEN   = 60    # only consider lines shorter than this
TOP_N          = 100   # how many top lines to report
MIN_COUNT      = 100   # ignore lines appearing fewer than this many times

# ---------------------------------------------------------------------------
# Import current boilerplate patterns from phase2_cleaning.py
# ---------------------------------------------------------------------------
sys.path.insert(0, SCRIPT_DIR)
from phase2_cleaning import BOILERPLATE_RE

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

for text in df["content"].dropna():
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) == 0:
            continue
        if 1 <= len(stripped) <= MAX_LINE_LEN:
            line_counter[stripped.upper()] += 1

print(f"  Found {len(line_counter):,} unique short lines\n")

# ---------------------------------------------------------------------------
# Report: for each top line, show whether it is already caught
# ---------------------------------------------------------------------------
SEP = "=" * 70

print(SEP)
print(f"TOP {TOP_N} SHORT LINES  (max {MAX_LINE_LEN} chars, min {MIN_COUNT} occurrences)")
print(f"{'COUNT':>8}  {'CAUGHT':>6}  LINE")
print(f"{'-----':>8}  {'------':>6}  {'----'}")
print(SEP)

already_caught = 0
not_caught     = 0

for line, count in line_counter.most_common(TOP_N):
    if count < MIN_COUNT:
        break
    caught = bool(BOILERPLATE_RE.fullmatch(line))
    status = "YES" if caught else "NO "
    if caught:
        already_caught += 1
    else:
        not_caught += 1
    print(f"  {count:>7,}  {status:>6}  {line}")

print(SEP)
print(f"\nSummary (of top {TOP_N}):")
print(f"  Already caught by current patterns : {already_caught}")
print(f"  Not caught (review manually)       : {not_caught}")
print(
    "\nNote: 'not caught' lines are not automatically boilerplate — "
    "review each one manually before adding a new pattern."
)
