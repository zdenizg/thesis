"""
Phase 4: Token-Level Stopword Discovery
========================================
Empirically identifies the most frequent tokens in the Phase 3 corpus
BEFORE archive stopword removal, to justify and validate ARCHIVE_STOPWORDS.

For each high-frequency token the script reports:
  - Raw frequency in the corpus
  - Which filter catches it:
      invalid       : fails is_valid_token (too short, contains digit, pure punct)
      nltk          : removed by NLTK English stopwords
      archive       : removed by ARCHIVE_STOPWORDS
      redundant     : in both NLTK and ARCHIVE (duplicate — can clean up)
      UNCAUGHT      : passes all filters → reaches the topic model

Run this before finalising ARCHIVE_STOPWORDS to ensure high-frequency
non-substantive tokens are covered, and to confirm substantive tokens
are not being over-removed.
"""

import os
import sys
from collections import Counter
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PHASE4_ROOT = os.path.dirname(SCRIPT_DIR)
PHASE3_DATA = os.path.join(os.path.dirname(PHASE4_ROOT), "phase 3 ", "data",
                           "pages_phase3_linefiltered.csv")

# ── Parameters ────────────────────────────────────────────────────────────────
TOP_N     = 150    # how many top tokens to report
MIN_COUNT = 200    # ignore tokens appearing fewer than this many times

# ── Import current filters from phase4_modeltext.py ───────────────────────────
sys.path.insert(0, SCRIPT_DIR)
from phase4_modeltext import (
    is_valid_token,
    ENGLISH_STOPWORDS,
    ARCHIVE_STOPWORDS,
    word_tokenize,
)

def which_filter(token: str) -> str:
    """Return which filter catches this token, or UNCAUGHT."""
    in_nltk    = token in ENGLISH_STOPWORDS
    in_archive = token in ARCHIVE_STOPWORDS
    if not is_valid_token(token):
        return "invalid"
    if in_nltk and in_archive:
        return "redundant"   # in both — ARCHIVE entry is unnecessary
    if in_nltk:
        return "nltk"
    if in_archive:
        return "archive"
    return "UNCAUGHT"

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading: {PHASE3_DATA}")
if not os.path.exists(PHASE3_DATA):
    sys.exit(f"ERROR: Input file not found:\n  {PHASE3_DATA}")

df = pd.read_csv(PHASE3_DATA, low_memory=False)
print(f"  Loaded {len(df):,} rows\n")

# ── Count all valid tokens (no stopword removal) ──────────────────────────────
print("Counting tokens …")
token_counter: Counter = Counter()

for text in df["content_clean_lines"].dropna():
    for tok in word_tokenize(str(text).lower()):
        token_counter[tok] += 1

print(f"  Total unique raw tokens : {len(token_counter):,}")

# Keep only tokens that pass is_valid_token (length/digit/punct check)
valid_counter = Counter({t: c for t, c in token_counter.items()
                         if is_valid_token(t)})
print(f"  After is_valid_token    : {len(valid_counter):,} unique tokens\n")

# ── Report ────────────────────────────────────────────────────────────────────
SEP = "=" * 72

print(SEP)
print(f"TOP {TOP_N} VALID TOKENS  (min {MIN_COUNT} occurrences)")
print(f"{'COUNT':>9}  {'FILTER':>12}  TOKEN")
print(f"{'-----':>9}  {'------':>12}  {'-----'}")
print(SEP)

filter_counts: Counter = Counter()
uncaught_tokens = []

for token, count in valid_counter.most_common(TOP_N):
    if count < MIN_COUNT:
        break
    f = which_filter(token)
    filter_counts[f] += 1
    if f == "UNCAUGHT":
        uncaught_tokens.append((token, count))
    print(f"  {count:>8,}  {f:>12}  {token}")

print(SEP)
print(f"\nSummary (of top {TOP_N}, min {MIN_COUNT} occurrences):")
for f, n in sorted(filter_counts.items()):
    print(f"  {f:<14}: {n}")

print(f"\nUncaught tokens (review — not automatically noise):")
if uncaught_tokens:
    for tok, cnt in uncaught_tokens:
        print(f"  {cnt:>8,}  {tok}")
else:
    print("  None — all high-frequency tokens are caught by a filter.")

if filter_counts.get("redundant", 0) > 0:
    print(f"\nRedundant entries in ARCHIVE_STOPWORDS (also in NLTK — safe to remove):")
    for token, count in valid_counter.most_common(TOP_N):
        if count < MIN_COUNT:
            break
        if which_filter(token) == "redundant":
            print(f"  {count:>8,}  {token}")

print(
    "\nNote: 'UNCAUGHT' tokens are not automatically noise — "
    "review each one before adding to ARCHIVE_STOPWORDS."
)
