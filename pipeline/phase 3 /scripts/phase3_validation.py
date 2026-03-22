"""
Phase 3 Validation Script
=========================
Read-only validation of pages_phase3_linefiltered.csv.
Checks that line-level metadata filtering was conservative and correct.
"""

import re
import random
import pandas as pd
import numpy as np
from collections import Counter

random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "pages_phase3_linefiltered.csv")

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("=" * 70)
print("PHASE 3 VALIDATION")
print("=" * 70)

print("\n[1] Loading dataset …")
df = pd.read_csv(INPUT_PATH, low_memory=False)

# ---------------------------------------------------------------------------
# 2. Basic dataset information
# ---------------------------------------------------------------------------
print("\n[2] BASIC DATASET INFORMATION")
print(f"  Rows              : {len(df):,}")
print(f"  Unique file_ids   : {df['file_id'].nunique():,}")
print(f"  Columns ({len(df.columns)})    : {df.columns.tolist()}")
print("\n  Missing values per column:")
missing = df.isnull().sum()
for col, n in missing[missing > 0].items():
    print(f"    {col:<35} {n:>8,}  ({100*n/len(df):.1f}%)")

# ---------------------------------------------------------------------------
# 3. Descriptive statistics for line-removal metrics
# ---------------------------------------------------------------------------
print("\n[3] LINE-REMOVAL DESCRIPTIVE STATISTICS")
metrics = ['lines_before', 'lines_after', 'lines_removed', 'line_removal_ratio']
print(df[metrics].describe().round(3).to_string())

# Distribution of removal ratios
bins = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.01]
labels = ['0-5%','5-10%','10-20%','20-30%','30-40%','40-50%','50-75%','75-100%']
df['removal_bin'] = pd.cut(df['line_removal_ratio'], bins=bins, labels=labels, right=False)
print("\n  Removal ratio distribution (pages with text only):")
has_text = df[df['lines_before'] > 0]
for label, count in has_text['removal_bin'].value_counts().sort_index().items():
    pct = 100 * count / len(has_text)
    print(f"    {label:<12} {count:>7,}  ({pct:5.1f}%)")

# ---------------------------------------------------------------------------
# 4. High-removal pages (ratio > 0.4)  — 10 random examples
# ---------------------------------------------------------------------------
print("\n[4] HIGH-REMOVAL PAGES  (line_removal_ratio > 0.40)  — 10 random examples")
high = df[df['line_removal_ratio'] > 0.40].copy()
print(f"  Total high-removal pages: {len(high):,}")

sample_high = high.sample(min(10, len(high)), random_state=42)
for _, row in sample_high.iterrows():
    before = str(row['content_clean_ocr'] or '')[:500].replace('\n', ' ↵ ')
    after  = str(row['content_clean_lines'] or '')[:500].replace('\n', ' ↵ ')
    print(f"\n  file_id={row['file_id']}  page={row['page_number']}  "
          f"lines_before={row['lines_before']}  lines_after={row['lines_after']}  "
          f"ratio={row['line_removal_ratio']:.2f}")
    print(f"  BEFORE: {before}")
    print(f"  AFTER : {after}")

# ---------------------------------------------------------------------------
# 5. Low-removal pages (ratio < 0.1)  — 10 random examples
# ---------------------------------------------------------------------------
print("\n[5] LOW-REMOVAL PAGES  (line_removal_ratio < 0.10)  — 10 random examples")
low = df[(df['line_removal_ratio'] < 0.10) & (df['lines_before'] > 5)].copy()
print(f"  Total low-removal pages: {len(low):,}")

sample_low = low.sample(min(10, len(low)), random_state=99)
for _, row in sample_low.iterrows():
    before = str(row['content_clean_ocr'] or '')[:500].replace('\n', ' ↵ ')
    after  = str(row['content_clean_lines'] or '')[:500].replace('\n', ' ↵ ')
    print(f"\n  file_id={row['file_id']}  page={row['page_number']}  "
          f"lines_before={row['lines_before']}  lines_after={row['lines_after']}  "
          f"ratio={row['line_removal_ratio']:.2f}")
    print(f"  BEFORE: {before}")
    print(f"  AFTER : {after}")

# ---------------------------------------------------------------------------
# 6. Vocabulary comparison: top-30 tokens before vs after
# ---------------------------------------------------------------------------
print("\n[6] TOP-30 TOKEN COMPARISON  (content_clean_ocr  vs  content_clean_lines)")

def top_tokens(series, n=30):
    """Count word-like tokens across all texts in a Series."""
    counter = Counter()
    for text in series.dropna():
        tokens = re.findall(r'[a-z]{2,}', str(text))
        counter.update(tokens)
    return counter.most_common(n)

tok_before = top_tokens(df['content_clean_ocr'])
tok_after  = top_tokens(df['content_clean_lines'])

before_dict = dict(tok_before)
after_dict  = dict(tok_after)

# Tokens of interest: routing words that should decrease
routing_tokens = {
    'division', 'station', 'distribution', 'dissem', 'rybat',
    'nodis', 'limdis', 'exdis', 'priority', 'immediate',
    'secret', 'confidential', 'unclassified', 'film',
}

print(f"\n  {'TOKEN':<20} {'BEFORE':>12} {'AFTER':>12} {'CHANGE':>10}")
print("  " + "-" * 56)

# Show top-30 before with after count
for token, cnt_b in tok_before:
    cnt_a = after_dict.get(token, 0)
    change = cnt_a - cnt_b
    flag = "  ← ROUTING" if token in routing_tokens else ""
    print(f"  {token:<20} {cnt_b:>12,} {cnt_a:>12,} {change:>+10,}{flag}")

print(f"\n  Routing token counts (selected):")
print(f"  {'TOKEN':<20} {'BEFORE':>12} {'AFTER':>12} {'% CHANGE':>10}")
print("  " + "-" * 56)

# Count all occurrences of routing tokens in full text (not just top-30)
def count_token_in_series(series, token):
    total = 0
    for text in series.dropna():
        total += len(re.findall(r'\b' + token + r'\b', str(text)))
    return total

for tok in sorted(routing_tokens):
    cnt_b = count_token_in_series(df['content_clean_ocr'], tok)
    cnt_a = count_token_in_series(df['content_clean_lines'], tok)
    pct = (100 * (cnt_a - cnt_b) / cnt_b) if cnt_b > 0 else 0
    print(f"  {tok:<20} {cnt_b:>12,} {cnt_a:>12,} {pct:>+9.1f}%")

# ---------------------------------------------------------------------------
# 7. Character retention ratio
# ---------------------------------------------------------------------------
print("\n[7] CHARACTER RETENTION RATIO  (char_ratio = len(after) / len(before))")

has_both = df[df['content_clean_ocr'].notna() & df['content_clean_lines'].notna()].copy()
has_both = has_both[has_both['content_clean_ocr'].str.len() > 0].copy()
has_both['char_ratio'] = (
    has_both['content_clean_lines'].str.len() /
    has_both['content_clean_ocr'].str.len()
)

print(has_both['char_ratio'].describe().round(4).to_string())

low_retention = has_both[has_both['char_ratio'] < 0.5]
print(f"\n  Pages with char_ratio < 0.50: {len(low_retention):,}  "
      f"({100*len(low_retention)/len(has_both):.1f}% of pages with text)")

print("\n  Sample low-retention pages:")
for _, row in low_retention.sample(min(5, len(low_retention)), random_state=7).iterrows():
    before = str(row['content_clean_ocr'])[:400].replace('\n', ' ↵ ')
    after  = str(row['content_clean_lines'])[:400].replace('\n', ' ↵ ')
    print(f"\n  file_id={row['file_id']}  page={row['page_number']}  "
          f"char_ratio={row['char_ratio']:.3f}  "
          f"line_removal_ratio={row['line_removal_ratio']:.2f}")
    print(f"  BEFORE: {before}")
    print(f"  AFTER : {after}")

# ---------------------------------------------------------------------------
# 8. 10 completely random page comparisons
# ---------------------------------------------------------------------------
print("\n[8] 10 RANDOM PAGE COMPARISONS (visual narrative check)")

rand_pages = df[df['content_clean_ocr'].notna()].sample(10, random_state=55)
for i, (_, row) in enumerate(rand_pages.iterrows(), 1):
    before = str(row['content_clean_ocr'])[:600].replace('\n', ' ↵ ')
    after  = str(row['content_clean_lines'])[:600].replace('\n', ' ↵ ')
    print(f"\n  [{i}] file_id={row['file_id']}  page={row['page_number']}  "
          f"lines_before={row['lines_before']}  lines_after={row['lines_after']}  "
          f"ratio={row['line_removal_ratio']:.2f}")
    print(f"  BEFORE: {before}")
    print(f"  AFTER : {after}")

# ---------------------------------------------------------------------------
# 9. Diagnostic summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("[9] DIAGNOSTIC SUMMARY")
print("=" * 70)

avg_ratio  = has_both['line_removal_ratio'].mean()
med_ratio  = has_both['line_removal_ratio'].median()
pct_zero   = 100 * (has_both['line_removal_ratio'] == 0).sum() / len(has_both)
pct_gt30   = 100 * (has_both['line_removal_ratio'] > 0.30).sum() / len(has_both)
pct_gt50   = 100 * (has_both['line_removal_ratio'] > 0.50).sum() / len(has_both)
avg_char   = has_both['char_ratio'].mean()

print(f"  Average line_removal_ratio   : {avg_ratio:.3f}  ({avg_ratio*100:.1f}%)")
print(f"  Median  line_removal_ratio   : {med_ratio:.3f}  ({med_ratio*100:.1f}%)")
print(f"  Pages with 0 % removed       : {pct_zero:.1f}%")
print(f"  Pages with >30% removed      : {pct_gt30:.1f}%")
print(f"  Pages with >50% removed      : {pct_gt50:.1f}%")
print(f"  Average character retention  : {avg_char:.3f}  ({avg_char*100:.1f}%)")

print()
if avg_ratio < 0.15 and pct_gt50 < 5:
    verdict = "CONSERVATIVE ✓  Average removal is low; most pages retain >85% of lines."
elif avg_ratio < 0.25 and pct_gt50 < 10:
    verdict = "MODERATE  —  Removal is moderate. Spot-check high-removal pages."
else:
    verdict = "AGGRESSIVE ✗  High average removal. Review filtering rules."

print(f"  Verdict: {verdict}")

# Routing token reduction summary
print()
routing_decreases = []
for tok in sorted(routing_tokens):
    cnt_b = count_token_in_series(df['content_clean_ocr'], tok)
    cnt_a = count_token_in_series(df['content_clean_lines'], tok)
    if cnt_b > 0:
        routing_decreases.append((tok, cnt_b, cnt_a, (cnt_a-cnt_b)/cnt_b))

if all(pct <= 0 for _, _, _, pct in routing_decreases):
    print("  Routing token reduction: ALL monitored tokens decreased or unchanged ✓")
else:
    increased = [(t, p) for t, b, a, p in routing_decreases if p > 0]
    print(f"  Routing tokens that INCREASED after filtering: "
          f"{[(t, f'{p*100:+.1f}%') for t,p in increased]}")

print(f"  Phase 3 status: script ran on {len(df):,} pages, output saved to")
print(f"  {INPUT_PATH}")
print("=" * 70)
