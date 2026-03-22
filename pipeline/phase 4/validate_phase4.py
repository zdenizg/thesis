"""
Phase 4 Validation Script
Validates stopword removal and token normalization output.
Read-only — no modifications to any dataset.
"""

import pandas as pd
import numpy as np
from collections import Counter
import random
import re

random.seed(42)

# ─────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_HERE, "data", "pages_phase4_modeltext.csv"), low_memory=False)

# ─────────────────────────────────────────────
# 2. Basic dataset information
# ─────────────────────────────────────────────
print("=" * 60)
print("SECTION 2 — BASIC DATASET INFORMATION")
print("=" * 60)
print(f"Rows               : {len(df):,}")
print(f"Unique file_ids    : {df['file_id'].nunique():,}")
print(f"Columns            : {list(df.columns)}")
print("\nMissing values per column:")
print(df.isnull().sum().to_string())

# ─────────────────────────────────────────────
# 3. Descriptive statistics
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — DESCRIPTIVE STATISTICS")
print("=" * 60)
for col in ["word_count_clean", "token_count_model_no_lemma", "token_count_model_lemma"]:
    print(f"\n--- {col} ---")
    print(df[col].describe().to_string())

# ─────────────────────────────────────────────
# 4. Token retention ratio
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — TOKEN RETENTION RATIO")
print("=" * 60)

df["token_ratio"] = df["token_count_model_no_lemma"] / df["word_count_clean"].replace(0, np.nan)

print("\nDescriptive statistics for token_ratio:")
print(df["token_ratio"].describe().to_string())

low_ratio = df[df["token_ratio"] < 0.3].copy()
print(f"\nPages with token_ratio < 0.3: {len(low_ratio):,}")

if len(low_ratio) > 0:
    sample = low_ratio.sample(min(10, len(low_ratio)), random_state=42)
    for _, row in sample.iterrows():
        clean_preview = str(row.get("content_clean_lines", ""))[:300]
        model_preview = str(row.get("content_model_no_lemma", ""))[:300]
        print(f"\n  file_id        : {row['file_id']}")
        print(f"  page_number    : {row['page_number']}")
        print(f"  word_count_clean          : {row['word_count_clean']}")
        print(f"  token_count_model_no_lemma: {row['token_count_model_no_lemma']}")
        print(f"  content_clean_lines preview:\n    {clean_preview}")
        print(f"  content_model_no_lemma preview:\n    {model_preview}")

# ─────────────────────────────────────────────
# 5. Vocabulary size comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — VOCABULARY SIZE COMPARISON")
print("=" * 60)

def build_vocab(series):
    vocab = set()
    for text in series.dropna():
        tokens = str(text).split()
        vocab.update(t.lower() for t in tokens)
    return vocab

vocab_clean = build_vocab(df["content_clean_lines"])
vocab_model = build_vocab(df["content_model_no_lemma"])

reduction = (1 - len(vocab_model) / len(vocab_clean)) * 100
print(f"Vocabulary size — content_clean_lines       : {len(vocab_clean):,}")
print(f"Vocabulary size — content_model_no_lemma    : {len(vocab_model):,}")
print(f"Percentage reduction                        : {reduction:.1f}%")

# ─────────────────────────────────────────────
# 6. Top 30 tokens
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 — TOP 30 TOKENS")
print("=" * 60)

def top_tokens(series, n=30):
    counter = Counter()
    for text in series.dropna():
        tokens = str(text).split()
        counter.update(t.lower() for t in tokens)
    return counter.most_common(n)

print("\nTop 30 — content_clean_lines:")
for rank, (tok, cnt) in enumerate(top_tokens(df["content_clean_lines"]), 1):
    print(f"  {rank:>2}. {tok:<25} {cnt:,}")

print("\nTop 30 — content_model_no_lemma:")
for rank, (tok, cnt) in enumerate(top_tokens(df["content_model_no_lemma"]), 1):
    print(f"  {rank:>2}. {tok:<25} {cnt:,}")

# ─────────────────────────────────────────────
# 7. Important entity counts
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7 — IMPORTANT ENTITY VERIFICATION")
print("=" * 60)

entities = ["oswald", "soviet", "mexico", "cia", "embassy"]

def count_token_in_series(series, token):
    count = 0
    for text in series.dropna():
        count += str(text).lower().split().count(token)
    return count

print(f"\n{'Token':<15} {'In content_clean_lines':>22} {'In content_model_no_lemma':>26}")
print("-" * 65)
for ent in entities:
    cnt_clean = count_token_in_series(df["content_clean_lines"], ent)
    cnt_model = count_token_in_series(df["content_model_no_lemma"], ent)
    print(f"{ent:<15} {cnt_clean:>22,} {cnt_model:>26,}")

# ─────────────────────────────────────────────
# 8. Pages with zero model tokens
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 8 — PAGES WITH ZERO MODEL TOKENS")
print("=" * 60)

zero_token = df[df["token_count_model_no_lemma"] == 0]
print(f"Pages where token_count_model_no_lemma == 0: {len(zero_token):,}")

if len(zero_token) > 0:
    sample_zero = zero_token.sample(min(10, len(zero_token)), random_state=42)
    for _, row in sample_zero.iterrows():
        clean_preview = str(row.get("content_clean_lines", ""))[:400]
        model_preview = str(row.get("content_model_no_lemma", ""))[:400]
        print(f"\n  file_id        : {row['file_id']}")
        print(f"  page_number    : {row['page_number']}")
        print(f"  content_clean_lines:\n    {clean_preview}")
        print(f"  content_model_no_lemma:\n    {model_preview}")

# ─────────────────────────────────────────────
# 9. Random page comparisons
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 9 — 10 RANDOM PAGE COMPARISONS")
print("=" * 60)

sample_pages = df.sample(10, random_state=42)
for i, (_, row) in enumerate(sample_pages.iterrows(), 1):
    clean_preview = str(row.get("content_clean_lines", ""))[:500]
    model_preview = str(row.get("content_model_no_lemma", ""))[:500]
    print(f"\n--- Page {i} | file_id={row['file_id']} | page={row['page_number']} ---")
    print(f"  [CLEAN]  {clean_preview}")
    print(f"  [MODEL]  {model_preview}")

# ─────────────────────────────────────────────
# 10. Diagnostic summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 10 — DIAGNOSTIC SUMMARY")
print("=" * 60)

median_ratio = df["token_ratio"].median()
pct_low = (df["token_ratio"] < 0.3).sum() / len(df) * 100
n_zero = len(zero_token)

entity_counts = {ent: count_token_in_series(df["content_model_no_lemma"], ent) for ent in entities}
entities_present = all(v > 0 for v in entity_counts.values())

print(f"\nToken retention (median ratio)  : {median_ratio:.3f}")
print(f"Pages with ratio < 0.3         : {pct_low:.1f}% ({(df['token_ratio'] < 0.3).sum():,} pages)")
print(f"Pages with zero model tokens   : {n_zero:,}")
print(f"Vocabulary reduction           : {reduction:.1f}%")
print(f"All key entities present       : {entities_present}")

print("\nConclusion:")
if median_ratio >= 0.3 and pct_low < 10 and entities_present and reduction < 70:
    print("  Phase 4 output appears VALID and ready for topic modeling.")
    print("  Token reduction is reasonable; important entities are preserved.")
else:
    issues = []
    if median_ratio < 0.3:
        issues.append(f"median token ratio is low ({median_ratio:.3f})")
    if pct_low >= 10:
        issues.append(f"{pct_low:.1f}% of pages have ratio < 0.3")
    if not entities_present:
        missing = [e for e, v in entity_counts.items() if v == 0]
        issues.append(f"missing entities: {missing}")
    if reduction >= 70:
        issues.append(f"vocabulary reduction is high ({reduction:.1f}%)")
    print("  WARNING — potential issues detected:")
    for iss in issues:
        print(f"    - {iss}")
