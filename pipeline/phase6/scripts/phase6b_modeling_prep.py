"""
Phase 6b: Modeling Preparation
Strips residual archive tokens, chunks long documents, drops short ones.
Input : phase6/data/documents_for_modeling.csv
Output: phase6/data/documents_final.csv
"""

import re
import random
from pathlib import Path

import pandas as pd

random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parents[1]
INPUT_PATH  = BASE / "data" / "documents_for_modeling.csv"
OUTPUT_PATH = BASE / "data" / "documents_final.csv"

# ── Step 1a — Token-level blacklist ───────────────────────────────────────────
ARCHIVE_TOKEN_BLACKLIST = {
    'umbra', 'noforn', 'orcon', 'wnintel', 'moray', 'tud',
    'decl', 'drv', 'css', 'originator', 'ernment',
    'fpmr', 'cfr', 'sgswirl', 'hcf', 'limdis',
    'rybat', 'exdis', 'nodis', 'typic', 'slugs',
    'docid', 'nw', 'iden', 'mhfno', 'sensind',
    'cite', 'ref', 'per', 'via',
}

def _apply_token_blacklist(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return " ".join(t for t in text.split() if t.lower() not in ARCHIVE_TOKEN_BLACKLIST)

# ── Step 1b — Phrase-level stripping ─────────────────────────────────────────
_META_PHRASES = [
    "doc id",
    "jfk assassination system identification",
    "record number",
    "document type",
    "textual document",
    "current status",
    "opening criteria",
    "originating",
    "last review",
    "restrictions",
    "nsa hcf",
    "moray",
    "agency file number",
    "records series",
    "record series",
]
_META_RE = re.compile(
    r"(?<!\S)(?:" +
    "|".join(re.escape(p) for p in sorted(_META_PHRASES, key=len, reverse=True)) +
    r")(?!\S)",
    re.IGNORECASE,
)

def _strip_phrases(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _META_RE.sub(" ", text)
    return re.sub(r" {2,}", " ", cleaned).strip()

def clean_text(text: str) -> str:
    return _strip_phrases(_apply_token_blacklist(text))

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading Phase 6 data …")
df = pd.read_csv(INPUT_PATH, low_memory=False)
docs_before_chunking = len(df)
print(f"  Loaded {docs_before_chunking:,} documents")

# ── Step 1 — Strip ────────────────────────────────────────────────────────────
print("Stripping archive tokens and phrases …")
df["document_text"]       = df["document_text"].apply(clean_text)
df["document_text_lemma"] = df["document_text_lemma"].apply(clean_text)
df["token_count"]         = df["document_text"].str.split().str.len().fillna(0).astype(int)
df["token_count_lemma"]   = df["document_text_lemma"].str.split().str.len().fillna(0).astype(int)

# ── Step 2 — Chunk documents over 5,000 tokens ───────────────────────────────
CHUNK_SIZE = 5_000

print("Chunking long documents …")
rows = []

for _, row in df.iterrows():
    tokens     = row["document_text"].split() if isinstance(row["document_text"], str) else []
    tokens_lem = row["document_text_lemma"].split() if isinstance(row["document_text_lemma"], str) else []

    if len(tokens) <= CHUNK_SIZE:
        rows.append(row.to_dict())
        continue

    n_chunks = (len(tokens) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(n_chunks):
        chunk_tok = tokens[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        chunk_lem = tokens_lem[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        new_row = row.to_dict()
        new_row["file_id"]             = f"{row['file_id']}_chunk_{i+1:03d}"
        new_row["document_text"]       = " ".join(chunk_tok)
        new_row["document_text_lemma"] = " ".join(chunk_lem)
        new_row["token_count"]         = len(chunk_tok)
        new_row["token_count_lemma"]   = len(chunk_lem)
        rows.append(new_row)

df_chunked = pd.DataFrame(rows)
docs_after_chunking = len(df_chunked)

# ── Step 3 — Drop documents with fewer than 50 tokens ────────────────────────
df_final     = df_chunked[df_chunked["token_count"] >= 50].copy()
docs_dropped = docs_after_chunking - len(df_final)

# ── Save ──────────────────────────────────────────────────────────────────────
df_final.to_csv(OUTPUT_PATH, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
tc = df_final["token_count"]

bins = [
    ("<50",        (tc < 50).sum()),
    ("50–200",     ((tc >= 50)   & (tc < 200)).sum()),
    ("200–500",    ((tc >= 200)  & (tc < 500)).sum()),
    ("500–1000",   ((tc >= 500)  & (tc < 1000)).sum()),
    ("1000–5000",  ((tc >= 1000) & (tc <= 5000)).sum()),
    ("5000+",      (tc > 5000).sum()),
]

print("\n" + "=" * 60)
print("PHASE 6b MODELING PREP SUMMARY")
print("=" * 60)
print(f"Documents before chunking  : {docs_before_chunking:,}")
print(f"Documents after  chunking  : {docs_after_chunking:,}")
print(f"Documents dropped (<50 tok): {docs_dropped:,}")
print(f"Final document count       : {len(df_final):,}")
print(f"Max token count            : {tc.max():,}  (should be ≤ 5,000)")

print("\nToken distribution:")
for label, count in bins:
    bar = "█" * int(count / len(df_final) * 40) if len(df_final) else ""
    print(f"  {label:<12} {count:>6,}  {count/len(df_final)*100:>5.1f}%  {bar}")

print("\n5 random document previews:")
sample = df_final[df_final["token_count"] > 0].sample(min(5, len(df_final)), random_state=42)
for i, (_, row) in enumerate(sample.iterrows(), 1):
    preview = str(row["document_text"])[:300]
    print(f"\n  [{i}] {row['file_id']}  ({row['token_count']:,} tokens)")
    print(f"      {preview}")

print(f"\nSaved → {OUTPUT_PATH.relative_to(BASE.parent)}")
print(f"Rows: {len(df_final):,}  |  Columns: {len(df_final.columns)}")
