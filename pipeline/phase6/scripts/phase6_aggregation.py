"""
Phase 6: Document Aggregation
Concatenates page-level model text into one row per document.
Input : phase5/data/pages_for_modeling.csv
Output: phase6/data/documents_for_modeling.csv
"""

import random
from pathlib import Path

import pandas as pd

random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parents[1]
INPUT_PATH  = BASE.parent / "phase5" / "data" / "pages_for_modeling.csv"
OUTPUT_PATH = BASE / "data" / "documents_for_modeling.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading Phase 5 data …")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"  Loaded {len(df):,} pages across {df['file_id'].nunique():,} documents")

# ── Aggregation ───────────────────────────────────────────────────────────────
print("Aggregating by file_id …")
df = df.sort_values(["file_id", "page_number"])

def most_common(series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else None

def join_and_count(series):
    tokens = [v for v in series.dropna() if isinstance(v, str)]
    text = " ".join(tokens)
    count = sum(len(t.split()) for t in tokens)
    return text, count

docs_rows = []
for file_id, group in df.groupby("file_id", sort=False):
    text,       tc  = join_and_count(group["content_model_no_lemma"])
    text_lemma, tcl = join_and_count(group["content_model_lemma"])
    docs_rows.append({
        "file_id":             file_id,
        "document_text":       text,
        "document_text_lemma": text_lemma,
        "token_count":         tc,
        "token_count_lemma":   tcl,
        "pages_retained":      len(group),
        "pages_total":         group["number_of_pages"].max(),
        "document_type":       most_common(group["document_type"]),
        "ocr_difficulty":      most_common(group["ocr_difficulty"]),
        "includes_handwriting": group["includes_handwriting"].any(),
    })

docs = pd.DataFrame(docs_rows)
docs["retention_ratio"]   = (
    docs["pages_retained"] / docs["pages_total"].replace(0, float("nan"))
).fillna(0).clip(upper=1.0).round(4)

docs = docs[[
    "file_id", "document_text", "document_text_lemma",
    "token_count", "token_count_lemma",
    "pages_retained", "pages_total", "retention_ratio",
    "document_type", "ocr_difficulty", "includes_handwriting",
]]

# ── Save ──────────────────────────────────────────────────────────────────────
docs.to_csv(OUTPUT_PATH, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
tc        = docs["token_count"]
zero_docs = (tc == 0).sum()

bins = [
    ("<50",       (tc < 50).sum()),
    ("50–200",    ((tc >= 50)   & (tc < 200)).sum()),
    ("200–500",   ((tc >= 200)  & (tc < 500)).sum()),
    ("500–1000",  ((tc >= 500)  & (tc < 1000)).sum()),
    ("1000+",     (tc >= 1000).sum()),
]

print("\n" + "=" * 60)
print("PHASE 6 AGGREGATION SUMMARY")
print("=" * 60)
print(f"Total documents produced : {len(docs):,}")
print(f"Avg token count          : {tc.mean():.1f}")
print(f"Min token count          : {tc.min():,}")
print(f"Max token count          : {tc.max():,}")
print(f"Documents with 0 tokens  : {zero_docs:,}")

print("\nToken distribution:")
for label, count in bins:
    bar = "█" * int(count / len(docs) * 40)
    print(f"  {label:<10} {count:>6,}  {count/len(docs)*100:>5.1f}%  {bar}")

print("\nTop 10 longest documents:")
top10 = docs.nlargest(10, "token_count")[["file_id", "token_count", "pages_retained", "document_type"]]
for _, row in top10.iterrows():
    print(f"  {row['file_id']:<40} {row['token_count']:>8,} tokens  {row['pages_retained']:>4} pages  {row['document_type']}")

print("\n5 random document previews:")
sample = docs[docs["token_count"] > 0].sample(min(5, len(docs)), random_state=42)
for i, (_, row) in enumerate(sample.iterrows(), 1):
    preview = str(row["document_text"])[:300]
    print(f"\n  [{i}] {row['file_id']}  ({row['token_count']:,} tokens, {row['pages_retained']} pages)")
    print(f"      {preview}")

print(f"\nSaved → {OUTPUT_PATH.relative_to(BASE.parent)}")
print(f"Rows: {len(docs):,}  |  Columns: {len(docs.columns)}")
