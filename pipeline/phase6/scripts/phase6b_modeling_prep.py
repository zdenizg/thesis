"""
Phase 6B — Modeling Preparation
================================
Input:  phase6/data/documents_for_modeling.csv
Output: phase6/data/documents_final.csv

Three-step post-processing:
  1. Strip residual archive tokens and metadata phrases
  2. Chunk documents over CHUNK_SIZE tokens into numbered sub-documents
  3. Drop documents with fewer than MIN_DOC_TOKENS tokens

Dependencies: pandas, tqdm, re, pathlib
"""

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE6_DIR = SCRIPT_DIR.parent

INPUT_CSV = PHASE6_DIR / "data" / "documents_for_modeling.csv"
OUTPUT_CSV = PHASE6_DIR / "data" / "documents_final.csv"

SEPARATOR = "=" * 60
REQUIRED_COLUMNS = {
    "file_id",
    "document_text",
    "document_text_lemma",
    "token_count",
    "token_count_lemma",
}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
CHUNK_SIZE = 5_000
MIN_DOC_TOKENS = 50

# ---------------------------------------------------------------------------
# Step 1a — Token-level blacklist
# ---------------------------------------------------------------------------
ARCHIVE_TOKEN_BLACKLIST = {
    'umbra', 'noforn', 'orcon', 'wnintel', 'moray', 'tud',
    'decl', 'drv', 'css', 'originator', 'ernment',
    'fpmr', 'cfr', 'sgswirl', 'hcf', 'limdis',
    'rybat', 'exdis', 'nodis', 'typic', 'slugs',
    'docid', 'nw', 'iden', 'mhfno', 'sensind',
}

# ---------------------------------------------------------------------------
# Step 1b — Phrase-level stripping
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Text-cleaning functions
# ---------------------------------------------------------------------------

def _apply_token_blacklist(text: str) -> str:
    """Remove blacklisted archive tokens from whitespace-delimited text."""
    if not isinstance(text, str) or not text:
        return ""
    return " ".join(t for t in text.split() if t.lower() not in ARCHIVE_TOKEN_BLACKLIST)


def _strip_phrases(text: str) -> str:
    """Remove configured metadata phrases and normalize extra spaces."""
    if not isinstance(text, str):
        return ""
    cleaned = _META_RE.sub(" ", text)
    return re.sub(r" {2,}", " ", cleaned).strip()


def clean_text(text: str) -> str:
    """Apply token-level and phrase-level cleanup to a document text."""
    return _strip_phrases(_apply_token_blacklist(text))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    """Print a consistently formatted section header."""
    print(f"\n{title}")


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print_section("Loading data...")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    validate_columns(df, REQUIRED_COLUMNS, INPUT_CSV.name)
    print(f"  Input:  {INPUT_CSV.name:<35} {len(df):,} documents")
    return df


def strip_archive_residue(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Stripping archive tokens and phrases...")
    tqdm.pandas(desc="Cleaning text")
    df["document_text"] = df["document_text"].progress_apply(clean_text)
    tqdm.pandas(desc="Cleaning lemma text")
    df["document_text_lemma"] = df["document_text_lemma"].progress_apply(clean_text)
    df["token_count"] = df["document_text"].str.split().str.len().fillna(0).astype(int)
    df["token_count_lemma"] = df["document_text_lemma"].str.split().str.len().fillna(0).astype(int)
    return df


def chunk_long_documents(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Chunking long documents...")
    if df.empty:
        return df.copy()

    n_long = int((df["token_count"] > CHUNK_SIZE).sum())
    print(f"  Documents over {CHUNK_SIZE:,} tokens: {n_long:,}")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking", unit="doc"):
        tokens = row["document_text"].split() if isinstance(row["document_text"], str) else []
        tokens_lem = row["document_text_lemma"].split() if isinstance(row["document_text_lemma"], str) else []

        if len(tokens) <= CHUNK_SIZE:
            rows.append(row.to_dict())
            continue

        n_chunks = (len(tokens) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in range(n_chunks):
            chunk_tok = tokens[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
            chunk_lem = tokens_lem[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
            new_row = row.to_dict()
            new_row["file_id"] = f"{row['file_id']}_chunk_{i + 1:03d}"
            new_row["document_text"] = " ".join(chunk_tok)
            new_row["document_text_lemma"] = " ".join(chunk_lem)
            new_row["token_count"] = len(chunk_tok)
            new_row["token_count_lemma"] = len(chunk_lem)
            rows.append(new_row)

    return pd.DataFrame(rows, columns=df.columns)


def drop_short_documents(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    print_section("Dropping short documents...")
    before = len(df)

    # Drop by raw token count
    df = df[df["token_count"] >= MIN_DOC_TOKENS].copy()
    dropped_raw = before - len(df)
    print(f"  Dropped {dropped_raw:,} documents with raw tokens < {MIN_DOC_TOKENS}")

    # Drop by lemma token count
    before_lemma = len(df)
    df = df[df["token_count_lemma"] >= MIN_DOC_TOKENS].copy()
    dropped_lemma = before_lemma - len(df)
    print(f"  Dropped {dropped_lemma:,} documents with lemma tokens < {MIN_DOC_TOKENS}")

    return df, dropped_raw, dropped_lemma


def save_output(df: pd.DataFrame) -> None:
    print_section("Saving output...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  → {OUTPUT_CSV.relative_to(PHASE6_DIR)}  {len(df):>10,} documents")


def print_summary(
    docs_before: int,
    docs_after_chunk: int,
    dropped_raw: int,
    dropped_lemma: int,
    df_final: pd.DataFrame,
) -> None:
    tc = df_final["token_count"]
    n_final = len(df_final)

    def fmt_stat(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:>8.1f}"

    def fmt_max(series: pd.Series) -> str:
        if series.empty:
            return "n/a"
        return f"{int(series.max()):>8,}"

    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  Documents before chunking   : {docs_before:>8,}")
    print(f"  Documents after chunking    : {docs_after_chunk:>8,}")
    print(f"  Dropped (raw tokens < {MIN_DOC_TOKENS})   : {dropped_raw:>8,}")
    print(f"  Dropped (lemma tokens < {MIN_DOC_TOKENS}) : {dropped_lemma:>8,}")
    print(f"  Final document count        : {n_final:>8,}")
    print(f"  Mean tokens per document   : {fmt_stat(tc.mean())}")
    print(f"  Median tokens per document : {fmt_stat(tc.median())}")
    print(f"  Max token count            : {fmt_max(tc)}  (limit: {CHUNK_SIZE:,})")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEPARATOR)
    print("PHASE 6B — Modeling Preparation")
    print(SEPARATOR)

    df = load_data()
    docs_before = len(df)
    df = strip_archive_residue(df)
    df_chunked = chunk_long_documents(df)
    docs_after_chunk = len(df_chunked)
    df_final, dropped_raw, dropped_lemma = drop_short_documents(df_chunked)
    save_output(df_final)
    print_summary(docs_before, docs_after_chunk, dropped_raw, dropped_lemma, df_final)


if __name__ == "__main__":
    main()
