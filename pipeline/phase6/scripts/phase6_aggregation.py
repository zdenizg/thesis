"""
Phase 6A — Document Aggregation
================================
Input:  phase5/data/pages_for_modeling.csv
Output: phase6/data/documents_for_modeling.csv

Concatenates page-level model text into one row per document (file_id).
Computes document-level token counts and metadata summaries.

Dependencies: pandas, tqdm, pathlib
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE6_DIR = SCRIPT_DIR.parent
PHASE5_DIR = PHASE6_DIR.parent / "phase5"

INPUT_CSV = PHASE5_DIR / "data" / "pages_for_modeling.csv"
OUTPUT_CSV = PHASE6_DIR / "data" / "documents_for_modeling.csv"

SEPARATOR = "=" * 60
REQUIRED_COLUMNS = {"file_id", "page_number", "number_of_pages",
                    "content_model_no_lemma", "content_model_lemma",
                    "document_type", "ocr_difficulty", "includes_handwriting"}

OUTPUT_COLUMN_ORDER = [
    "file_id", "document_text", "document_text_lemma",
    "token_count", "token_count_lemma",
    "pages_retained", "pages_total", "retention_ratio",
    "document_type", "ocr_difficulty", "includes_handwriting",
]


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


def most_common(series: pd.Series) -> object:
    """Return the most frequent value, or None if the series is empty."""
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else None


def join_and_count(series: pd.Series) -> tuple[str, int]:
    """Join non-null string values and count total tokens."""
    tokens = [v for v in series.dropna() if isinstance(v, str)]
    text = " ".join(tokens)
    count = sum(len(t.split()) for t in tokens)
    return text, count


def empty_documents_frame() -> pd.DataFrame:
    """Return an empty output DataFrame with the expected column order."""
    return pd.DataFrame(columns=OUTPUT_COLUMN_ORDER)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print_section("Loading data...")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    validate_columns(df, REQUIRED_COLUMNS, INPUT_CSV.name)
    print(f"  Input:  {INPUT_CSV.name:<35} {len(df):,} rows")
    return df


def aggregate_documents(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Aggregating to document level...")
    if df.empty:
        return empty_documents_frame()

    df = df.sort_values(["file_id", "page_number"])

    docs_rows = []
    for file_id, group in tqdm(df.groupby("file_id", sort=False),
                               desc="Aggregating", unit="doc"):
        text, tc = join_and_count(group["content_model_no_lemma"])
        text_lemma, tcl = join_and_count(group["content_model_lemma"])
        docs_rows.append({
            "file_id": file_id,
            "document_text": text,
            "document_text_lemma": text_lemma,
            "token_count": tc,
            "token_count_lemma": tcl,
            "pages_retained": len(group),
            "pages_total": group["number_of_pages"].max(),
            "document_type": most_common(group["document_type"]),
            "ocr_difficulty": most_common(group["ocr_difficulty"]),
            "includes_handwriting": group["includes_handwriting"].any(),
        })

    docs = pd.DataFrame(docs_rows)
    docs["retention_ratio"] = (
        docs["pages_retained"] / docs["pages_total"].replace(0, float("nan"))
    ).fillna(0).clip(upper=1.0).round(4)

    return docs[OUTPUT_COLUMN_ORDER]


def save_output(docs: pd.DataFrame) -> None:
    print_section("Saving output...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    docs.to_csv(OUTPUT_CSV, index=False)
    print(f"  → {OUTPUT_CSV.relative_to(PHASE6_DIR)}  {len(docs):>10,} documents")


def print_summary(docs: pd.DataFrame, n_pages: int) -> None:
    tc = docs["token_count"]
    n_docs = len(docs)
    gt5000 = int((tc > 5000).sum())

    def fmt_stat(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:>8.1f}"

    def fmt_max(series: pd.Series) -> str:
        if series.empty:
            return "n/a"
        return f"{int(series.max()):>8,}"

    def fmt_pct(count: int) -> str:
        if n_docs == 0:
            return "n/a"
        return f"{count / n_docs:.1%}"

    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  Pages aggregated           : {n_pages:>8,}")
    print(f"  Documents produced         : {n_docs:>8,}")
    print(f"  Mean tokens per document   : {fmt_stat(tc.mean())}")
    print(f"  Median tokens per document : {fmt_stat(tc.median())}")
    print(f"  Documents with 0 tokens    : {int((tc == 0).sum()):>8,}")
    print(f"  Max tokens per document    : {fmt_max(tc)}")
    print(f"  Documents > 5,000 tokens   : {gt5000:>8,}  ({fmt_pct(gt5000)})")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEPARATOR)
    print("PHASE 6A — Document Aggregation")
    print(SEPARATOR)

    df = load_data()
    n_pages = len(df)
    docs = aggregate_documents(df)
    save_output(docs)
    print_summary(docs, n_pages)


if __name__ == "__main__":
    main()
