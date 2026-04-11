"""
Phase 1 — Structural Analysis of JFK Document Pages

Loads the merged JFK pages dataset, removes empty/null content rows,
validates that all pages from the 55 previously-missing files are present, computes
per-page structural metrics (character/word/line counts, text-composition
ratios, code-like line detection), sets heuristic classification flags,
and saves the enriched dataset.

Dependencies: pandas, tqdm, re, pathlib
"""

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Enable tqdm for pandas .apply() calls
tqdm.pandas()

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_DIR = SCRIPT_DIR.parent

INPUT_CSV = PHASE_DIR / "JFK_Pages_Merged.csv"
MISSING_CSV = PHASE_DIR / "jfk_categorization_55missing.csv"
OUTPUT_CSV = PHASE_DIR / "data" / "pages_phase1_structural.csv"

SEPARATOR = "=" * 60
CODE_LIKE_LINE_PATTERN = re.compile(r"^[A-Z]+/\d+(/[A-Z]+)?$")
MAIN_REQUIRED_COLUMNS = {"content", "file_id", "page_number"}
MISSING_REQUIRED_COLUMNS = {"file_id", "page_number"}


# ---------------------------------------------------------------------------
# Text-composition metric functions
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    """Print a consistently formatted section header."""
    print(f"\n{title}")


def validate_columns(
    df: pd.DataFrame, required_columns: set[str], source_name: str
) -> None:
    """Raise a clear error if an input DataFrame is missing required columns."""
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"{source_name} is missing required columns: {missing_list}")


def with_progress(series: pd.Series, func, *, desc: str) -> pd.Series:
    """Run a pandas apply with a labeled tqdm progress bar."""
    tqdm.pandas(desc=desc)
    return series.progress_apply(func)

def uppercase_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    if pd.isna(text):
        return 0.0
    text = str(text)
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    upper = sum(1 for c in alpha if c.isupper())
    return upper / len(alpha)


def numeric_ratio(text: str) -> float:
    """Fraction of whitespace-delimited tokens that are purely digits."""
    if pd.isna(text):
        return 0.0
    text = str(text)
    tokens = text.split()
    if not tokens:
        return 0.0
    numeric = sum(1 for t in tokens if t.isdigit())
    return numeric / len(tokens)


def short_token_ratio(text: str) -> float:
    """Fraction of tokens with length <= 2 (catches OCR noise / initials)."""
    if pd.isna(text):
        return 0.0
    text = str(text)
    tokens = text.split()
    if not tokens:
        return 0.0
    short = sum(1 for t in tokens if len(t) <= 2)
    return short / len(tokens)


def unique_word_ratio(text: str) -> float:
    """Type-token ratio — higher values suggest more varied vocabulary."""
    if pd.isna(text):
        return 0.0
    text = str(text)
    tokens = text.split()
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return unique / len(tokens)


def code_like_line_ratio(text: str) -> float:
    """Fraction of lines matching CIA routing-code patterns (e.g. DIR/1234/OPS)."""
    if pd.isna(text):
        return 0.0
    text = str(text)
    lines = text.split("\n")
    if not lines:
        return 0.0
    code_like = 0
    for line in lines:
        line = line.strip()
        if CODE_LIKE_LINE_PATTERN.match(line):
            code_like += 1
    return code_like / len(lines)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, int]:
    """Load the merged CSV and drop rows with empty or null content.

    Returns the filtered DataFrame and the count of removed rows.
    """
    print_section("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    validate_columns(df, MAIN_REQUIRED_COLUMNS, INPUT_CSV.name)
    total_before = len(df)
    print(f"  Input:  {INPUT_CSV.name:<35} {total_before:,} rows")

    # Remove pages where OCR produced no usable text
    df = df[df["content"].notna() & (df["content"].str.strip() != "")].copy()
    removed = total_before - len(df)
    print(f"  Kept:   {len(df):<35,} Removed {removed:,} empty/null rows")

    # Validate that every page from the 55-missing supplement is present
    missing_csv = pd.read_csv(MISSING_CSV)
    validate_columns(missing_csv, MISSING_REQUIRED_COLUMNS, MISSING_CSV.name)
    missing_keys = set(
        zip(
            missing_csv["file_id"].astype(str).str.strip(),
            missing_csv["page_number"].astype(str).str.strip(),
        )
    )
    merged_keys = set(
        zip(
            df["file_id"].astype(str).str.strip(),
            df["page_number"].astype(str).str.strip(),
        )
    )
    not_found = missing_keys - merged_keys

    n_files = missing_csv["file_id"].nunique()
    n_pages = len(missing_keys)

    if not_found:
        print(
            f"  WARNING: {len(not_found)} pages from "
            f"{MISSING_CSV.name} are NOT in the merged dataset:"
        )
        for key in sorted(not_found):
            print(f"    file_id={key[0]}, page_number={key[1]}")
    else:
        print(
            f"  Merged: {MISSING_CSV.name:<35} "
            f"{n_files} files, {n_pages:,} pages — all present"
        )

    return df, removed


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-page structural and text-composition metrics."""
    print_section("Computing metrics...")

    # Basic counts (vectorised — no progress bar needed)
    df["char_count"] = df["content"].str.len()
    df["word_count"] = df["content"].str.split().str.len()
    df["line_count"] = df["content"].str.count("\n") + 1

    # Row-by-row composition metrics (slow on 83k rows, so show progress)
    df["uppercase_ratio"] = with_progress(
        df["content"], uppercase_ratio, desc="Uppercase ratio"
    )
    df["numeric_ratio"] = with_progress(
        df["content"], numeric_ratio, desc="Numeric ratio"
    )
    df["short_token_ratio"] = with_progress(
        df["content"], short_token_ratio, desc="Short-token ratio"
    )
    df["unique_word_ratio"] = with_progress(
        df["content"], unique_word_ratio, desc="Unique-word ratio"
    )

    # Structure detection
    df["code_like_line_ratio"] = with_progress(
        df["content"], code_like_line_ratio, desc="Code-like line ratio"
    )

    return df


def compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Set heuristic boolean flags for page classification."""
    print_section("Computing flags...")

    # Keyword-based flags (vectorised regex search)
    df["contains_distribution_keyword"] = df["content"].str.lower().str.contains(
        r"\b(?:distribution|division|station|hqs)\b", na=False
    )
    df["contains_classification_keyword"] = df["content"].str.lower().str.contains(
        r"\b(?:secret|confidential|classified)\b", na=False
    )

    # Composite heuristic flags
    df["is_low_content_page"] = df["word_count"] < 40
    df["is_likely_distribution_page"] = (
        (df["code_like_line_ratio"] > 0.3) | df["contains_distribution_keyword"]
    )
    df["is_likely_cover_page"] = (df["word_count"] < 60) & (
        df["uppercase_ratio"] > 0.5
    )

    return df


def save_output(df: pd.DataFrame) -> None:
    """Write the enriched DataFrame to CSV."""
    print_section("Saving output...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  → {OUTPUT_CSV.relative_to(PHASE_DIR)}  {len(df):>10,} rows")


def print_summary(df: pd.DataFrame, removed: int) -> None:
    """Print a summary table of key counts."""
    total = len(df)
    low = df["is_low_content_page"].sum()
    cover = df["is_likely_cover_page"].sum()
    dist = df["is_likely_distribution_page"].sum()
    total_display = f"{total:,}"

    def format_ratio(count: int) -> str:
        if total == 0:
            return "n/a"
        return f"{count / total:.1%}"

    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  Pages analysed           : {total_display:>6}")
    print(f"  Empty rows removed       : {removed:>6,}")
    print(f"  Low-content pages        : {low:>6,}  ({format_ratio(low)})")
    print(f"  Likely cover pages       : {cover:>6,}  ({format_ratio(cover)})")
    print(f"  Likely distribution pages: {dist:>6,}  ({format_ratio(dist)})")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEPARATOR)
    print("PHASE 1 — Structural Analysis")
    print(SEPARATOR)

    df, removed = load_data()
    df = compute_metrics(df)
    df = compute_flags(df)
    save_output(df)
    print_summary(df, removed)


if __name__ == "__main__":
    main()
