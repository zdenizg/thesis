"""
Phase 3 — Conservative Line-Level Metadata Filtering
=====================================================
Input:  phase2/data/pages_phase2_cleaned.csv
Output: phase3/data/pages_phase3_linefiltered.csv

Goal: Remove lines that are routing metadata, office codes, distribution
headings, or other non-substantive archive lines, while keeping all
narrative historical content.  When in doubt, keep the line.

Dependencies: pandas, numpy, tqdm, re, pathlib
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
PHASE2_DIR = PHASE3_DIR.parent / "phase2"

INPUT_CSV = PHASE2_DIR / "data" / "pages_phase2_cleaned.csv"
OUTPUT_CSV = PHASE3_DIR / "data" / "pages_phase3_linefiltered.csv"

SEPARATOR = "=" * 60
REQUIRED_COLUMNS = {"content_clean_ocr", "file_id"}

# ---------------------------------------------------------------------------
# Compiled patterns  (all matching is done on the *stripped* lowercase line)
# ---------------------------------------------------------------------------

# A) Known distribution / routing header phrases (exact or prefix match)
METADATA_PHRASES = [
    "field distribution",
    "field dissem",
    "hqs distribution",
    "hdqrs dissem",
    "headquarters distribution",
    "liaison dissem",
    "record copy",
    "routing slip",
    "cable distribution",
    "eyes only",
    "no foreign dissem",
    "no foreign distribution",
    "noforn",
    "dissemination controls",
    "handle via",
    "controlled dissem",
    "destroy after",
    "do not reproduce",
    "not releasable",
]

# B) Lines that are just a single known routing/admin keyword
STANDALONE_KEYWORDS = {
    "index", "abstract", "film", "cable", "dissem",
    "distribution", "division", "station", "rybat",
    "pi/files", "ip/files", "ip/fis", "nfd",
    "piph", "limdis", "exdis", "nodis",
    "priority", "immediate", "routine", "flash",
    "unclassified", "confidential", "secret", "top secret",
    "sanitized", "redacted",
    # FBI memo routing label block
    "assoc. dir.", "ext. affairs", "telephone rm.", "asst. dir.:",
    "admin.", "comp. syst.", "spec. inv.", "gen. inv.",
    "files com.", "legal coun.", "ident.", "laboratory",
    "training", "inspection",
    # Form field headers
    "position title", "date of birth", "country", "remarks",
    "specific duty no.", "(when filled in)", "classified message",
    # Archive stamp variant missed by Phase 2
    "nw docid: page",
}

# C) Known CIA filing / action code phrases (prefix match)
FILING_PHRASES = [
    "travel program",
    "prepare for",
    "code no.",
    "rybat",
    "iso/dcu",
    "lp/edi",
    "ip/fis",
    "ip/files",
    "pi/files",
    "for filing",
    "for foia review",
    "same as released",
    "document number",
    "foia review",
]

# D) A "real" word: >= 3 letters, no digits
RE_WORD = re.compile(r"[a-z]{3,}")
RE_CODE_LIKE = re.compile(r"[a-z0-9/\-\[\]\.]{2,30}")
NON_EMPTY_LINE_PATTERN = r"(?m)^\s*\S.*$"


# ---------------------------------------------------------------------------
# Line-classification functions (exported for use by discovery script)
# ---------------------------------------------------------------------------

def _line_is_code_like(line: str) -> bool:
    """Return True for pure routing/office code tokens (e.g. c/sr/ci/r, wh/3/b).

    The line must be <= 30 chars, contain a slash, and consist only of
    letters, digits, slashes, hyphens, brackets, and dots.
    """
    if len(line) > 30:
        return False
    if '/' not in line:
        return False
    if RE_CODE_LIKE.fullmatch(line):
        return True
    return False


def _line_is_metadata_phrase(line: str) -> bool:
    """Return True when the line starts with or equals a known metadata phrase."""
    for phrase in METADATA_PHRASES:
        if line == phrase or line.startswith(phrase):
            return True
    return False


def _line_is_standalone_keyword(line: str) -> bool:
    """Return True when the stripped line is exactly one of the admin keywords."""
    # Also allow trailing parentheticals / numbers: "nfd (8)" -> "nfd"
    core = re.sub(r'[\s\(\)\[\]0-9,]+$', '', line).strip()
    return core in STANDALONE_KEYWORDS


def _line_is_low_content(line: str) -> bool:
    """Return True for lines with very low natural-language content.

    Criteria (ALL must hold):
      - Line is short (<= 35 chars after strip)
      - Fewer than 2 'real' words (>= 3 consecutive letters)
      - The line is not empty (empty lines are kept as structural separators)
    """
    if not line:
        return False
    if len(line) > 35:
        return False
    real_words = RE_WORD.findall(line)
    if len(real_words) >= 2:
        return False
    # Has 0 or 1 real word: check that the rest is codes/digits/punct
    non_word = re.sub(r'[a-z]{3,}', '', line).strip()
    if len(non_word) > len(line) * 0.6:
        return True
    return False


def _line_is_filing_action(line: str) -> bool:
    """Return True for known CIA filing / action code lines."""
    for phrase in FILING_PHRASES:
        if line.startswith(phrase):
            return True
    return False


# ---------------------------------------------------------------------------
# Main line-filter function
# ---------------------------------------------------------------------------

def filter_lines(text: str) -> str:
    """Apply all metadata-removal rules line by line.

    Returns the filtered text with the same line structure.
    """
    if not isinstance(text, str):
        return text

    result_lines = []
    for raw_line in text.split('\n'):
        line = raw_line.strip().lower()

        # Apply rules in order of confidence (most-certain first)
        if _line_is_code_like(line):
            continue
        if _line_is_metadata_phrase(line):
            continue
        if _line_is_standalone_keyword(line):
            continue
        if _line_is_filing_action(line):
            continue
        if _line_is_low_content(line):
            continue

        result_lines.append(raw_line)

    return '\n'.join(result_lines)


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


def with_progress(series: pd.Series, func, *, desc: str) -> pd.Series:
    """Run a pandas apply with a labeled tqdm progress bar."""
    tqdm.pandas(desc=desc)
    return series.progress_apply(func)


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


def run_line_filtering(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Filtering lines...")
    df['content_clean_lines'] = with_progress(
        df['content_clean_ocr'], filter_lines, desc="Line filtering"
    )
    return df


def compute_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    clean_text = df['content_clean_ocr'].fillna('')
    filtered_text = df['content_clean_lines'].fillna('')

    df['_lines_before'] = clean_text.str.count(NON_EMPTY_LINE_PATTERN)
    df['_lines_after'] = filtered_text.str.count(NON_EMPTY_LINE_PATTERN)
    df['_lines_removed'] = df['_lines_before'] - df['_lines_after']
    df['_line_removal_ratio'] = np.where(
        df['_lines_before'] > 0,
        df['_lines_removed'] / df['_lines_before'],
        0.0
    )
    # Character-level retention
    clean_len = clean_text.str.len().replace(0, np.nan)
    filtered_len = filtered_text.str.len()
    df['_char_retention'] = (filtered_len / clean_len).fillna(1.0)
    return df


def save_output(df: pd.DataFrame) -> None:
    print_section("Saving output...")
    # Keep diagnostic columns in output (without _ prefix) for downstream validation
    df = df.rename(columns={
        '_lines_before': 'lines_before',
        '_lines_after': 'lines_after',
        '_lines_removed': 'lines_removed',
        '_line_removal_ratio': 'line_removal_ratio',
    })
    # Drop internal-only columns
    save_cols = [c for c in df.columns if not c.startswith("_")]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[save_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"  → {OUTPUT_CSV.relative_to(PHASE3_DIR)}  {len(df):>10,} rows")


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of line-filtering effects."""
    total = len(df)
    has_text = df[df['_lines_before'] > 0]

    mean_ratio = has_text['_line_removal_ratio'].mean()
    median_ratio = has_text['_line_removal_ratio'].median()
    no_removal = int((df['_lines_removed'] == 0).sum())
    gt50 = int((df['_line_removal_ratio'] > 0.50).sum())
    char_retention = has_text['_char_retention'].mean()

    def fmt_pct(count: int) -> str:
        if total == 0:
            return "n/a"
        return f"{count / total:.1%}"

    def fmt_ratio(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:>7.1%}"

    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  Pages analysed          : {total:>8,}")
    print(f"  Mean line removal ratio : {fmt_ratio(mean_ratio)}")
    print(f"  Median line removal ratio: {fmt_ratio(median_ratio)}")
    print(f"  Pages with no removal    : {no_removal:>8,}  ({fmt_pct(no_removal)})")
    print(f"  Pages > 50% removed      : {gt50:>8,}  ({fmt_pct(gt50)})")
    print(f"  Char retention rate      : {fmt_ratio(char_retention)}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEPARATOR)
    print("PHASE 3 — Line-level Metadata Filtering")
    print(SEPARATOR)

    df = load_data()
    df = run_line_filtering(df)
    df = compute_diagnostics(df)
    save_output(df)
    print_summary(df)


if __name__ == "__main__":
    main()
