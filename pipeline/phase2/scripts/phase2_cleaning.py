"""
Phase 2 — Boilerplate Removal, Dehyphenation, and OCR Normalisation
====================================================================
Input : phase1/data/pages_phase1_structural.csv
Output: phase2/data/pages_phase2_cleaned.csv

Columns added
-------------
  content_clean_boilerplate — content with standalone archive-noise lines removed
  content_clean_ocr         — after dehyphenation + light OCR normalisation
  word_count_clean          — word count of content_clean_ocr
  char_count_clean          — character count of content_clean_ocr

Rules
-----
- Original `content` column is preserved unchanged.
- Only standalone lines are removed (not sub-sentence occurrences).
- Dehyphenation joins line-end hyphens where OCR split a lower/mixed-case
  word across lines, while preserving deliberate compounds (CIA-FBI).
- OCR normalisation is conservative: lowercase, whitespace collapse,
  symbol-only / purely-numeric token removal.

Dependencies: pandas, tqdm, re, pathlib
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
PHASE2_DIR = SCRIPT_DIR.parent
PHASE1_DIR = PHASE2_DIR.parent / "phase1"

INPUT_CSV = PHASE1_DIR / "data" / "pages_phase1_structural.csv"
OUTPUT_CSV = PHASE2_DIR / "data" / "pages_phase2_cleaned.csv"

SEPARATOR = "=" * 60
REQUIRED_COLUMNS = {"content", "file_id", "word_count"}

# ---------------------------------------------------------------------------
# Boilerplate patterns  (matched against FULL stripped lines, case-insensitive)
# ---------------------------------------------------------------------------
# Each entry is a raw regex that must match the *entire* stripped line.
# Patterns are ordered from most-specific to most-general.

_BOILERPLATE_PATTERNS_RAW = [
    # -- FOIA / Declassification notices ---------------------------------------
    r"^\d{4}\s+release\s+under\s+the\s+president\s+john\s+f\.?\s*kennedy\s+assassination\s+records\s+act.*$",
    r"^release\s+under\s+the\s+president\s+john\s+f\.?\s*kennedy\s+assassination\s+records\s+act.*$",
    r"^for\s+foia\s+review$",
    r"^same\s+as\s+released$",
    r"^document\s+number.*$",
    r"^record\s+copy$",
    r"^all\s+information\s+contained$",
    r"^herein\s+is\s+unclassified.*$",

    # -- Archive record ID stamps ----------------------------------------------
    r"^\d{2}-0{5}$",                   # "14-00000", "13-00000"
    r"^nw\s+\d+\s+docl?d:?\s*\d+.*$",  # "NW 88613 DOCLD:32199554"

    # -- Filing / metadata markers ---------------------------------------------
    r"^prepare\s+for\s+filming$",
    r"^for\s+filing$",
    r"^index$",
    r"^abstract$",
    r"^cable\s+iden.*$",
    r"^doc\s*\d*$",

    # -- Classification stamps -------------------------------------------------
    r"^secret$",
    r"^top\s+secret$",
    r"^confidential$",
    r"^classified$",
    r"^classification$",
    r"^restricted$",
    r"^unclassified$",

    # -- Distribution headings -------------------------------------------------
    r"^field\s+distribution.*$",
    r"^hqs\s+distribution.*$",
    r"^distribution:.*$",
    r"^(?:\w+\s+)*division$",
    r"^station$",
]

BOILERPLATE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _BOILERPLATE_PATTERNS_RAW),
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Dehyphenation
# ---------------------------------------------------------------------------
# Matches a line ending with an alphabetic char + hyphen, followed by a newline
# and a next line starting with a lowercase letter. This intentionally joins
# lower/mixed-case OCR splits while leaving all-caps continuations unchanged.
# It also avoids deliberate compounds (which do not straddle lines) and
# archival codes like "104-10009" (digits before the hyphen).
_DEHYPHEN_RE = re.compile(r"([A-Za-z])-\n([a-z])")


def dehyphenate(text: str) -> tuple[str, int]:
    """Join line-end hyphens that look like OCR-broken words.

    Returns the fixed text and the number of joins performed.
    """
    if not isinstance(text, str):
        return ("", 0)
    count = len(_DEHYPHEN_RE.findall(text))
    fixed = _DEHYPHEN_RE.sub(r"\1\2", text)
    return fixed, count


# ---------------------------------------------------------------------------
# OCR-normalisation helpers
# ---------------------------------------------------------------------------
# Tokens that are purely non-alphanumeric symbols
_SYMBOL_ONLY_RE = re.compile(r"^[^\w]+$")

# Tokens that are purely numeric (digits with possible separators)
_PURE_NUMERIC_RE = re.compile(r"^\d[\d.,/-]*$")


def _is_removable_token(tok: str) -> bool:
    """Return True if `tok` should be dropped during OCR normalisation."""
    if len(tok) == 0:
        return True
    if len(tok) == 1:
        # Keep single alphabetic chars (likely abbreviations); drop punctuation
        return not tok.isalpha()
    if _SYMBOL_ONLY_RE.match(tok):
        return True
    if _PURE_NUMERIC_RE.match(tok):
        return True
    return False


def normalize_ocr(text: str) -> str:
    """Apply light OCR normalisation while preserving line-break structure.

    Per line: lowercase → collapse whitespace → drop symbol-only / numeric /
    single-punctuation tokens → rejoin.
    """
    if not isinstance(text, str) or text.strip() == "":
        return text if isinstance(text, str) else ""

    output_lines = []
    for line in text.splitlines():
        line_lower = re.sub(r"[ \t]+", " ", line.lower()).strip()
        if not line_lower:
            output_lines.append("")
            continue
        tokens = line_lower.split(" ")
        kept = [t for t in tokens if not _is_removable_token(t)]
        output_lines.append(" ".join(kept))

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# Boilerplate removal
# ---------------------------------------------------------------------------

def remove_boilerplate_lines(text: str) -> tuple[str, int]:
    """Remove lines that are entirely archive boilerplate.

    Returns (cleaned_text, lines_removed).
    """
    if not isinstance(text, str) or text.strip() == "":
        return (text if isinstance(text, str) else ""), 0

    kept_lines = []
    lines_removed = 0

    for line in text.splitlines():
        stripped = line.strip()
        if stripped and BOILERPLATE_RE.fullmatch(stripped):
            lines_removed += 1
        else:
            kept_lines.append(line)

    return "\n".join(kept_lines), lines_removed


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


def run_boilerplate_removal(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Removing boilerplate...")
    results = with_progress(
        df["content"], remove_boilerplate_lines, desc="Boilerplate removal"
    )
    unpacked = pd.DataFrame(
        results.tolist(),
        index=df.index,
        columns=["content_clean_boilerplate", "_lines_removed"],
    )
    df[["content_clean_boilerplate", "_lines_removed"]] = unpacked
    return df


def run_dehyphenation(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Dehyphenating OCR-broken words...")
    results = with_progress(
        df["content_clean_boilerplate"], dehyphenate, desc="Dehyphenation"
    )
    unpacked = pd.DataFrame(
        results.tolist(),
        index=df.index,
        columns=["content_clean_boilerplate", "_dehyphen_joins"],
    )
    df[["content_clean_boilerplate", "_dehyphen_joins"]] = unpacked
    return df


def run_ocr_normalisation(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Normalising text...")
    df["content_clean_ocr"] = with_progress(
        df["content_clean_boilerplate"], normalize_ocr, desc="OCR normalisation"
    )
    return df


def compute_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    clean_text = df["content_clean_ocr"].fillna("")
    df["word_count_clean"] = clean_text.str.split().str.len()
    df["char_count_clean"] = clean_text.str.len()
    df["_word_frac_removed"] = (
        (df["word_count"] - df["word_count_clean"])
        .clip(lower=0)
        .div(df["word_count"].replace(0, np.nan))
        .fillna(0)
    )
    return df


def save_output(df: pd.DataFrame) -> None:
    print_section("Saving output...")
    save_cols = [c for c in df.columns if not c.startswith("_")]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[save_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"  → {OUTPUT_CSV.relative_to(PHASE2_DIR)}  {len(df):>10,} rows")


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of cleaning effects."""
    total = len(df)
    lines_removed = int(df["_lines_removed"].sum())
    dehyphen_joins = int(df["_dehyphen_joins"].sum())
    pages_with_removal = int((df["_lines_removed"] > 0).sum())
    pages_unaffected = total - pages_with_removal
    avg_wc_before = df["word_count"].mean()
    avg_wc_after = df["word_count_clean"].mean()
    pct_change = 0.0
    if pd.notna(avg_wc_before) and avg_wc_before != 0:
        pct_change = (avg_wc_after - avg_wc_before) / avg_wc_before * 100

    def fmt_pct(count: int) -> str:
        if total == 0:
            return "n/a"
        return f"{count / total:.1%}"

    def fmt_mean(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:>8.1f}"

    pct_change_display = "n/a"
    if pd.notna(avg_wc_before) and avg_wc_before != 0 and pd.notna(avg_wc_after):
        pct_change_display = f"{pct_change:+.1f}%"

    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  Pages analysed            : {total:>8,}")
    print(f"  Boilerplate lines removed : {lines_removed:>8,}")
    print(f"  Dehyphenation joins       : {dehyphen_joins:>8,}")
    print(
        f"  Pages with any removal    : {pages_with_removal:>8,}  "
        f"({fmt_pct(pages_with_removal)})"
    )
    print(
        f"  Pages unaffected          : {pages_unaffected:>8,}  "
        f"({fmt_pct(pages_unaffected)})"
    )
    print(f"  Mean word count before    : {fmt_mean(avg_wc_before)}")
    print(
        f"  Mean word count after     : {fmt_mean(avg_wc_after)}  "
        f"({pct_change_display})"
    )
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEPARATOR)
    print("PHASE 2 — Boilerplate Removal and OCR Normalisation")
    print(SEPARATOR)

    df = load_data()
    df = run_boilerplate_removal(df)
    df = run_dehyphenation(df)
    df = run_ocr_normalisation(df)
    df = compute_diagnostics(df)
    save_output(df)
    print_summary(df)


if __name__ == "__main__":
    main()
