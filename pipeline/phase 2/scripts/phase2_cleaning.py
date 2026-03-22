"""
Phase 2: Archive-Specific Boilerplate Removal and Light OCR Normalization
=========================================================================
Input : ../data/pages_phase1_structural.csv  (symlinked / copied from Phase 1 output)
Output: ../data/pages_phase2_cleaned.csv

Columns added
-------------
  content_clean_boilerplate – content with standalone archive-noise lines removed
  content_clean_ocr         – content_clean_boilerplate with light OCR normalization
  word_count_clean          – word count of content_clean_ocr
  char_count_clean          – character count of content_clean_ocr

Rules
-----
- Original `content` column is preserved unchanged.
- Only standalone lines are removed (not sub-sentence occurrences).
- OCR normalization is conservative: lowercase, whitespace, symbol-only tokens.
  Names, dates, locations, and intelligence acronyms are kept intact by design
  (they survive tokenisation unless they are purely numeric or single-char).
"""

import re
import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this script's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PHASE2_ROOT = os.path.dirname(SCRIPT_DIR)
PHASE1_ROOT = os.path.join(os.path.dirname(PHASE2_ROOT), "phase 1 ")

INPUT_PATH  = os.path.join(PHASE1_ROOT, "data", "pages_phase1_structural.csv")
OUTPUT_PATH = os.path.join(PHASE2_ROOT, "data", "pages_phase2_cleaned.csv")

# ---------------------------------------------------------------------------
# Boilerplate patterns  (matched against FULL stripped lines, case-insensitive)
# ---------------------------------------------------------------------------
# Each entry is a raw regex that must match the *entire* stripped line.
# Patterns are ordered from most-specific to most-general to avoid accidental
# over-removal.

_BOILERPLATE_PATTERNS_RAW = [

    # -- FOIA / Declassification notices -------------------------------------
    # e.g. "2025 RELEASE UNDER THE PRESIDENT JOHN F. KENNEDY ASSASSINATION RECORDS ACT OF 1992"
    r"^\d{4}\s+release\s+under\s+the\s+president\s+john\s+f\.?\s*kennedy\s+assassination\s+records\s+act.*$",
    r"^release\s+under\s+the\s+president\s+john\s+f\.?\s*kennedy\s+assassination\s+records\s+act.*$",
    r"^for\s+foia\s+review$",
    r"^same\s+as\s+released$",
    r"^document\s+number.*$",
    r"^record\s+copy$",
    # Declassification notice split across lines: "ALL INFORMATION CONTAINED" / "HEREIN IS UNCLASSIFIED"
    r"^all\s+information\s+contained$",
    r"^herein\s+is\s+unclassified.*$",

    # -- Archive record ID stamps --------------------------------------------
    # e.g. "14-00000", "13-00000" — appear on virtually every page
    r"^\d{2}-0{5}$",
    # Document tracking stamps: e.g. "NW 88613 DOCLD:32199554"
    r"^nw\s+\d+\s+docl?d:?\s*\d+.*$",

    # -- Filing / metadata markers -------------------------------------------
    r"^prepare\s+for\s+filming$",
    r"^for\s+filing$",
    r"^index$",
    r"^abstract$",
    r"^cable\s+iden.*$",
    r"^doc\s*\d*$",          # bare "doc" or "doc 12" as the entire line

    # -- Classification stamps -----------------------------------------------
    r"^secret$",
    r"^top\s+secret$",
    r"^confidential$",
    r"^classified$",
    r"^classification$",     # standalone form field header
    r"^restricted$",
    r"^unclassified$",

    # -- Distribution headings -----------------------------------------------
    # e.g. "FIELD DISTRIBUTION - BD #5847"
    r"^field\s+distribution.*$",
    r"^hqs\s+distribution.*$",
    r"^distribution:.*$",
    # e.g. "AF DIVISION", "EUR DIVISION", "FE DIVISION", or bare "DIVISION"
    r"^(?:\w+\s+)*division$",
    r"^station$",
]

# Compile once at module load
BOILERPLATE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _BOILERPLATE_PATTERNS_RAW),
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# OCR-normalisation helpers
# ---------------------------------------------------------------------------

# Tokens that are purely non-alphanumeric symbols (no letters or digits at all)
_SYMBOL_ONLY_RE = re.compile(r"^[^\w]+$")

# Tokens that are purely numeric (digits, possible separators)
_PURE_NUMERIC_RE = re.compile(r"^\d[\d\s.,/-]*$")


def _is_removable_token(tok: str) -> bool:
    """Return True if `tok` should be dropped during OCR normalisation."""
    if len(tok) == 0:
        return True
    if len(tok) == 1:
        # keep single letters that are likely abbreviations embedded in text;
        # drop isolated punctuation / symbols
        return not tok.isalpha()
    if _SYMBOL_ONLY_RE.match(tok):
        return True
    if _PURE_NUMERIC_RE.match(tok):
        return True
    return False


def normalize_ocr(text: str) -> str:
    """
    Apply light OCR normalisation to a page text while preserving line breaks.

    Steps (per line):
      1. Lowercase.
      2. Collapse repeated whitespace within the line to a single space.
      3. Tokenise on whitespace.
      4. Drop tokens that are: purely symbolic, purely numeric, or length-1
         non-alpha characters.
      5. Rejoin tokens; skip blank lines but keep the line-break structure.
    """
    if not isinstance(text, str) or text.strip() == "":
        return text if isinstance(text, str) else ""

    output_lines = []
    for line in text.splitlines():
        # Step 1-2: lowercase + collapse whitespace
        line_lower = re.sub(r"[ \t]+", " ", line.lower()).strip()
        if not line_lower:
            # preserve blank lines to keep page structure
            output_lines.append("")
            continue
        # Step 3-4: filter tokens
        tokens = line_lower.split(" ")
        kept = [t for t in tokens if not _is_removable_token(t)]
        output_lines.append(" ".join(kept))

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# Boilerplate removal
# ---------------------------------------------------------------------------

def remove_boilerplate_lines(text: str) -> tuple[str, int]:
    """
    Remove lines from `text` that are entirely archive boilerplate.

    Returns
    -------
    cleaned_text : str
    lines_removed : int
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
# Main processing
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 2: Boilerplate Removal & OCR Normalisation")
    print("=" * 60)

    # -- 1. Load dataset -------------------------------------------------------
    print(f"\nLoading: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        sys.exit(f"ERROR: Input file not found:\n  {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    total_pages = len(df)
    print(f"  Loaded {total_pages:,} rows  |  {df['file_id'].nunique():,} documents")

    # -- 2. Boilerplate removal -----------------------------------------------
    print("\nRemoving boilerplate lines …")
    results = df["content"].apply(remove_boilerplate_lines)
    df["content_clean_boilerplate"] = results.apply(lambda x: x[0])
    df["_lines_removed"]            = results.apply(lambda x: x[1])

    # -- 4. OCR normalisation -------------------------------------------------
    print("Applying OCR normalisation …")
    df["content_clean_ocr"] = df["content_clean_boilerplate"].apply(normalize_ocr)

    # -- 5. Diagnostic metrics ------------------------------------------------
    print("Computing diagnostic metrics …")

    def word_count(text):
        if not isinstance(text, str):
            return 0
        return len(text.split())

    def char_count(text):
        if not isinstance(text, str):
            return 0
        return len(text)

    df["word_count_clean"] = df["content_clean_ocr"].apply(word_count)
    df["char_count_clean"] = df["content_clean_ocr"].apply(char_count)

    # Fraction of words removed by the full cleaning pipeline
    df["_word_frac_removed"] = (
        (df["word_count"] - df["word_count_clean"])
        .clip(lower=0)
        .div(df["word_count"].replace(0, np.nan))
        .fillna(0)
    )

    # -- 6. Save output -------------------------------------------------------
    # Drop internal helper columns before saving
    save_cols = [c for c in df.columns if not c.startswith("_")]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df[save_cols].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")

    # -- 7. Summary statistics ------------------------------------------------
    total_lines_removed   = int(df["_lines_removed"].sum())
    pages_heavy_clean     = int((df["_word_frac_removed"] > 0.20).sum())
    avg_wc_before         = df["word_count"].mean()
    avg_wc_after          = df["word_count_clean"].mean()

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"  Total pages processed          : {total_pages:>10,}")
    print(f"  Avg word count  (before clean) : {avg_wc_before:>10.1f}")
    print(f"  Avg word count  (after  clean) : {avg_wc_after:>10.1f}")
    print(f"  Total boilerplate lines removed: {total_lines_removed:>10,}")
    print(f"  Pages where >20% text removed  : {pages_heavy_clean:>10,}  "
          f"({100*pages_heavy_clean/total_pages:.1f}%)")

    # -- 8. Example comparisons -----------------------------------------------
    print("\n" + "=" * 60)
    print("EXAMPLE PAGE COMPARISONS  (5 random pages with content)")
    print("=" * 60)

    # Choose pages that actually had something removed to make the comparison useful
    candidates = df[
        (df["_lines_removed"] > 0) &
        df["content"].notna()
    ]
    if len(candidates) < 5:
        candidates = df[df["content"].notna()]

    sample = candidates.sample(n=min(5, len(candidates)), random_state=42)

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        raw_preview   = str(row["content"])[:500].replace("\n", "↵ ")
        clean_preview = str(row["content_clean_ocr"])[:500].replace("\n", "↵ ")
        print(f"\n--- Example {i} ---")
        print(f"  file_id    : {row['file_id']}")
        print(f"  page_number: {row['page_number']}")
        print(f"  lines removed (boilerplate): {int(row['_lines_removed'])}")
        print(f"  words before: {int(row['word_count'])}   words after: {int(row['word_count_clean'])}")
        print(f"\n  [ORIGINAL  ] {raw_preview}")
        print(f"\n  [CLEANED   ] {clean_preview}")

    print("\nDone.")


if __name__ == "__main__":
    main()
