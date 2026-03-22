"""
Phase 3: Conservative Line-Level Metadata Filtering
====================================================
Input:  phase 2/data/pages_phase2_cleaned.csv
Output: phase 3/data/pages_phase3_linefiltered.csv

Goal: Remove lines that are routing metadata, office codes, distribution
headings, or other non-substantive archive lines, while keeping all
narrative historical content.  When in doubt, keep the line.
"""

import re
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "..", "phase 2", "data", "pages_phase2_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pages_phase3_linefiltered.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    # FBI memo routing label block (appears on thousands of FBI cover sheets)
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

# D) Lines dominated by slashes or very short tokens (code-dominated)
#    We tokenize and check what fraction looks like ordinary English words
RE_WORD = re.compile(r"[a-z]{3,}")          # a "real" word: ≥3 letters, no digits


def _line_is_code_like(line: str) -> bool:
    """
    Return True when a line looks like a pure routing/office code token,
    e.g.  'c/sr/ci/r'  'lp/edi'  'af/1'  'wh/3/b'
    The line must:
      - be ≤ 30 characters
      - contain at least one slash
      - consist only of letters, digits, slashes (no normal sentence chars)
    """
    if len(line) > 30:
        return False
    if '/' not in line:
        return False
    # Allow only letters, digits, slashes, hyphens, and brackets
    if re.fullmatch(r'[a-z0-9/\-\[\]\.]{2,30}', line):
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
    # Also allow trailing parentheticals / numbers: "nfd (8)" → "nfd"
    core = re.sub(r'[\s\(\)\[\]0-9,]+$', '', line).strip()
    return core in STANDALONE_KEYWORDS


def _line_is_low_content(line: str) -> bool:
    """
    Return True for lines with very low natural-language content.
    Criteria (ALL must hold to be considered low-content):
      - Line is short (≤ 35 chars after strip)
      - Fewer than 2 'real' words (≥3 consecutive letters)
      - The line is not empty (empty lines are kept as-is)
    """
    if not line:
        return False   # keep empty lines (they're structural)
    if len(line) > 35:
        return False   # longer lines are likely narrative
    real_words = RE_WORD.findall(line)
    if len(real_words) >= 2:
        return False   # has at least two word-like tokens → keep
    # Has 0 or 1 real word: check that the rest is codes/digits/punct
    non_word = re.sub(r'[a-z]{3,}', '', line).strip()
    if len(non_word) > len(line) * 0.6:
        return True
    return False


def _line_is_filing_action(line: str) -> bool:
    """
    Return True for known CIA filing / action code lines that were part of
    the routing slip appended to cables and memos.
    Examples: 'travel program', 'prepare for ficking', 'code no. (2,3,4)',
              'rybat rest code', 'iso/dcu]', 'lp/edi abstract', 'ip/fis',
              'piph', 'nfd (8)', 'ip/files', 'for filing'
    """
    FILING_PHRASES = [
        "travel program",
        "prepare for",          # "prepare for filing", "prepare for ficking" (OCR of filing)
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
    for phrase in FILING_PHRASES:
        if line.startswith(phrase):
            return True
    return False


# ---------------------------------------------------------------------------
# Main line-filter function
# ---------------------------------------------------------------------------

def filter_lines(text: str) -> str:
    """
    Apply all metadata-removal rules line by line.
    Returns the filtered text with the same line structure.
    """
    if not isinstance(text, str):
        return text   # preserve NaN / non-string values unchanged

    result_lines = []
    for raw_line in text.split('\n'):
        line = raw_line.strip().lower()   # all rule-matching is case-insensitive

        # --- Apply rules in order of confidence (most-certain first) ---

        # Rule 1: pure routing/office code  (e.g. c/sr/ci/r, wh/3/b)
        if _line_is_code_like(line):
            continue

        # Rule 2: known metadata phrase (distribution headers etc.)
        if _line_is_metadata_phrase(line):
            continue

        # Rule 3: single standalone admin keyword
        if _line_is_standalone_keyword(line):
            continue

        # Rule 4: known CIA filing/action code line
        if _line_is_filing_action(line):
            continue

        # Rule 5: very low natural-language content (short & code-dominated)
        if _line_is_low_content(line):
            continue

        # If none of the rules fired → keep the line
        result_lines.append(raw_line)

    return '\n'.join(result_lines)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def count_non_empty_lines(text) -> int:
    """Count non-empty lines in a text value; return 0 for NaN."""
    if not isinstance(text, str):
        return 0
    return sum(1 for ln in text.split('\n') if ln.strip())


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading Phase 2 data …")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

# ---------------------------------------------------------------------------
# Apply line filtering
# ---------------------------------------------------------------------------

print("Filtering lines …")
df['content_clean_lines'] = df['content_clean_ocr'].apply(filter_lines)

# ---------------------------------------------------------------------------
# Diagnostic columns
# ---------------------------------------------------------------------------

df['lines_before']        = df['content_clean_ocr'].apply(count_non_empty_lines)
df['lines_after']         = df['content_clean_lines'].apply(count_non_empty_lines)
df['lines_removed']       = df['lines_before'] - df['lines_after']
df['line_removal_ratio']  = np.where(
    df['lines_before'] > 0,
    df['lines_removed'] / df['lines_before'],
    0.0
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

print(f"Saving to {OUTPUT_PATH} …")
df.to_csv(OUTPUT_PATH, index=False)
print("  Done.")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

total_pages = len(df)
pages_with_text = df[df['lines_before'] > 0]
heavy_removal   = df[df['line_removal_ratio'] > 0.30]

print("\n" + "=" * 60)
print("PHASE 3  LINE-FILTERING  SUMMARY")
print("=" * 60)
print(f"Total pages processed          : {total_pages:,}")
print(f"Pages with non-empty OCR text  : {len(pages_with_text):,}")
print(f"Avg lines_before               : {pages_with_text['lines_before'].mean():.1f}")
print(f"Avg lines_after                : {pages_with_text['lines_after'].mean():.1f}")
print(f"Avg lines_removed              : {pages_with_text['lines_removed'].mean():.1f}")
print(f"Avg line_removal_ratio         : {pages_with_text['line_removal_ratio'].mean():.3f}")
print(f"Pages with >30 % lines removed : {len(heavy_removal):,}  "
      f"({100*len(heavy_removal)/total_pages:.1f} % of all pages)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Before / After examples  (10 pages with most lines removed)
# ---------------------------------------------------------------------------

print("\n10 BEFORE / AFTER EXAMPLES  (pages with most lines removed)\n")
sample = (
    pages_with_text[pages_with_text['lines_removed'] > 0]
    .nlargest(10, 'lines_removed')
)

for rank, (_, row) in enumerate(sample.iterrows(), start=1):
    before_preview = (row['content_clean_ocr'] or '')[:600].replace('\n', ' ↵ ')
    after_preview  = (row['content_clean_lines'] or '')[:600].replace('\n', ' ↵ ')
    print(f"[{rank}] file_id={row['file_id']}  page={row['page_number']}  "
          f"removed={row['lines_removed']} lines  "
          f"ratio={row['line_removal_ratio']:.2f}")
    print(f"  BEFORE: {before_preview}")
    print(f"  AFTER : {after_preview}")
    print()
