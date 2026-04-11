"""
Phase 4 — Tokenisation, Stopword Removal and Lemmatisation
===========================================================
Input:  phase3/data/pages_phase3_linefiltered.csv
Output: phase4/data/pages_phase4_modeltext.csv

Columns added
-------------
  content_model_no_lemma   — stopword-filtered tokens (no lemmatisation)
  content_model_lemma      — stopword-filtered + lemmatised tokens
  token_count_model_no_lemma
  token_count_model_lemma

Rules
-----
- Tokens must be >= 2 characters, contain no digits, and not be pure punctuation.
- Stopwords = NLTK English + archive-specific terms.
- Lemmatised output applies a second stopword pass to catch lemma→stopword mappings
  (e.g. "records" → "record" which is an archive stopword).
- Cold War anchor terms (cuba, soviet, mexico, embassy, oswald, cia, fbi,
  surveillance) are verified NOT to be in any stopword list.

Dependencies: pandas, numpy, tqdm, nltk, re, pathlib
"""

import re
import ssl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Fix SSL for NLTK downloads on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    """Download an NLTK resource only when it is not already available."""
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


ensure_nltk_resource("corpora/stopwords", "stopwords")
ensure_nltk_resource("corpora/wordnet", "wordnet")
ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

tqdm.pandas()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE4_DIR = SCRIPT_DIR.parent
PHASE3_DIR = PHASE4_DIR.parent / "phase3"

INPUT_CSV = PHASE3_DIR / "data" / "pages_phase3_linefiltered.csv"
OUTPUT_CSV = PHASE4_DIR / "data" / "pages_phase4_modeltext.csv"

SEPARATOR = "=" * 60
REQUIRED_COLUMNS = {"content_clean_lines", "file_id"}

# ---------------------------------------------------------------------------
# Stopword lists
# ---------------------------------------------------------------------------
ENGLISH_STOPWORDS = set(stopwords.words('english'))

ARCHIVE_STOPWORDS = {
    "record", "document", "copy", "review", "released", "act",
    "page", "division", "station", "office", "memo", "cable",
    "information", "attached",
    # Frequent administrative tokens with low topical value
    "mr", "mr.", "date", "number", "section",
    # High-frequency generic tokens with no topical signal
    "would", "may", "one", "also", "use", "made", "new",
    # Memo field header — n=42,891, dominates as header not narrative
    "subject",
    # Reporting verbs (zero topical signal)
    "said", "stated", "advised",
    # Modal verbs not in NLTK stopwords
    "could", "must",
    # Document type / template words
    "memorandum", "following", "concerning", "see", "type",
    # Spanish function word in intercepted docs
    "que",
    # Possessive marker produced by Penn Treebank tokenizer
    "'s",
}

ALL_STOPWORDS = ENGLISH_STOPWORDS | ARCHIVE_STOPWORDS

# Cold War anchor terms that must NOT be in any stopword list
COLD_WAR_ANCHORS = [
    "cuba", "soviet", "mexico", "embassy",
    "oswald", "cia", "fbi", "surveillance",
]

# ---------------------------------------------------------------------------
# Token filter
# ---------------------------------------------------------------------------
PUNCT_RE = re.compile(r'^[^\w]+$')

_lemmatizer = WordNetLemmatizer()


def is_valid_token(token: str) -> bool:
    """Return True if token should be kept (>= 2 chars, no digits, not pure punct)."""
    if len(token) < 2:
        return False
    if any(ch.isdigit() for ch in token):
        return False
    if PUNCT_RE.match(token):
        return False
    return True


def _tokenize_clean(text: str) -> list[str]:
    """Lowercase, tokenize, remove invalid tokens only (no stopwords)."""
    if not isinstance(text, str) or not text.strip():
        return []
    return [t for t in word_tokenize(text.lower()) if is_valid_token(t)]


def tokenize_and_filter(text: str) -> list[str]:
    """Lowercase, tokenize, remove invalid tokens and stopwords."""
    return [t for t in _tokenize_clean(text) if t not in ALL_STOPWORDS]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatise a token list without any additional filtering."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


def lemmatize_and_filter(tokens: list[str]) -> list[str]:
    """Lemmatise tokens, then apply the second stopword pass."""
    return [tok for tok in lemmatize_tokens(tokens) if tok not in ALL_STOPWORDS]


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


def check_anchor_terms() -> list[str]:
    """Return any Cold War anchor terms that are mistakenly in the stopword list."""
    return [t for t in COLD_WAR_ANCHORS if t in ALL_STOPWORDS]


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


def build_pre_stopword_vocab(df: pd.DataFrame) -> int:
    """Count unique valid tokens before stopword removal."""
    vocab: set[str] = set()
    for text in tqdm(df['content_clean_lines'].dropna(), desc="Pre-SW vocabulary", unit="row"):
        vocab.update(_tokenize_clean(str(text)))
    return len(vocab)


def run_tokenisation(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Processing tokens...")

    token_lists = with_progress(
        df['content_clean_lines'],
        tokenize_and_filter,
        desc="Tokenise (no lemma)",
    )
    df['content_model_no_lemma'] = token_lists.str.join(' ')
    df['token_count_model_no_lemma'] = token_lists.str.len().fillna(0).astype(int)

    # Tokenize → remove stopwords → lemmatize → remove stopwords again
    lemma_token_lists = with_progress(
        token_lists,
        lemmatize_and_filter,
        desc="Tokenise + lemmatise",
    )
    df['content_model_lemma'] = lemma_token_lists.str.join(' ')
    df['token_count_model_lemma'] = lemma_token_lists.str.len().fillna(0).astype(int)

    return df


def build_post_stopword_vocab(df: pd.DataFrame) -> int:
    """Count unique tokens after stopword removal (lemma column)."""
    vocab: set[str] = set()
    for text in tqdm(df['content_model_lemma'].dropna(), desc="Post-SW vocabulary", unit="row"):
        if isinstance(text, str) and text.strip():
            vocab.update(text.split())
    return len(vocab)


def save_output(df: pd.DataFrame) -> None:
    print_section("Saving output...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  → {OUTPUT_CSV.relative_to(PHASE4_DIR)}  {len(df):>10,} rows")


def print_summary(
    df: pd.DataFrame,
    vocab_before: int,
    vocab_after: int,
    blocked_anchors: list[str],
) -> None:
    total = len(df)

    # Use word_count_clean as the "before" if available
    if 'word_count_clean' in df.columns:
        before_wc = df['word_count_clean'].replace(0, np.nan)
    else:
        before_wc = (df['token_count_model_no_lemma'] + 1).replace(0, np.nan)
    retention = (df['token_count_model_no_lemma'] / before_wc).dropna()
    median_retention = retention.median()

    zero_pages = int((df['token_count_model_lemma'] == 0).sum())

    # Anchor term presence in model text
    anchor_status = []
    model_text_joined = df['content_model_lemma'].fillna('').str.cat(sep=' ')
    model_tokens = set(model_text_joined.split())
    for term in COLD_WAR_ANCHORS:
        present = term in model_tokens
        mark = "ok" if present else "MISSING"
        anchor_status.append(f"{term} {mark}")

    def fmt_ratio(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:>7.1%}"

    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  Pages analysed             : {total:>8,}")
    print(f"  Vocabulary before stopwords  : {vocab_before:>8,}")
    print(f"  Vocabulary after stopwords   : {vocab_after:>8,}")
    print(f"  Median token retention       : {fmt_ratio(median_retention)}")
    print(f"  Pages with zero tokens       : {zero_pages:>8,}")

    if blocked_anchors:
        print(f"  WARNING — anchors in stoplist: {', '.join(blocked_anchors)}")
    else:
        print(f"  Cold War anchors present     : {', '.join(anchor_status)}")

    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEPARATOR)
    print("PHASE 4 — Tokenisation, Stopword Removal and Lemmatisation")
    print(SEPARATOR)

    # Check anchor terms before processing
    blocked = check_anchor_terms()
    if blocked:
        print(f"\n  WARNING: Cold War anchors found in stopword list: {blocked}")
        print("  These will be removed from topic model text!")

    df = load_data()
    vocab_before = build_pre_stopword_vocab(df)
    df = run_tokenisation(df)
    vocab_after = build_post_stopword_vocab(df)
    save_output(df)
    print_summary(df, vocab_before, vocab_after, blocked)


if __name__ == "__main__":
    main()
