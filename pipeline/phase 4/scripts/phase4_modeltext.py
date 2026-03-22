"""
Phase 4: Model-Ready Text Preparation
Applies stopword filtering and token normalization for topic modeling.
"""

import ssl
import sys
import re
import pandas as pd
from collections import Counter

# Fix SSL for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_PATH  = "/Users/denizguvenol/Desktop/thesis/cleaning/phase 3 /data/pages_phase3_linefiltered.csv"
OUTPUT_PATH = "/Users/denizguvenol/Desktop/thesis/cleaning/phase 4/data/pages_phase4_modeltext.csv"

# ── Stopword lists ───────────────────────────────────────────────────────────
ENGLISH_STOPWORDS = set(stopwords.words('english'))

ARCHIVE_STOPWORDS = {
    "record", "document", "copy", "review", "released", "act",
    "page", "division", "station", "office", "memo", "cable",
    "information", "attached",
    # Added: frequent administrative tokens with low topical value
    "mr", "mr.", "date", "number", "section",
    # Added: high-frequency generic tokens with no topical signal
    "would", "may", "one", "also", "use", "made", "new",
    # Added: memo field header — n=42,891, dominates as header not narrative
    "subject",
    # Added via token discovery: reporting verbs (zero topical signal)
    "said", "stated", "advised",
    # Added via token discovery: modal verbs not in NLTK stopwords
    "could", "must",
    # Added via token discovery: document type / template words
    "memorandum", "following", "concerning", "see", "type",
    # Added via token discovery: Spanish function word in intercepted docs
    "que",
}

ALL_STOPWORDS = ENGLISH_STOPWORDS | ARCHIVE_STOPWORDS

# ── Token filter ─────────────────────────────────────────────────────────────
PUNCT_RE = re.compile(r'^[^\w]+$')

def is_valid_token(token: str) -> bool:
    """Return True if token should be kept."""
    if len(token) < 3:
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

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [lemmatizer.lemmatize(t) for t in tokens]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

# ── Process ───────────────────────────────────────────────────────────────────
print("Processing tokens …")

# Vocabulary before stopword removal (on raw tokenised text)
print("  Building pre-stopword vocabulary …")
pre_sw_vocab: Counter = Counter()
for text in df['content_clean_lines']:
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if is_valid_token(t)]
        pre_sw_vocab.update(tokens)

vocab_before = len(pre_sw_vocab)

# Tokenize + filter (no lemma)
print("  Tokenizing (no lemma) …")
df['content_model_no_lemma'] = df['content_clean_lines'].apply(
    lambda t: ' '.join(tokenize_and_filter(t))
)
df['token_count_model_no_lemma'] = df['content_model_no_lemma'].apply(
    lambda t: len(t.split()) if isinstance(t, str) and t.strip() else 0
)

# Tokenize → remove stopwords → lemmatize → remove stopwords again
# First pass removes stopwords in original form (e.g. "was", "has").
# Second pass removes tokens whose lemma is a stopword (e.g. "records"→"record").
print("  Lemmatizing …")
df['content_model_lemma'] = df['content_clean_lines'].apply(
    lambda t: ' '.join(
        tok for tok in lemmatize_tokens(tokenize_and_filter(t))
        if tok not in ALL_STOPWORDS
    )
)
df['token_count_model_lemma'] = df['content_model_lemma'].apply(
    lambda t: len(t.split()) if isinstance(t, str) and t.strip() else 0
)

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"Saving to {OUTPUT_PATH} …")
df.to_csv(OUTPUT_PATH, index=False)
print("  Saved.")

# ── Diagnostics ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSTICS")
print("=" * 60)

total_pages = len(df)
avg_no_lemma = df['token_count_model_no_lemma'].mean()
avg_lemma    = df['token_count_model_lemma'].mean()

print(f"Total pages processed        : {total_pages:,}")
print(f"Avg tokens (no lemma)        : {avg_no_lemma:.1f}")
print(f"Avg tokens (lemma)           : {avg_lemma:.1f}")
print(f"Vocabulary size before SW    : {vocab_before:,}")

# Post-stopword vocab (lemma)
post_sw_vocab: Counter = Counter()
for text in df['content_model_lemma']:
    if isinstance(text, str) and text.strip():
        post_sw_vocab.update(text.split())

vocab_after = len(post_sw_vocab)
print(f"Vocabulary size after SW     : {vocab_after:,}")

print("\nTop 30 tokens in model text (no lemma):")
no_lemma_vocab: Counter = Counter()
for text in df['content_model_no_lemma']:
    if isinstance(text, str) and text.strip():
        no_lemma_vocab.update(text.split())
for rank, (token, count) in enumerate(no_lemma_vocab.most_common(30), 1):
    print(f"  {rank:2d}. {token:<20s} {count:,}")

print("\nTop 30 tokens in model text (lemma):")
for rank, (token, count) in enumerate(post_sw_vocab.most_common(30), 1):
    print(f"  {rank:2d}. {token:<20s} {count:,}")

# ── 10 Example pages ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXAMPLE PAGES (10 samples)")
print("=" * 60)

sample = df[df['content_clean_lines'].notna() & (df['token_count_model_lemma'] > 10)].sample(
    n=min(10, total_pages), random_state=42
)

for i, (_, row) in enumerate(sample.iterrows(), 1):
    orig  = str(row['content_clean_lines'])[:200].replace('\n', ' ')
    model = str(row['content_model_lemma'])[:200]
    print(f"\n--- Page {i} | file_id={row['file_id']} | page={row['page_number']} ---")
    print(f"  ORIGINAL  : {orig}")
    print(f"  MODEL TEXT: {model}")

print("\nDone.")
