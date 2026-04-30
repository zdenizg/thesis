"""
07 — Build Baseline Corpus  (Stage 4, RQ2)
============================================
Input:  pipeline/phase1/data/pages_phase1_structural.csv
        (83,568 rows — Phase 1 output, post-empty-page-removal,
        pre-archival-cleaning)
Output: lda/outputs/baseline/baseline_documents.csv
        lda/outputs/baseline/baseline_dictionary.gensim
        lda/outputs/baseline/baseline_corpus.mm  (+ corpus.mm.index)
        lda/outputs/baseline/baseline_metadata.json

Builds the minimal-preprocessing baseline corpus per
`lda/specs/baseline_spec.md`:

  1. lowercase
  2. NLTK Penn Treebank tokenisation
  3. NLTK English stopword removal (standard list only)
  4. WordNet lemmatisation (default POS = noun)
  5. token filter: len >= 2, no digits, not pure punctuation
  6. aggregation by file_id
  7. chunking at 5,000 tokens
  8. drop documents (or chunks) under 50 tokens
  9. Gensim Dictionary with filter_extremes(no_below=5, no_above=0.5)
 10. bag-of-words corpus

Deliberately omitted (these are what the baseline measures the effect
of): Phase 2 archive boilerplate regex, Phase 3 line filtering, Phase 4
archive stopwords, Phase 5 page-quality filtering, Phase 6B blacklist.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LDA_DIR = SCRIPT_DIR.parent
REPO_ROOT = LDA_DIR.parent

INPUT_CSV = REPO_ROOT / "pipeline" / "phase1" / "data" / "pages_phase1_structural.csv"
OUTPUT_DIR = LDA_DIR / "outputs" / "baseline"

DOCS_PATH = OUTPUT_DIR / "baseline_documents.csv"
DICT_PATH = OUTPUT_DIR / "baseline_dictionary.gensim"
CORPUS_PATH = OUTPUT_DIR / "baseline_corpus.mm"
META_PATH = OUTPUT_DIR / "baseline_metadata.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = 5_000
MIN_DOC_TOKENS = 50

NO_BELOW = 5
NO_ABOVE = 0.5

# Reference numbers from the full pipeline (printed for comparison).
FULL_PIPELINE_DOC_COUNT = 4_049
FULL_PIPELINE_VOCAB = None  # filled in lazily from corpus_metadata.json

SEP = "=" * 60
SUBSEP = "-" * 60


# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------
def ensure_nltk_resources() -> None:
    """Download anything missing so the script is self-contained."""
    needed = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource, pkg in needed:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(pkg, quiet=True)


# ---------------------------------------------------------------------------
# Token-level normalisation
# ---------------------------------------------------------------------------
def is_keepable(token: str) -> bool:
    """Token filter: length >= 2, no digits, at least one alpha char
    (i.e. not pure punctuation)."""
    if len(token) < 2:
        return False
    has_alpha = False
    for c in token:
        if c.isdigit():
            return False
        if c.isalpha():
            has_alpha = True
    return has_alpha


def make_normaliser():
    """Build a single-page tokeniser closure with cached stopwords +
    lemmatiser. Returns a function page_text → list[str]."""
    stop_set = set(stopwords.words("english"))
    lemmatiser = WordNetLemmatizer()

    def normalise(page_text: str) -> list[str]:
        if not isinstance(page_text, str) or not page_text:
            return []
        # Lowercase before tokenisation so the stopword list (which is
        # all-lowercase) and the lemmatiser see consistent input.
        tokens = word_tokenize(page_text.lower())
        out: list[str] = []
        for tok in tokens:
            if tok in stop_set:
                continue
            lem = lemmatiser.lemmatize(tok)
            if lem in stop_set:
                continue
            if is_keepable(lem):
                out.append(lem)
        return out

    return normalise


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def load_phase1(path: Path) -> pd.DataFrame:
    print(f"\nLoading {path.relative_to(REPO_ROOT)} ...")
    df = pd.read_csv(path, low_memory=False, usecols=["file_id", "page_number", "content"])
    df["content"] = df["content"].fillna("")
    print(f"  Pages loaded            : {len(df):,}")
    print(f"  Unique file_ids         : {df['file_id'].nunique():,}")
    return df


def normalise_pages(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the minimal-preprocessing token pipeline to every page."""
    print("\nNormalising pages (lowercase → tokenise → stopword → lemma → filter) ...")
    normalise = make_normaliser()
    tqdm.pandas(desc="  normalise")
    df = df.copy()
    df["tokens"] = df["content"].progress_apply(normalise)
    df["page_token_count"] = df["tokens"].str.len()
    return df


def aggregate_by_file(df_pages: pd.DataFrame) -> pd.DataFrame:
    """Group pages by file_id, preserve page_number ordering, concatenate
    token lists into a single per-file token sequence."""
    print("\nAggregating pages by file_id ...")
    df_pages = df_pages.sort_values(["file_id", "page_number"], kind="stable")
    grouped = df_pages.groupby("file_id", sort=False)["tokens"].apply(
        lambda lists: [t for sub in lists for t in sub]
    )
    df_docs = grouped.reset_index()
    df_docs["token_count"] = df_docs["tokens"].str.len()
    print(f"  Documents (pre-chunk)   : {len(df_docs):,}")
    print(f"  Mean tokens/doc          : {df_docs['token_count'].mean():,.1f}")
    print(f"  Median tokens/doc        : {df_docs['token_count'].median():,.1f}")
    return df_docs


def chunk_documents(df_docs: pd.DataFrame, chunk_size: int) -> tuple[pd.DataFrame, int]:
    """Split documents whose token count exceeds chunk_size into
    numbered sub-documents. Returns (chunked dataframe, count of input
    documents that were chunked)."""
    print(f"\nChunking documents > {chunk_size:,} tokens ...")
    n_long = int((df_docs["token_count"] > chunk_size).sum())
    print(f"  Documents to chunk      : {n_long:,}")

    rows: list[dict] = []
    for _, row in tqdm(df_docs.iterrows(), total=len(df_docs),
                       desc="  chunking", unit="doc"):
        tokens = row["tokens"]
        if len(tokens) <= chunk_size:
            rows.append({
                "file_id": row["file_id"],
                "tokens": tokens,
                "token_count": len(tokens),
            })
            continue
        n_chunks = (len(tokens) + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            chunk = tokens[i * chunk_size:(i + 1) * chunk_size]
            rows.append({
                "file_id": f"{row['file_id']}_chunk_{i + 1:03d}",
                "tokens": chunk,
                "token_count": len(chunk),
            })
    return pd.DataFrame(rows), n_long


def drop_short(df: pd.DataFrame, min_tokens: int) -> tuple[pd.DataFrame, int]:
    """Drop documents under min_tokens. The baseline pipeline emits a
    single token stream per (sub-)document, so the 'BOTH raw AND lemma <
    50' rule from the full pipeline reduces to a single comparison
    here."""
    print(f"\nDropping documents under {min_tokens} tokens ...")
    keep = df["token_count"] >= min_tokens
    dropped = int((~keep).sum())
    print(f"  Dropped                  : {dropped:,}")
    out = df[keep].copy().reset_index(drop=True)
    print(f"  Remaining                : {len(out):,}")
    return out, dropped


def build_dictionary_and_corpus(
    texts: list[list[str]],
) -> tuple[Dictionary, list, int, int]:
    print("\nBuilding dictionary ...")
    dictionary = Dictionary(texts)
    vocab_before = len(dictionary)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    vocab_after = len(dictionary)
    print(f"  Vocabulary before filter : {vocab_before:,}")
    print(f"  Vocabulary after filter  : {vocab_after:,}")

    print("\nBuilding bag-of-words corpus ...")
    corpus = [dictionary.doc2bow(doc) for doc in tqdm(texts, desc="  doc2bow")]
    return dictionary, corpus, vocab_before, vocab_after


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.perf_counter()

    print(SEP)
    print("07 — Build Baseline Corpus  (Stage 4, RQ2)")
    print(SEP)
    print(f"  CHUNK_SIZE              : {CHUNK_SIZE:,}")
    print(f"  MIN_DOC_TOKENS          : {MIN_DOC_TOKENS}")
    print(f"  Dictionary filter       : no_below={NO_BELOW}, no_above={NO_ABOVE}")
    print()
    print("  Baseline applies ONLY: lowercase → Penn Treebank tokenise →")
    print("  NLTK English stopwords → WordNet lemma → filter (>=2 chars,")
    print("  no digits, not pure punctuation). NO archive-specific cleaning.")

    if not INPUT_CSV.exists():
        raise SystemExit(f"ERROR: input not found: {INPUT_CSV}")

    print("\nEnsuring NLTK resources ...")
    ensure_nltk_resources()

    df_pages = load_phase1(INPUT_CSV)
    df_pages = normalise_pages(df_pages)
    df_docs = aggregate_by_file(df_pages)

    docs_pre_chunk = len(df_docs)
    df_chunked, n_chunked = chunk_documents(df_docs, CHUNK_SIZE)
    docs_post_chunk = len(df_chunked)

    df_final, n_dropped_short = drop_short(df_chunked, MIN_DOC_TOKENS)

    texts = df_final["tokens"].tolist()
    dictionary, corpus, vocab_before, vocab_after = build_dictionary_and_corpus(texts)

    n_empty = sum(1 for bow in corpus if not bow)
    if n_empty:
        print(f"  WARNING: {n_empty:,} documents are empty after filter_extremes.")

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSaving artefacts ...")
    df_final.assign(
        document_text_lemma=df_final["tokens"].str.join(" "),
    )[["file_id", "document_text_lemma", "token_count"]].to_csv(DOCS_PATH, index=False)
    print(f"  → {DOCS_PATH.relative_to(LDA_DIR)}")

    dictionary.save(str(DICT_PATH))
    print(f"  → {DICT_PATH.relative_to(LDA_DIR)}")

    MmCorpus.serialize(str(CORPUS_PATH), corpus)
    print(f"  → {CORPUS_PATH.relative_to(LDA_DIR)}")

    metadata = {
        "input_file": str(INPUT_CSV.relative_to(REPO_ROOT)),
        "input_page_count": int(len(df_pages)),
        "input_unique_file_ids": int(df_pages["file_id"].nunique()),
        "documents_pre_chunk": docs_pre_chunk,
        "documents_chunked": int(n_chunked),
        "documents_post_chunk": docs_post_chunk,
        "documents_dropped_short": int(n_dropped_short),
        "documents_final": int(len(df_final)),
        "vocab_before_filter": int(vocab_before),
        "vocab_after_filter": int(vocab_after),
        "no_below": NO_BELOW,
        "no_above": NO_ABOVE,
        "chunk_size": CHUNK_SIZE,
        "min_doc_tokens": MIN_DOC_TOKENS,
        "mean_tokens_per_doc_final": round(float(df_final["token_count"].mean()), 2),
        "median_tokens_per_doc_final": float(df_final["token_count"].median()),
        "max_tokens_per_doc_final": int(df_final["token_count"].max()),
        "empty_after_dictionary_filter": int(n_empty),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  → {META_PATH.relative_to(LDA_DIR)}")

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Side-by-side summary
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("BASELINE CORPUS — SUMMARY")
    print(SEP)
    print(f"  Pages in (Phase 1)              : {len(df_pages):>10,}")
    print(f"  Files (pre-chunk)               : {docs_pre_chunk:>10,}")
    print(f"  Files chunked at {CHUNK_SIZE:,}        : {n_chunked:>10,}")
    print(f"  Sub-documents after chunking    : {docs_post_chunk:>10,}")
    print(f"  Dropped (< {MIN_DOC_TOKENS} tokens)            : {n_dropped_short:>10,}")
    print(f"  Final document count            : {len(df_final):>10,}")
    print(f"  Vocabulary (after filter_extremes): {vocab_after:>8,}")

    print(f"\n{SUBSEP}")
    print("Comparison with full pipeline")
    print(SUBSEP)
    full_meta_path = LDA_DIR / "outputs" / "corpus_metadata.json"
    if full_meta_path.exists():
        with open(full_meta_path) as f:
            full_meta = json.load(f)
        print(f"  Full pipeline documents         : "
              f"{full_meta.get('document_count', 'n/a'):>10}")
        print(f"  Full pipeline vocab             : "
              f"{full_meta.get('vocab_after_filter', 'n/a'):>10}")
    else:
        print(f"  Full pipeline documents (ref)   : {FULL_PIPELINE_DOC_COUNT:>10,}")
        print(f"  Full pipeline vocab (ref)       : (run script 01)")
    print(f"  Baseline   documents            : {len(df_final):>10,}")
    print(f"  Baseline   vocab                : {vocab_after:>10,}")
    print(f"\n  Elapsed                         : {elapsed:>9.1f}s")
    print(SEP)


if __name__ == "__main__":
    main()
