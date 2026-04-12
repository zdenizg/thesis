"""
01 — Build Dictionary & Corpus
================================
Input:  pipeline/phase6/data/documents_final.csv
Output: lda/outputs/dictionary.gensim
        lda/outputs/corpus.mm  (+corpus.mm.index)
        lda/outputs/corpus_metadata.json

Reads the frozen Phase 6B output, builds a Gensim Dictionary with
filter_extremes(no_below=5, no_above=0.5) per lda_params.md, and
serialises the bag-of-words corpus in Matrix Market format.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LDA_DIR = SCRIPT_DIR.parent
REPO_ROOT = LDA_DIR.parent

INPUT_CSV = REPO_ROOT / "pipeline" / "phase6" / "data" / "documents_final.csv"
OUTPUT_DIR = LDA_DIR / "outputs"

DICT_PATH = OUTPUT_DIR / "dictionary.gensim"
CORPUS_PATH = OUTPUT_DIR / "corpus.mm"
META_PATH = OUTPUT_DIR / "corpus_metadata.json"

# ---------------------------------------------------------------------------
# Dictionary filtering thresholds (from lda/specs/lda_params.md)
# ---------------------------------------------------------------------------
NO_BELOW = 5
NO_ABOVE = 0.5

# Column containing pre-cleaned, lemmatised text (from Phase 6B)
TEXT_COLUMN = "document_text_lemma"
ID_COLUMN = "file_id"


def main() -> None:
    t0 = time.perf_counter()
    separator = "=" * 60

    # ------------------------------------------------------------------
    # 1. Load documents
    # ------------------------------------------------------------------
    print(separator)
    print("01 — Build Dictionary & Corpus")
    print(separator)

    if not INPUT_CSV.exists():
        print(f"ERROR: input file not found: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading {INPUT_CSV.name} ...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    if TEXT_COLUMN not in df.columns:
        print(f"ERROR: column '{TEXT_COLUMN}' not found. "
              f"Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"  Documents loaded: {len(df):,}")

    # ------------------------------------------------------------------
    # 2. Tokenise by whitespace (text is already pre-cleaned)
    # ------------------------------------------------------------------
    print("\nTokenising by whitespace split ...")
    texts = df[TEXT_COLUMN].fillna("").str.split().tolist()

    doc_lengths = [len(t) for t in texts]
    print(f"  Total tokens: {sum(doc_lengths):,}")
    print(f"  Mean tokens/doc: {sum(doc_lengths) / len(doc_lengths):,.1f}")

    # ------------------------------------------------------------------
    # 3. Build dictionary
    # ------------------------------------------------------------------
    print("\nBuilding dictionary ...")
    dictionary = Dictionary(texts)
    vocab_before = len(dictionary)
    print(f"  Vocabulary BEFORE filtering: {vocab_before:,}")

    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    vocab_after = len(dictionary)
    vocab_dropped = vocab_before - vocab_after
    pct_dropped = 100.0 * vocab_dropped / vocab_before if vocab_before > 0 else 0.0

    print(f"  Vocabulary AFTER filtering:  {vocab_after:,}")
    print(f"  Terms dropped: {vocab_dropped:,} ({pct_dropped:.1f}%)")

    # ------------------------------------------------------------------
    # 4. Build bag-of-words corpus
    # ------------------------------------------------------------------
    print("\nBuilding bag-of-words corpus ...")
    corpus = [dictionary.doc2bow(doc) for doc in tqdm(texts, desc="doc2bow")]

    empty_indices = [i for i, bow in enumerate(corpus) if len(bow) == 0]
    n_empty = len(empty_indices)

    if n_empty > 0:
        print(f"\n  WARNING: {n_empty:,} documents are EMPTY after filtering.")
        # Show identifiers for empty documents
        if ID_COLUMN in df.columns:
            empty_ids = df.iloc[empty_indices][ID_COLUMN].tolist()
            display_limit = 20
            for fid in empty_ids[:display_limit]:
                print(f"    - {fid}")
            if n_empty > display_limit:
                print(f"    ... and {n_empty - display_limit} more")
        else:
            print(f"    Row indices: {empty_indices[:20]}")
        print("  These documents remain in the corpus (not dropped).")
        print("  Inspect before proceeding with LDA training.")
    else:
        print(f"  Empty documents after filtering: 0")

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving dictionary → {DICT_PATH.relative_to(LDA_DIR)}")
    dictionary.save(str(DICT_PATH))

    print(f"Saving corpus     → {CORPUS_PATH.relative_to(LDA_DIR)}")
    MmCorpus.serialize(str(CORPUS_PATH), corpus)

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 6. Corpus metadata
    # ------------------------------------------------------------------
    empty_ids = (
        df.iloc[empty_indices][ID_COLUMN].tolist()
        if n_empty > 0 and ID_COLUMN in df.columns
        else [int(i) for i in empty_indices]
    )
    metadata = {
        "input_file": INPUT_CSV.name,
        "input_row_count": len(df),
        "vocab_before_filter": vocab_before,
        "vocab_after_filter": vocab_after,
        "no_below": NO_BELOW,
        "no_above": NO_ABOVE,
        "document_count": len(corpus),
        "empty_document_count": n_empty,
        "empty_document_identifiers": empty_ids,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saving metadata   → {META_PATH.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("DONE")
    print(separator)
    print(f"  Documents:       {len(df):>10,}")
    print(f"  Vocabulary:      {vocab_after:>10,}")
    print(f"  Corpus entries:  {len(corpus):>10,}")
    print(f"  Empty docs:      {n_empty:>10,}")
    print(f"  Elapsed:         {elapsed:>9.1f}s")
    print(separator)


if __name__ == "__main__":
    main()
