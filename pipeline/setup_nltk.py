"""
One-time NLTK data download for the preprocessing pipeline.

Phase 4 (tokenisation, stopword removal, lemmatisation) requires the
five NLTK resources listed below. Phase 4 will auto-download them on
first run via its own ensure_nltk_resource() helper, but this script
lets you pre-fetch them — useful for CI, air-gapped environments, or
simply to fail fast at install time rather than at first pipeline run.

Resources fetched
-----------------
  stopwords    — NLTK English stopword list, used by phase 4 via
                 `from nltk.corpus import stopwords` to build
                 ENGLISH_STOPWORDS in phase4_modeltext.py.
  wordnet      — WordNet lemma database, used by
                 `nltk.stem.WordNetLemmatizer` to lemmatise tokens
                 in phase 4's content_model_lemma column.
  omw-1.4      — Open Multilingual WordNet, a sibling dataset that
                 WordNet 3.1+ requires for full lemma coverage. The
                 lemmatiser will silently degrade without it.
  punkt        — Legacy Punkt sentence/word tokeniser model, required
                 by `nltk.tokenize.word_tokenize` on older NLTK versions.
                 Kept for compatibility.
  punkt_tab    — Newer tabular form of the Punkt model. Required by
                 NLTK 3.9+; word_tokenize raises LookupError without it.

Usage:
    python pipeline/setup_nltk.py
"""

import ssl

# Fix SSL verification on macOS where the system Python's certificate
# store is sometimes not configured for NLTK's download server.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk

REQUIRED = [
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]


def main() -> None:
    print("Fetching NLTK resources for the preprocessing pipeline...")
    for resource_path, download_name in REQUIRED:
        try:
            nltk.data.find(resource_path)
            print(f"  [skip] {download_name} (already installed)")
        except LookupError:
            print(f"  [get ] {download_name} ...", end=" ", flush=True)
            nltk.download(download_name, quiet=True)
            print("done")
    print("NLTK setup complete.")


if __name__ == "__main__":
    main()
