"""
13 — Reporting-Verbs Stopword Test  (Stage 4, deferred item #2)
================================================================
Re-runs Phase 4 → Phase 6B → LDA(k=25) with the reporting verbs
"said", "stated", "advised" RETAINED (i.e., REMOVED from the archive
stopword list). Everything else — the rest of ARCHIVE_STOPWORDS, the
Phase 5 filters, the Phase 6 chunking and blacklist, the dictionary
filter_extremes settings, and the LDA hyperparameters — is identical
to the main pipeline.

The pipeline scripts on disk are NOT modified. This script reimplements
the relevant logic inline so that the only difference from the main run
is the stopword set.

Inputs
------
  pipeline/phase3/data/pages_phase3_linefiltered.csv
  lda/reports/topics_k25_top_words.csv      (main pipeline reference)

Outputs
-------
  lda/outputs/reporting_verbs/dictionary.gensim
  lda/outputs/reporting_verbs/corpus.mm  (+corpus.mm.index)
  lda/outputs/reporting_verbs/lda_model.gensim (+ .state, +.id2word)
  lda/outputs/reporting_verbs/documents.csv
  lda/reports/reporting_verbs_test.csv
  lda/reports/reporting_verbs_test_topics.md
  lda/reports/reporting_verbs_test_summary.md
"""

import re
import ssl
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NLTK bootstrap (mirrors phase4_modeltext.py)
# ---------------------------------------------------------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
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

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, LdaModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LDA_DIR = SCRIPT_DIR.parent
REPO_ROOT = LDA_DIR.parent

INPUT_CSV = REPO_ROOT / "pipeline" / "phase3" / "data" / "pages_phase3_linefiltered.csv"
MAIN_TOPWORDS_CSV = LDA_DIR / "reports" / "topics_k25_top_words.csv"

OUTPUT_DIR = LDA_DIR / "outputs" / "reporting_verbs"
DICT_PATH = OUTPUT_DIR / "dictionary.gensim"
CORPUS_PATH = OUTPUT_DIR / "corpus.mm"
MODEL_PATH = OUTPUT_DIR / "lda_model.gensim"
DOCS_PATH = OUTPUT_DIR / "documents.csv"

REPORTS_DIR = LDA_DIR / "reports"
RESULTS_CSV = REPORTS_DIR / "reporting_verbs_test.csv"
TOPICS_MD = REPORTS_DIR / "reporting_verbs_test_topics.md"
SUMMARY_MD = REPORTS_DIR / "reporting_verbs_test_summary.md"

SEP = "=" * 60
SUBSEP = "-" * 60

# ---------------------------------------------------------------------------
# Stopwords — copied verbatim from pipeline/phase4/scripts/phase4_modeltext.py
# WITH "said", "stated", "advised" REMOVED from the archive list.
# ---------------------------------------------------------------------------
RETAINED_VERBS = {"said", "stated", "advised"}

ENGLISH_STOPWORDS = set(stopwords.words("english"))

ARCHIVE_STOPWORDS = {
    "record", "document", "copy", "review", "released", "act",
    "page", "division", "station", "office", "memo", "cable",
    "information", "attached",
    "mr", "mr.", "dr.", "jr.", "mrs.", "sr.", "date", "number", "section",
    "would", "may", "one", "also", "use", "made", "new",
    "subject",
    # Reporting verbs INTENTIONALLY OMITTED for this test:
    # "said", "stated", "advised",
    "could", "must",
    "memorandum", "following", "concerning", "see", "type",
    "que",
    "'s",
    "n't",
}

assert RETAINED_VERBS.isdisjoint(ARCHIVE_STOPWORDS), (
    "Reporting verbs leaked back into ARCHIVE_STOPWORDS"
)

ALL_STOPWORDS = ENGLISH_STOPWORDS | ARCHIVE_STOPWORDS

# ---------------------------------------------------------------------------
# Phase 4 token rules (verbatim)
# ---------------------------------------------------------------------------
PUNCT_RE = re.compile(r"^[^\w]+$")
INITIAL_RE = re.compile(r"^[a-z]\.$")

_lemmatizer = WordNetLemmatizer()


def is_valid_token(token: str) -> bool:
    if len(token) < 2:
        return False
    if any(ch.isdigit() for ch in token):
        return False
    if PUNCT_RE.match(token):
        return False
    if INITIAL_RE.match(token):
        return False
    return True


def _tokenize_clean(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return [t for t in word_tokenize(text.lower()) if is_valid_token(t)]


def tokenize_and_filter(text: str) -> list[str]:
    return [t for t in _tokenize_clean(text) if t not in ALL_STOPWORDS]


def lemmatize_and_filter(tokens: list[str]) -> list[str]:
    return [t for t in (_lemmatizer.lemmatize(tok) for tok in tokens)
            if t not in ALL_STOPWORDS]


# ---------------------------------------------------------------------------
# Phase 5 thresholds (verbatim)
# ---------------------------------------------------------------------------
MIN_TOKENS = 15
NON_ASCII_THRESH = 0.05


def non_ascii_ratio(text: object) -> float:
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


# ---------------------------------------------------------------------------
# Phase 6B blacklist + chunking (verbatim)
# ---------------------------------------------------------------------------
ARCHIVE_TOKEN_BLACKLIST = {
    "umbra", "noforn", "orcon", "wnintel", "moray", "tud",
    "decl", "drv", "css", "originator", "ernment",
    "fpmr", "cfr", "sgswirl", "hcf", "limdis",
    "rybat", "exdis", "nodis", "typic", "slugs",
    "docid", "nw", "iden", "mhfno", "sensind",
}

_META_PHRASES = [
    "doc id",
    "jfk assassination system identification",
    "record number",
    "document type",
    "textual document",
    "current status",
    "opening criteria",
    "originating",
    "last review",
    "restrictions",
    "nsa hcf",
    "moray",
    "agency file number",
    "records series",
    "record series",
]
_META_RE = re.compile(
    r"(?<!\S)(?:"
    + "|".join(re.escape(p) for p in sorted(_META_PHRASES, key=len, reverse=True))
    + r")(?!\S)",
    re.IGNORECASE,
)

CHUNK_SIZE = 5_000
MIN_DOC_TOKENS = 50


def _apply_token_blacklist(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return " ".join(t for t in text.split() if t.lower() not in ARCHIVE_TOKEN_BLACKLIST)


def _strip_phrases(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _META_RE.sub(" ", text)
    return re.sub(r" {2,}", " ", cleaned).strip()


def clean_text_p6b(text: str) -> str:
    return _strip_phrases(_apply_token_blacklist(text))


# ---------------------------------------------------------------------------
# LDA configuration (frozen, lda/specs/lda_params.md)
# ---------------------------------------------------------------------------
K = 25
TOP_N_WORDS = 20

LDA_PARAMS = dict(
    alpha="auto",
    eta="auto",
    passes=10,
    iterations=400,
    chunksize=2000,
    update_every=3,
    minimum_probability=0.01,
    eval_every=None,
    random_state=42,
)

NO_BELOW = 5
NO_ABOVE = 0.5

MAIN_PIPELINE_CV_K25 = 0.6128
DELTA_THRESHOLD = 0.01

# ---------------------------------------------------------------------------
# Pipeline reimplementations
# ---------------------------------------------------------------------------


def run_phase4(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}")
    print("PHASE 4 (re-run with reporting verbs RETAINED)")
    print(SEP)
    print(f"  Stopword count: {len(ALL_STOPWORDS):,}")
    print(f"  Verbs retained (excluded from stoplist): "
          f"{sorted(RETAINED_VERBS)}")
    for v in sorted(RETAINED_VERBS):
        assert v not in ALL_STOPWORDS, f"{v} unexpectedly still in stoplist"

    tqdm.pandas(desc="Tokenise (no lemma)")
    no_lemma_lists = df["content_clean_lines"].progress_apply(tokenize_and_filter)
    df["content_model_no_lemma"] = no_lemma_lists.str.join(" ")
    df["token_count_model_no_lemma"] = no_lemma_lists.str.len().fillna(0).astype(int)

    tqdm.pandas(desc="Tokenise + lemmatise")
    lemma_lists = no_lemma_lists.progress_apply(lemmatize_and_filter)
    df["content_model_lemma"] = lemma_lists.str.join(" ")
    df["token_count_model_lemma"] = lemma_lists.str.len().fillna(0).astype(int)

    return df


def run_phase5(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}")
    print("PHASE 5 (filter pages)")
    print(SEP)
    n_in = len(df)

    tqdm.pandas(desc="Non-ASCII ratio")
    mask_non_english = df["content"].progress_apply(non_ascii_ratio) > NON_ASCII_THRESH
    masks = {
        "sparse": df["token_count_model_lemma"] < MIN_TOKENS,
        "low_content": df["is_low_content_page"] == True,  # noqa: E712
        "cover": df["is_likely_cover_page"] == True,  # noqa: E712
        "non_english": mask_non_english,
    }
    excluded = pd.Series(False, index=df.index)
    for label, m in masks.items():
        m = m.fillna(False).astype(bool)
        print(f"  {label:<14}: {int(m.sum()):>8,}")
        excluded = excluded | m

    retained = df.loc[~excluded].copy()
    print(f"  Pages in            : {n_in:>8,}")
    print(f"  Pages retained      : {len(retained):>8,}")
    print(f"  Pages excluded      : {int(excluded.sum()):>8,}")
    return retained


def run_phase6a(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}")
    print("PHASE 6A (aggregate page → document)")
    print(SEP)

    df = df.sort_values(["file_id", "page_number"])
    rows = []
    for fid, g in tqdm(df.groupby("file_id", sort=False),
                       desc="Aggregating", unit="doc"):
        text_no_lemma = " ".join(
            v for v in g["content_model_no_lemma"].dropna() if isinstance(v, str)
        )
        text_lemma = " ".join(
            v for v in g["content_model_lemma"].dropna() if isinstance(v, str)
        )
        rows.append({
            "file_id": fid,
            "document_text": text_no_lemma,
            "document_text_lemma": text_lemma,
            "token_count": len(text_no_lemma.split()),
            "token_count_lemma": len(text_lemma.split()),
            "pages_retained": len(g),
        })
    docs = pd.DataFrame(rows)
    print(f"  Documents produced  : {len(docs):>8,}")
    if len(docs):
        tc = docs["token_count"]
        print(f"  Mean tokens/doc     : {tc.mean():>8.1f}")
        print(f"  Median tokens/doc   : {tc.median():>8.1f}")
        print(f"  Docs > {CHUNK_SIZE:,} tokens  : "
              f"{int((tc > CHUNK_SIZE).sum()):>8,}")
    return docs


def run_phase6b(docs: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}")
    print("PHASE 6B (blacklist strip + chunk > 5,000 + drop < 50)")
    print(SEP)

    tqdm.pandas(desc="Cleaning text")
    docs["document_text"] = docs["document_text"].progress_apply(clean_text_p6b)
    tqdm.pandas(desc="Cleaning lemma text")
    docs["document_text_lemma"] = docs["document_text_lemma"].progress_apply(clean_text_p6b)
    docs["token_count"] = docs["document_text"].str.split().str.len().fillna(0).astype(int)
    docs["token_count_lemma"] = docs["document_text_lemma"].str.split().str.len().fillna(0).astype(int)

    n_long = int((docs["token_count"] > CHUNK_SIZE).sum())
    print(f"  Documents > {CHUNK_SIZE:,} pre-chunk: {n_long:,}")

    rows = []
    for _, row in tqdm(docs.iterrows(), total=len(docs),
                       desc="Chunking", unit="doc"):
        toks = row["document_text"].split() if isinstance(row["document_text"], str) else []
        toks_lem = row["document_text_lemma"].split() if isinstance(row["document_text_lemma"], str) else []

        if len(toks) <= CHUNK_SIZE:
            rows.append(row.to_dict())
            continue

        n_chunks = (len(toks) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in range(n_chunks):
            ct = toks[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
            cl = toks_lem[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
            new_row = row.to_dict()
            new_row["file_id"] = f"{row['file_id']}_chunk_{i + 1:03d}"
            new_row["document_text"] = " ".join(ct)
            new_row["document_text_lemma"] = " ".join(cl)
            new_row["token_count"] = len(ct)
            new_row["token_count_lemma"] = len(cl)
            rows.append(new_row)

    chunked = pd.DataFrame(rows, columns=docs.columns)
    print(f"  Documents after chunk: {len(chunked):,}")

    before = len(chunked)
    chunked = chunked[chunked["token_count"] >= MIN_DOC_TOKENS].copy()
    print(f"  Dropped (raw < {MIN_DOC_TOKENS}) : {before - len(chunked):,}")
    before_lem = len(chunked)
    chunked = chunked[chunked["token_count_lemma"] >= MIN_DOC_TOKENS].copy()
    print(f"  Dropped (lem < {MIN_DOC_TOKENS}) : {before_lem - len(chunked):,}")
    print(f"  Final document count : {len(chunked):,}")

    return chunked.reset_index(drop=True)


# ---------------------------------------------------------------------------
# LDA training + reporting
# ---------------------------------------------------------------------------


def build_dictionary_and_corpus(
    docs: pd.DataFrame,
) -> tuple[Dictionary, list, list[list[str]]]:
    print(f"\n{SEP}")
    print(f"DICTIONARY  (filter_extremes no_below={NO_BELOW}, no_above={NO_ABOVE})")
    print(SEP)
    texts = docs["document_text_lemma"].fillna("").str.split().tolist()
    dictionary = Dictionary(texts)
    print(f"  Vocab BEFORE filter : {len(dictionary):,}")
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    print(f"  Vocab AFTER  filter : {len(dictionary):,}")
    corpus = [dictionary.doc2bow(t) for t in tqdm(texts, desc="doc2bow")]
    n_empty = sum(1 for bow in corpus if len(bow) == 0)
    print(f"  Empty docs (post)   : {n_empty:,}")
    return dictionary, corpus, texts


def train_lda(corpus: list, dictionary: Dictionary) -> tuple[LdaModel, float]:
    print(f"\n{SEP}")
    print(f"LDA TRAINING  (k={K})")
    print(SEP)
    t0 = time.perf_counter()
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=K,
        **LDA_PARAMS,
    )
    train_sec = time.perf_counter() - t0
    print(f"  Trained in {train_sec:.1f}s")
    a = lda.alpha
    print(f"  alpha (learned)     : "
          f"min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}")
    return lda, train_sec


def compute_coherence(
    lda: LdaModel, dictionary: Dictionary, texts: list[list[str]],
) -> float:
    print(f"\n{SEP}")
    print("COHERENCE  (c_v)")
    print(SEP)
    t0 = time.perf_counter()
    cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary,
                        coherence="c_v")
    c_v = cm.get_coherence()
    print(f"  c_v                 : {c_v:.4f}  ({time.perf_counter() - t0:.1f}s)")
    return float(c_v)


def build_top_words_table(lda: LdaModel) -> pd.DataFrame:
    rows = []
    for topic_id in range(K):
        pairs = lda.show_topic(topic_id, topn=TOP_N_WORDS)
        row: dict = {"topic_id": topic_id}
        for i, (word, weight) in enumerate(pairs, 1):
            row[f"word_{i}"] = word
            row[f"weight_{i}"] = round(float(weight), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def find_verbs_in_topics(top_words: pd.DataFrame) -> dict[str, list[int]]:
    """Return {verb: [topic_ids in which it appears in top-20]}."""
    word_cols = [c for c in top_words.columns if c.startswith("word_")]
    out: dict[str, list[int]] = {v: [] for v in sorted(RETAINED_VERBS)}
    for _, row in top_words.iterrows():
        topic_id = int(row["topic_id"])
        words_in_topic = {str(row[c]).lower() for c in word_cols
                          if isinstance(row[c], str)}
        for v in out:
            if v in words_in_topic:
                out[v].append(topic_id)
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def write_results_csv(
    *,
    test_c_v: float,
    test_n_docs: int,
    test_vocab: int,
    main_n_docs: int | None,
    main_vocab: int | None,
) -> None:
    rows = [
        {
            "model": "main_pipeline_k25",
            "c_v": MAIN_PIPELINE_CV_K25,
            "num_docs": main_n_docs if main_n_docs is not None else "",
            "vocab_size": main_vocab if main_vocab is not None else "",
        },
        {
            "model": "verbs_retained_k25",
            "c_v": round(test_c_v, 6),
            "num_docs": test_n_docs,
            "vocab_size": test_vocab,
        },
    ]
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"  → {RESULTS_CSV.relative_to(LDA_DIR)}")


def _topic_top10(row: pd.Series) -> list[str]:
    return [str(row[f"word_{i}"]) for i in range(1, 11) if f"word_{i}" in row.index]


def write_topics_md(
    test_top_words: pd.DataFrame, main_top_words: pd.DataFrame | None,
) -> None:
    lines: list[str] = [
        "# Reporting Verbs Test — Top-10 Words per Topic (k=25)",
        "",
        "Side-by-side comparison: main pipeline (verbs removed) vs "
        "verbs-retained run.",
        "",
    ]
    if main_top_words is None:
        lines.append("_Main-pipeline top-words file not found; "
                     "showing verbs-retained run only._")
        lines.append("")
        lines.append("| topic | verbs_retained — top 10 |")
        lines.append("|---:|---|")
        for _, row in test_top_words.iterrows():
            lines.append(f"| {int(row['topic_id'])} | "
                         f"{', '.join(_topic_top10(row))} |")
    else:
        lines.append("| topic | main_pipeline — top 10 | "
                     "verbs_retained — top 10 |")
        lines.append("|---:|---|---|")
        main_idx = main_top_words.set_index("topic_id")
        for _, row in test_top_words.iterrows():
            tid = int(row["topic_id"])
            test_top10 = ", ".join(_topic_top10(row))
            if tid in main_idx.index:
                main_top10 = ", ".join(_topic_top10(main_idx.loc[tid]))
            else:
                main_top10 = "_(missing)_"
            lines.append(f"| {tid} | {main_top10} | {test_top10} |")
    TOPICS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → {TOPICS_MD.relative_to(LDA_DIR)}")


def write_summary_md(
    *,
    test_c_v: float,
    test_n_docs: int,
    test_vocab: int,
    verbs_in_topics: dict[str, list[int]],
    verdict: str,
    delta: float,
) -> None:
    lines: list[str] = [
        "# Reporting Verbs Test — Summary",
        "",
        "**Question.** Does removing `said`, `stated`, `advised` from "
        "`ARCHIVE_STOPWORDS` improve LDA topic quality?",
        "",
        "**Method.** Re-ran Phase 4 → 6B → LDA(k=25) with these three "
        "verbs RETAINED. All other settings (rest of stoplist, Phase 5 "
        "filters, Phase 6 blacklist + chunking, dictionary "
        f"`filter_extremes(no_below={NO_BELOW}, no_above={NO_ABOVE})`, "
        "and frozen LDA hyperparameters) are identical to the main "
        "pipeline.",
        "",
        "## Results",
        "",
        f"| metric | main pipeline | verbs retained | Δ |",
        f"|---|---:|---:|---:|",
        f"| c_v (k=25) | {MAIN_PIPELINE_CV_K25:.4f} | "
        f"{test_c_v:.4f} | {delta:+.4f} |",
        f"| documents | — | {test_n_docs:,} | — |",
        f"| vocabulary | — | {test_vocab:,} | — |",
        "",
        "## Reporting verbs in top-20 topic words",
        "",
    ]
    for v in sorted(RETAINED_VERBS):
        topics = verbs_in_topics[v]
        if topics:
            lines.append(f"- `{v}` appears in topics: "
                         f"{', '.join(str(t) for t in topics)}")
        else:
            lines.append(f"- `{v}` does not appear in any topic's top-20")
    lines += [
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"_Decision threshold: |Δc_v| ≥ {DELTA_THRESHOLD:.2f} to recommend "
        "a change to the stopword list._",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → {SUMMARY_MD.relative_to(LDA_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(SEP)
    print("13 — Reporting-Verbs Stopword Test")
    print(SEP)
    print(f"  Input               : "
          f"{INPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"  Output dir          : "
          f"{OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print(f"  Reports dir         : "
          f"{REPORTS_DIR.relative_to(REPO_ROOT)}")

    if not INPUT_CSV.exists():
        raise SystemExit(f"ERROR: missing input {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load Phase 3 line-filtered pages
    # ------------------------------------------------------------------
    print(f"\nLoading {INPUT_CSV.name} ...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  Pages loaded        : {len(df):,}")

    # ------------------------------------------------------------------
    # Re-run Phases 4, 5, 6A, 6B with the modified stopword list
    # ------------------------------------------------------------------
    df = run_phase4(df)
    pages_retained = run_phase5(df)
    docs = run_phase6a(pages_retained)
    docs_final = run_phase6b(docs)
    docs_final.to_csv(DOCS_PATH, index=False)
    print(f"\nSaved → {DOCS_PATH.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # Dictionary + corpus + LDA
    # ------------------------------------------------------------------
    dictionary, corpus, texts = build_dictionary_and_corpus(docs_final)
    dictionary.save(str(DICT_PATH))
    MmCorpus.serialize(str(CORPUS_PATH), corpus)
    print(f"  Saved → {DICT_PATH.relative_to(LDA_DIR)}")
    print(f"  Saved → {CORPUS_PATH.relative_to(LDA_DIR)}")

    lda, _train_sec = train_lda(corpus, dictionary)
    lda.save(str(MODEL_PATH))
    print(f"  Saved → {MODEL_PATH.relative_to(LDA_DIR)}")

    test_c_v = compute_coherence(lda, dictionary, texts)
    test_n_docs = len(docs_final)
    test_vocab = len(dictionary)

    # ------------------------------------------------------------------
    # Reference: main-pipeline top words
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("MAIN-PIPELINE REFERENCE")
    print(SEP)
    if MAIN_TOPWORDS_CSV.exists():
        main_top_words = pd.read_csv(MAIN_TOPWORDS_CSV)
        print(f"  Loaded {MAIN_TOPWORDS_CSV.name}  "
              f"({len(main_top_words)} topics)")
    else:
        main_top_words = None
        print(f"  WARNING: {MAIN_TOPWORDS_CSV.name} not found — "
              "topics report will show test run only")

    test_top_words = build_top_words_table(lda)
    verbs_in_topics = find_verbs_in_topics(test_top_words)

    # ------------------------------------------------------------------
    # Comparison printouts
    # ------------------------------------------------------------------
    delta = test_c_v - MAIN_PIPELINE_CV_K25

    print(f"\n{SEP}")
    print("COMPARISON")
    print(SEP)
    print(f"  Corpus stats")
    print(f"    docs    (verbs retained): {test_n_docs:>8,}")
    print(f"    vocab   (verbs retained): {test_vocab:>8,}")
    print(f"  Coherence")
    print(f"    c_v  main pipeline (k=25): "
          f"{MAIN_PIPELINE_CV_K25:.4f}")
    print(f"    c_v  verbs retained (k=25): {test_c_v:.4f}")
    print(f"    Δ (verbs_retained − main): {delta:+.4f}")
    print(f"  Verbs in top-20 topic words")
    for v in sorted(RETAINED_VERBS):
        topics = verbs_in_topics[v]
        loc = ", ".join(str(t) for t in topics) if topics else "(none)"
        print(f"    {v:<8}: {loc}")

    if delta < -DELTA_THRESHOLD:
        verdict = (
            f"**Keep current stopword list.** Retaining said/stated/"
            f"advised lowered c_v by {abs(delta):.4f} (≥ "
            f"{DELTA_THRESHOLD:.2f}). The verbs-removed configuration "
            f"is materially better."
        )
    elif delta > DELTA_THRESHOLD:
        verdict = (
            f"**Remove said/stated/advised from ARCHIVE_STOPWORDS.** "
            f"Retaining them raised c_v by {delta:.4f} (≥ "
            f"{DELTA_THRESHOLD:.2f}). Update the stoplist and re-run "
            f"the pipeline."
        )
    else:
        verdict = (
            f"**Keep current stopword list.** |Δc_v| = "
            f"{abs(delta):.4f} < {DELTA_THRESHOLD:.2f}; the change is "
            f"within noise. Retaining the verbs adds no measurable "
            f"benefit, and the original justification (zero topical "
            f"signal) holds."
        )

    print(f"\n{SUBSEP}")
    print("VERDICT")
    print(SUBSEP)
    print(verdict)

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("REPORTS")
    print(SEP)
    write_results_csv(
        test_c_v=test_c_v,
        test_n_docs=test_n_docs,
        test_vocab=test_vocab,
        main_n_docs=None,
        main_vocab=None,
    )
    write_topics_md(test_top_words, main_top_words)
    write_summary_md(
        test_c_v=test_c_v,
        test_n_docs=test_n_docs,
        test_vocab=test_vocab,
        verbs_in_topics=verbs_in_topics,
        verdict=verdict,
        delta=delta,
    )

    print(f"\n{SEP}")
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    main()
