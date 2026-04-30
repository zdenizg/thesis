"""
06 — Chunk-Size Sensitivity
============================
Input:  pipeline/phase6/data/documents_for_modeling.csv  (Phase 6A output,
        2,560 documents, pre-chunking)
Output: lda/outputs/chunk_sensitivity/{chunk_size}/dictionary.gensim
        lda/outputs/chunk_sensitivity/{chunk_size}/corpus.mm
        lda/outputs/chunk_sensitivity/{chunk_size}/lda_model.gensim (+state, +id2word)
        lda/outputs/chunk_sensitivity/{chunk_size}/coherence.json
        lda/reports/chunk_sensitivity.csv
        lda/reports/chunk_sensitivity_topics.md
        lda/reports/chunk_sensitivity_summary.md

Tests whether the 5,000-token chunk size used in Phase 6B is the right
choice by comparing LDA at three chunk sizes (3000, 5000, 10000). For
each setting this script reimplements Phase 6B's chunking and filtering
logic in isolation, builds an independent dictionary and BoW corpus,
trains LDA at k=25 with the frozen hyperparameters, and computes c_v
coherence.

The 5000-token run reimplements Phase 6B from the same input the canonical
pipeline uses, so its coherence should be close to (but not identical
to) the main pipeline's k=25 model — small differences are expected
because the dictionary is rebuilt and because this script applies the
token blacklist only, not the metadata phrase-strip step that Phase 6B
also runs. A divergence above 0.01 c_v from the main model triggers a
warning.

Decision principle (lda_plan.md): the preferred chunk size balances
coherence and interpretability while avoiding unnecessary fragmentation.
If coherence differences are minor (~0.02 c_v), interpretability and
fragmentation counts become the deciding criteria.
"""

import json
import time
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, LdaModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LDA_DIR = SCRIPT_DIR.parent
REPO_ROOT = LDA_DIR.parent

INPUT_CSV = REPO_ROOT / "pipeline" / "phase6" / "data" / "documents_for_modeling.csv"
OUTPUTS_DIR = LDA_DIR / "outputs" / "chunk_sensitivity"
REPORTS_DIR = LDA_DIR / "reports"

CSV_PATH = REPORTS_DIR / "chunk_sensitivity.csv"
TOPICS_MD_PATH = REPORTS_DIR / "chunk_sensitivity_topics.md"
SUMMARY_MD_PATH = REPORTS_DIR / "chunk_sensitivity_summary.md"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZES = [3000, 5000, 10000]
K = 25
MIN_DOC_TOKENS = 50

NO_BELOW = 5
NO_ABOVE = 0.5

TOP_N_WORDS = 20

# Main pipeline's k=25 coherence (from lda/reports/coherence_fine.csv).
# Used as a sanity reference for the 5000-token run.
MAIN_K25_CV = 0.6128
WARN_THRESHOLD = 0.01

# Tolerance band from the decision principle.
TIE_TOLERANCE = 0.02

# Frozen LDA hyperparameters (lda/specs/lda_params.md).
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

# Phase 6B token blacklist (26 terms — copied verbatim from
# pipeline/phase6/scripts/phase6b_modeling_prep.py).
BLACKLIST = {
    "umbra", "noforn", "orcon", "wnintel", "moray", "limdis", "exdis",
    "nodis", "rybat", "typic", "slugs", "sensind", "mhfno", "iden",
    "docid", "nw", "decl", "drv", "originator", "cfr", "css",
    "ernment", "fpmr", "hcf", "sgswirl", "tud",
}

TEXT_COLUMN_RAW = "document_text"
TEXT_COLUMN_LEMMA = "document_text_lemma"
ID_COLUMN = "file_id"

SEP = "=" * 60
SUBSEP = "-" * 60


# ---------------------------------------------------------------------------
# Phase 6B logic — isolated, reusable
# ---------------------------------------------------------------------------
def strip_blacklist(text: str) -> str:
    """Drop blacklisted tokens from a whitespace-delimited string."""
    if not isinstance(text, str) or not text:
        return ""
    return " ".join(t for t in text.split() if t.lower() not in BLACKLIST)


def apply_blacklist(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the 26 blacklist tokens from both raw and lemma columns and
    refresh the token-count columns to reflect the cleaned text."""
    df = df.copy()
    tqdm.pandas(desc="  blacklist (raw)")
    df[TEXT_COLUMN_RAW] = df[TEXT_COLUMN_RAW].progress_apply(strip_blacklist)
    tqdm.pandas(desc="  blacklist (lemma)")
    df[TEXT_COLUMN_LEMMA] = df[TEXT_COLUMN_LEMMA].progress_apply(strip_blacklist)
    df["token_count"] = df[TEXT_COLUMN_RAW].str.split().str.len().fillna(0).astype(int)
    df["token_count_lemma"] = (
        df[TEXT_COLUMN_LEMMA].str.split().str.len().fillna(0).astype(int)
    )
    return df


def chunk_documents(df: pd.DataFrame, chunk_size: int) -> tuple[pd.DataFrame, int]:
    """Split documents whose raw token count exceeds chunk_size into
    numbered sub-documents. Returns (chunked dataframe, count of input
    documents that were chunked)."""
    n_long = int((df["token_count"] > chunk_size).sum())

    rows: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  chunking @ {chunk_size}", unit="doc"):
        raw = row[TEXT_COLUMN_RAW] if isinstance(row[TEXT_COLUMN_RAW], str) else ""
        lem = row[TEXT_COLUMN_LEMMA] if isinstance(row[TEXT_COLUMN_LEMMA], str) else ""
        tokens_raw = raw.split()
        tokens_lem = lem.split()

        if len(tokens_raw) <= chunk_size:
            rows.append(row.to_dict())
            continue

        n_chunks = (len(tokens_raw) + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            chunk_raw = tokens_raw[i * chunk_size:(i + 1) * chunk_size]
            chunk_lem = tokens_lem[i * chunk_size:(i + 1) * chunk_size]
            new_row = row.to_dict()
            new_row[ID_COLUMN] = f"{row[ID_COLUMN]}_chunk_{i + 1:03d}"
            new_row[TEXT_COLUMN_RAW] = " ".join(chunk_raw)
            new_row[TEXT_COLUMN_LEMMA] = " ".join(chunk_lem)
            new_row["token_count"] = len(chunk_raw)
            new_row["token_count_lemma"] = len(chunk_lem)
            rows.append(new_row)

    return pd.DataFrame(rows, columns=df.columns), n_long


def drop_short(df: pd.DataFrame, min_tokens: int) -> tuple[pd.DataFrame, int]:
    """Drop documents where BOTH raw and lemma token counts are below
    min_tokens. Returns (filtered dataframe, dropped count)."""
    keep = (df["token_count"] >= min_tokens) | (df["token_count_lemma"] >= min_tokens)
    dropped = int((~keep).sum())
    return df[keep].copy(), dropped


# ---------------------------------------------------------------------------
# Modelling helpers
# ---------------------------------------------------------------------------
def build_dictionary_and_corpus(
    texts: list[list[str]],
) -> tuple[Dictionary, list]:
    """Build a Gensim Dictionary with filter_extremes(no_below=5,
    no_above=0.5) and the matching BoW corpus."""
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    corpus = [dictionary.doc2bow(doc) for doc in texts]
    return dictionary, corpus


def train_and_score(
    dictionary: Dictionary,
    corpus: list,
    texts: list[list[str]],
    k: int,
) -> tuple[LdaModel, float, float]:
    """Train LDA at k with the frozen hyperparameters; return
    (model, c_v coherence, training seconds)."""
    t0 = time.perf_counter()
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        **LDA_PARAMS,
    )
    train_sec = time.perf_counter() - t0

    cm = CoherenceModel(
        model=lda,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    c_v = cm.get_coherence()
    return lda, c_v, train_sec


def topic_top_words(lda: LdaModel, k: int, n: int) -> list[list[str]]:
    """Return list of top-n word strings for each of the k topics."""
    return [
        [w for w, _ in lda.show_topic(t, topn=n)]
        for t in range(k)
    ]


# ---------------------------------------------------------------------------
# Per-chunk-size pipeline
# ---------------------------------------------------------------------------
def run_for_chunk_size(
    df_input: pd.DataFrame,
    chunk_size: int,
) -> dict:
    """Reapply Phase 6B in isolation at the given chunk size, train LDA at
    k=25, and persist all artefacts. Returns a results dictionary."""
    print(f"\n{SEP}")
    print(f"chunk_size = {chunk_size}")
    print(SEP)

    out_dir = OUTPUTS_DIR / str(chunk_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 6B in isolation ---
    print("\n[1/4] Stripping blacklist tokens ...")
    df = apply_blacklist(df_input)

    print(f"\n[2/4] Chunking documents > {chunk_size} tokens ...")
    df_chunked, n_chunked = chunk_documents(df, chunk_size)
    print(f"  Documents chunked      : {n_chunked:,}")
    print(f"  Rows after chunking    : {len(df_chunked):,}")

    print(f"\n[3a/4] Dropping documents under {MIN_DOC_TOKENS} tokens ...")
    df_final, n_dropped = drop_short(df_chunked, MIN_DOC_TOKENS)
    print(f"  Dropped (< {MIN_DOC_TOKENS} tokens) : {n_dropped:,}")
    print(f"  Final document count   : {len(df_final):,}")

    tc = df_final["token_count"]
    print(f"  Mean tokens/doc        : {tc.mean():,.1f}")
    print(f"  Median tokens/doc      : {tc.median():,.1f}")
    print(f"  Max tokens/doc         : {int(tc.max()):,}")

    # --- Dictionary + corpus ---
    print("\n[3b/4] Building dictionary and corpus ...")
    texts = df_final[TEXT_COLUMN_LEMMA].fillna("").str.split().tolist()
    dictionary, corpus = build_dictionary_and_corpus(texts)
    print(f"  Vocabulary             : {len(dictionary):,}")
    print(f"  Corpus entries         : {len(corpus):,}")

    dict_path = out_dir / "dictionary.gensim"
    corpus_path = out_dir / "corpus.mm"
    dictionary.save(str(dict_path))
    MmCorpus.serialize(str(corpus_path), corpus)
    print(f"  Saved → {dict_path.relative_to(LDA_DIR)}")
    print(f"  Saved → {corpus_path.relative_to(LDA_DIR)}")

    # --- LDA training + coherence ---
    print(f"\n[4/4] Training LDA at k={K} and computing c_v coherence ...")
    lda, c_v, train_sec = train_and_score(dictionary, corpus, texts, K)
    print(f"  Trained in {train_sec:.1f}s")
    print(f"  c_v coherence          : {c_v:.4f}")

    model_path = out_dir / "lda_model.gensim"
    lda.save(str(model_path))
    print(f"  Saved → {model_path.relative_to(LDA_DIR)}")

    # --- Coherence + metadata JSON ---
    coherence_payload = {
        "chunk_size": chunk_size,
        "k": K,
        "c_v": round(float(c_v), 6),
        "num_docs": len(corpus),
        "vocab_size": len(dictionary),
        "n_documents_chunked": n_chunked,
        "n_dropped_short": n_dropped,
        "mean_tokens_per_doc": round(float(tc.mean()), 2),
        "median_tokens_per_doc": float(tc.median()),
        "max_tokens_per_doc": int(tc.max()),
        "train_seconds": round(train_sec, 1),
    }
    coh_path = out_dir / "coherence.json"
    with open(coh_path, "w") as f:
        json.dump(coherence_payload, f, indent=2)
    print(f"  Saved → {coh_path.relative_to(LDA_DIR)}")

    # --- Top-20 words per topic (kept in memory for the comparison report) ---
    top20 = topic_top_words(lda, K, TOP_N_WORDS)

    return {
        **coherence_payload,
        "top_words": top20,
    }


# ---------------------------------------------------------------------------
# Comparison outputs
# ---------------------------------------------------------------------------
def write_csv(results: list[dict]) -> None:
    df = pd.DataFrame([
        {
            "chunk_size": r["chunk_size"],
            "num_docs": r["num_docs"],
            "vocab_size": r["vocab_size"],
            "c_v_coherence": r["c_v"],
        }
        for r in results
    ])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved → {CSV_PATH.relative_to(LDA_DIR)}")


def write_topics_md(results: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Chunk-Size Sensitivity — Top-10 Words per Topic")
    lines.append("")
    lines.append(f"Each section reports the {K} topics from an independent "
                 "LDA model trained at k = 25 with the frozen hyperparameters "
                 "(lda/specs/lda_params.md), differing only in the chunk-size "
                 "threshold applied during the isolated Phase 6B rerun.")
    lines.append("")
    for r in results:
        cs = r["chunk_size"]
        lines.append(f"## chunk_size = {cs}")
        lines.append("")
        lines.append(f"- num_docs: {r['num_docs']:,}")
        lines.append(f"- vocab_size: {r['vocab_size']:,}")
        lines.append(f"- c_v: {r['c_v']:.4f}")
        lines.append("")
        lines.append("| topic | top-10 words |")
        lines.append("|------:|:-------------|")
        for t, words in enumerate(r["top_words"]):
            top10 = ", ".join(words[:10])
            lines.append(f"| {t} | {top10} |")
        lines.append("")
    TOPICS_MD_PATH.write_text("\n".join(lines))
    print(f"Saved → {TOPICS_MD_PATH.relative_to(LDA_DIR)}")


def pick_recommendation(results: list[dict]) -> tuple[int, str]:
    """Apply the lda_plan.md decision principle:
       - if coherence spread is within TIE_TOLERANCE c_v, prefer the
         setting with the fewest chunked documents (least fragmentation),
         breaking ties by lower chunk_size (which is more interpretable
         per chunk);
       - otherwise prefer the setting with clearly higher coherence.
    Returns (recommended chunk_size, rationale string)."""
    cvs = [r["c_v"] for r in results]
    spread = max(cvs) - min(cvs)
    best_cv = max(cvs)
    best_by_cv = [r for r in results if r["c_v"] == best_cv][0]

    if spread <= TIE_TOLERANCE:
        ranked = sorted(results, key=lambda r: (r["n_documents_chunked"], r["chunk_size"]))
        rec = ranked[0]
        rationale = (
            f"Coherence spread across the three settings is {spread:.4f} c_v, "
            f"within the {TIE_TOLERANCE:.2f} tolerance band stated in the "
            f"decision principle. Coherence is therefore not decisive; "
            f"fragmentation becomes the deciding criterion. "
            f"chunk_size = {rec['chunk_size']} fragments {rec['n_documents_chunked']} "
            f"documents — the fewest of the three — so it is preferred."
        )
        return rec["chunk_size"], rationale

    rationale = (
        f"Coherence spread is {spread:.4f} c_v, exceeding the "
        f"{TIE_TOLERANCE:.2f} tolerance band. The setting with the highest "
        f"c_v (chunk_size = {best_by_cv['chunk_size']}, c_v = "
        f"{best_by_cv['c_v']:.4f}) is preferred."
    )
    return best_by_cv["chunk_size"], rationale


def write_summary_md(results: list[dict], recommendation: tuple[int, str]) -> None:
    rec_cs, rationale = recommendation
    best_cv = max(r["c_v"] for r in results)
    best_by_cv = [r for r in results if r["c_v"] == best_cv][0]

    lines: list[str] = []
    lines.append("# Chunk-Size Sensitivity — Summary")
    lines.append("")
    lines.append(f"Three independent LDA runs at k = {K} with the frozen "
                 "hyperparameters (lda/specs/lda_params.md), differing only "
                 "in the chunk-size threshold applied during an isolated "
                 "Phase 6B rerun on `pipeline/phase6/data/documents_for_modeling.csv`.")
    lines.append("")

    # Comparison table
    lines.append("## Comparison")
    lines.append("")
    lines.append("| chunk_size | num_docs | vocab_size | c_v | docs chunked | mean tokens/doc | median tokens/doc |")
    lines.append("|-----------:|---------:|-----------:|----:|-------------:|----------------:|------------------:|")
    for r in results:
        lines.append(
            f"| {r['chunk_size']:,} "
            f"| {r['num_docs']:,} "
            f"| {r['vocab_size']:,} "
            f"| {r['c_v']:.4f} "
            f"| {r['n_documents_chunked']:,} "
            f"| {r['mean_tokens_per_doc']:.1f} "
            f"| {r['median_tokens_per_doc']:.1f} |"
        )
    lines.append("")
    lines.append(f"Highest coherence: chunk_size = {best_by_cv['chunk_size']:,} "
                 f"(c_v = {best_by_cv['c_v']:.4f}).")
    lines.append("")

    # Fragmentation
    lines.append("## Fragmentation")
    lines.append("")
    lines.append("The Phase 6A input contains 2,560 unchunked documents. "
                 "Each setting splits any document whose raw-token count "
                 "exceeds the chunk-size threshold into numbered sub-documents.")
    lines.append("")
    for r in results:
        lines.append(
            f"- chunk_size = {r['chunk_size']:,}: "
            f"{r['n_documents_chunked']:,} input documents fragmented; "
            f"{r['num_docs']:,} sub-documents in the final corpus; "
            f"mean {r['mean_tokens_per_doc']:.1f} / median "
            f"{r['median_tokens_per_doc']:.1f} tokens per (sub-)document."
        )
    lines.append("")

    # Decision
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**Recommended chunk size: {rec_cs:,}.**")
    lines.append("")
    lines.append(rationale)
    lines.append("")

    # Interpretation paragraph
    spread = max(r["c_v"] for r in results) - min(r["c_v"] for r in results)
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"Coherence varies by {spread:.4f} c_v across the three settings. "
        "Smaller chunk sizes split more documents into more sub-documents, "
        "which can sharpen topic boundaries but also dilutes long-document "
        "context, while larger chunk sizes preserve context at the cost of "
        "uneven document-length distribution. Per the decision principle in "
        "`lda_plan.md`, when coherence differences across the three settings "
        f"fall within ~{TIE_TOLERANCE:.2f} c_v the choice is governed by "
        "fragmentation and interpretability rather than by coherence alone; "
        f"this empirical result is what the methodology's Section [X.X] "
        "placeholder is filled with."
    )
    lines.append("")

    SUMMARY_MD_PATH.write_text("\n".join(lines))
    print(f"Saved → {SUMMARY_MD_PATH.relative_to(LDA_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(SEP)
    print("06 — Chunk-Size Sensitivity")
    print(SEP)
    print(f"  k                       : {K}")
    print(f"  chunk sizes             : {CHUNK_SIZES}")
    print(f"  min tokens per doc      : {MIN_DOC_TOKENS}")
    print(f"  blacklist size          : {len(BLACKLIST)}")
    print(f"  dictionary filter       : no_below={NO_BELOW}, no_above={NO_ABOVE}")
    print(f"  hyperparameters         : alpha=auto, eta=auto, passes=10, "
          f"iterations=400, chunksize=2000, update_every=3, seed=42")

    if not INPUT_CSV.exists():
        raise SystemExit(f"ERROR: input not found: {INPUT_CSV}")

    print(f"\nLoading {INPUT_CSV.relative_to(REPO_ROOT)} ...")
    df_input = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  Documents loaded        : {len(df_input):,}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for chunk_size in CHUNK_SIZES:
        results.append(run_for_chunk_size(df_input, chunk_size))

    # ------------------------------------------------------------------
    # Sanity check on the 5000-token run
    # ------------------------------------------------------------------
    five_k = next((r for r in results if r["chunk_size"] == 5000), None)
    if five_k is not None:
        diff = abs(five_k["c_v"] - MAIN_K25_CV)
        print(f"\n{SUBSEP}")
        print("Sanity check vs main pipeline (k=25 on documents_final.csv)")
        print(SUBSEP)
        print(f"  Main pipeline c_v       : {MAIN_K25_CV:.4f}")
        print(f"  This run (5000) c_v     : {five_k['c_v']:.4f}")
        print(f"  Δ                       : {diff:.4f}")
        if diff > WARN_THRESHOLD:
            print(f"  WARNING: Δ exceeds {WARN_THRESHOLD:.2f} threshold.")
            print(f"           Expect small differences (this script applies the")
            print(f"           token blacklist only, not the metadata-phrase strip),")
            print(f"           but a gap this large should be sanity-checked.")
        else:
            print(f"  OK (within {WARN_THRESHOLD:.2f} tolerance).")

    # ------------------------------------------------------------------
    # Comparison outputs
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("Writing comparison reports")
    print(SEP)
    write_csv(results)
    write_topics_md(results)
    recommendation = pick_recommendation(results)
    write_summary_md(results, recommendation)

    # ------------------------------------------------------------------
    # Final stdout table + recommendation
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("FINAL COMPARISON")
    print(SEP)
    print(f"  {'chunk_size':>10} | {'num_docs':>8} | {'vocab':>6} | "
          f"{'c_v':>6} | {'chunked':>7}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")
    for r in results:
        print(f"  {r['chunk_size']:>10,} | {r['num_docs']:>8,} | "
              f"{r['vocab_size']:>6,} | {r['c_v']:>6.4f} | "
              f"{r['n_documents_chunked']:>7,}")

    rec_cs, rationale = recommendation
    print(f"\n{SUBSEP}")
    print(f"RECOMMENDATION: chunk_size = {rec_cs:,}")
    print(SUBSEP)
    print(rationale)
    print(SEP)


if __name__ == "__main__":
    main()
