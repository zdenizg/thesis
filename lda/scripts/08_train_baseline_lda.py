"""
08 — Train Baseline LDA  (Stage 4, RQ2)
==========================================
Input:  lda/outputs/baseline/baseline_dictionary.gensim
        lda/outputs/baseline/baseline_corpus.mm
        lda/outputs/baseline/baseline_documents.csv  (for c_v texts)
Output: lda/outputs/baseline/baseline_lda_k25.gensim (+ .state, +.id2word)
        lda/reports/baseline_coherence.csv
        lda/reports/baseline_topics_top_words.csv

Trains an LDA model at k=25 on the minimal-preprocessing baseline
corpus, with the same frozen hyperparameters used for the full
pipeline (lda/specs/lda_params.md). Coherence and the top-20 words per
topic are written in the same format as the full pipeline so that
script 09 can compare them directly.
"""

import time
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, LdaModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LDA_DIR = SCRIPT_DIR.parent

BASELINE_DIR = LDA_DIR / "outputs" / "baseline"
DICT_PATH = BASELINE_DIR / "baseline_dictionary.gensim"
CORPUS_PATH = BASELINE_DIR / "baseline_corpus.mm"
DOCS_PATH = BASELINE_DIR / "baseline_documents.csv"
MODEL_PATH = BASELINE_DIR / "baseline_lda_k25.gensim"

REPORTS_DIR = LDA_DIR / "reports"
COHERENCE_CSV = REPORTS_DIR / "baseline_coherence.csv"
TOPWORDS_CSV = REPORTS_DIR / "baseline_topics_top_words.csv"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K = 25
TOP_N_WORDS = 20

# Frozen hyperparameters (lda/specs/lda_params.md). Identical to the
# full pipeline so the only thing that varies is the preprocessing.
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

# Reference c_v from the full pipeline at k=25 (lda/reports/coherence_fine.csv).
FULL_PIPELINE_CV_K25 = 0.6128

SEP = "=" * 60
SUBSEP = "-" * 60


def build_top_words_table(lda: LdaModel, k: int) -> pd.DataFrame:
    """Match the column layout of lda/reports/topics_k25_top_words.csv."""
    rows = []
    for topic_id in range(k):
        pairs = lda.show_topic(topic_id, topn=TOP_N_WORDS)
        row: dict = {"topic_id": topic_id}
        for i, (word, weight) in enumerate(pairs, 1):
            row[f"word_{i}"] = word
            row[f"weight_{i}"] = round(float(weight), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print(SEP)
    print(f"08 — Train Baseline LDA  (k = {K}, Stage 4, RQ2)")
    print(SEP)
    print(f"  Hyperparameters: alpha=auto, eta=auto, passes=10, "
          f"iterations=400,\n                   chunksize=2000, "
          f"update_every=3, seed=42")

    for p in (DICT_PATH, CORPUS_PATH, DOCS_PATH):
        if not p.exists():
            raise SystemExit(f"ERROR: missing input — run script 07 first: {p}")

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print("\nLoading dictionary ...")
    dictionary = Dictionary.load(str(DICT_PATH))
    print(f"  Vocabulary              : {len(dictionary):,}")

    print("Loading corpus ...")
    mm = MmCorpus(str(CORPUS_PATH))
    corpus = list(mm)
    print(f"  Documents               : {len(corpus):,}")

    print("Loading texts for c_v coherence ...")
    df_docs = pd.read_csv(DOCS_PATH, low_memory=False)
    texts = df_docs["document_text_lemma"].fillna("").str.split().tolist()
    print(f"  Texts                   : {len(texts):,}")
    if len(texts) != len(corpus):
        print(f"  WARNING: text/corpus length mismatch "
              f"(texts={len(texts)}, corpus={len(corpus)})")

    # ------------------------------------------------------------------
    # 2. Train
    # ------------------------------------------------------------------
    print(f"\nTraining LdaModel at k = {K} ...")
    t0 = time.perf_counter()
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=K,
        **LDA_PARAMS,
    )
    train_sec = time.perf_counter() - t0
    print(f"  Trained in {train_sec:.1f}s")

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    lda.save(str(MODEL_PATH))
    print(f"  Saved → {MODEL_PATH.relative_to(LDA_DIR)}")

    a = lda.alpha
    eta_val = float(lda.eta.mean()) if hasattr(lda.eta, "mean") else float(lda.eta)
    print(f"  alpha (learned)         : min={a.min():.4f}  max={a.max():.4f}  "
          f"mean={a.mean():.4f}")
    print(f"  eta                     : {eta_val:.6f}")

    # ------------------------------------------------------------------
    # 3. Coherence + perplexity
    # ------------------------------------------------------------------
    print("\nComputing c_v coherence ...")
    t1 = time.perf_counter()
    cm = CoherenceModel(
        model=lda,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    c_v = cm.get_coherence()
    coh_sec = time.perf_counter() - t1
    log_perp = lda.log_perplexity(corpus)
    print(f"  c_v                      : {c_v:.4f}  ({coh_sec:.1f}s)")
    print(f"  log_perplexity           : {log_perp:.4f}")

    # ------------------------------------------------------------------
    # 4. Save coherence CSV (matches the layout of coherence_fine.csv)
    # ------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "k": K,
        "coherence_c_v": round(float(c_v), 6),
        "log_perplexity": round(float(log_perp), 4),
        "alpha_min": round(float(a.min()), 6),
        "alpha_max": round(float(a.max()), 6),
        "alpha_mean": round(float(a.mean()), 6),
        "eta": round(float(eta_val), 6),
        "train_seconds": round(train_sec, 1),
        "coherence_seconds": round(coh_sec, 1),
    }]).to_csv(COHERENCE_CSV, index=False)
    print(f"\nSaved → {COHERENCE_CSV.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 5. Save top-20 words
    # ------------------------------------------------------------------
    df_words = build_top_words_table(lda, K)
    df_words.to_csv(TOPWORDS_CSV, index=False)
    print(f"Saved → {TOPWORDS_CSV.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 6. Comparison printout
    # ------------------------------------------------------------------
    delta = c_v - FULL_PIPELINE_CV_K25
    direction = "higher" if delta > 0 else "lower" if delta < 0 else "equal"
    print(f"\n{SEP}")
    print("BASELINE vs. FULL PIPELINE  (preview — full comparison in script 09)")
    print(SEP)
    print(f"  Full pipeline c_v (k=25)   : {FULL_PIPELINE_CV_K25:.4f}")
    print(f"  Baseline      c_v (k=25)   : {c_v:.4f}")
    print(f"  Δ (baseline − full)        : {delta:+.4f}  "
          f"(baseline {direction} than full pipeline)")
    print(SEP)


if __name__ == "__main__":
    main()
