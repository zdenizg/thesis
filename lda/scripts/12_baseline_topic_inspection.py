"""
12 — Baseline Topic Inspection  (k = 10, qualitative check)
============================================================
Input:  lda/outputs/baseline/baseline_dictionary.gensim
        lda/outputs/baseline/baseline_corpus.mm
Output: lda/reports/baseline_k10_top_words.csv

Script 11 found that the baseline corpus's own best k is 10
(c_v = 0.6380). That higher coherence comes at a cost: with so few
topics on a corpus this size, each topic has to absorb several
distinct themes. To make that visible, this script retrains the
baseline LDA at k = 10 with the same frozen hyperparameters
(seed = 42) and dumps the top-20 words per topic so they can be
read alongside the full pipeline's k = 25 topics.

No noise check, no representative docs, no pyLDAvis — purely a
qualitative readout.
"""

import time
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LDA_DIR = SCRIPT_DIR.parent

BASELINE_DIR = LDA_DIR / "outputs" / "baseline"
DICT_PATH = BASELINE_DIR / "baseline_dictionary.gensim"
CORPUS_PATH = BASELINE_DIR / "baseline_corpus.mm"

REPORTS_DIR = LDA_DIR / "reports"
OUT_CSV = REPORTS_DIR / "baseline_k10_top_words.csv"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K = 10
TOP_N_WORDS = 20

# Frozen hyperparameters — identical to scripts 08 and 11.
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

SEP = "=" * 60


def main() -> None:
    print(SEP)
    print(f"12 — Baseline Topic Inspection  (k = {K})")
    print(SEP)

    for p in (DICT_PATH, CORPUS_PATH):
        if not p.exists():
            raise SystemExit(f"ERROR: missing input — run script 07 first: {p}")

    print("\nLoading dictionary ...")
    dictionary = Dictionary.load(str(DICT_PATH))
    print(f"  Vocabulary  : {len(dictionary):,}")

    print("Loading corpus ...")
    mm = MmCorpus(str(CORPUS_PATH))
    corpus = list(mm)
    print(f"  Documents   : {len(corpus):,}")

    print(f"\nTraining LdaModel at k = {K} (seed = 42) ...")
    t0 = time.perf_counter()
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=K,
        **LDA_PARAMS,
    )
    print(f"  Trained in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Top-20 words per topic
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for topic_id in range(K):
        pairs = lda.show_topic(topic_id, topn=TOP_N_WORDS)
        row: dict = {"topic_id": topic_id}
        for i, (word, weight) in enumerate(pairs, 1):
            row[f"word_{i}"] = word
            row[f"weight_{i}"] = round(float(weight), 6)
        rows.append(row)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_words = pd.DataFrame(rows)
    df_words.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # Stdout printout
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print(f"BASELINE TOP-{TOP_N_WORDS} WORDS  (k = {K})")
    print(SEP)
    for topic_id in range(K):
        pairs = lda.show_topic(topic_id, topn=TOP_N_WORDS)
        words = ", ".join(w for w, _ in pairs)
        print(f"\ntopic {topic_id:>2}:")
        print(f"  {words}")
    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
