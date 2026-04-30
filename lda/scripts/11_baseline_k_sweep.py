"""
11 — Baseline k Sweep  (Stage 4, RQ2 fairness check)
======================================================
Input:  lda/outputs/baseline/baseline_dictionary.gensim
        lda/outputs/baseline/baseline_corpus.mm
        lda/outputs/baseline/baseline_documents.csv  (for c_v texts)
Output: lda/reports/baseline_k_sweep.csv
        lda/reports/baseline_k_sweep_summary.md

Finds the baseline corpus's own optimal k by sweeping
k ∈ {10, 15, 20, 25, 30, 35, 40} with the same frozen LDA
hyperparameters as the full pipeline. The point is fairness: k = 25
was chosen on the full-pipeline coherence curve, so anchoring the
baseline at that k may understate its best achievable c_v. This
script reports the baseline's best-case c_v so we can compare
"full pipeline at its best k" vs "baseline at its best k."

No model files are saved — this is a fairness check, not training.
"""

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

BASELINE_DIR = LDA_DIR / "outputs" / "baseline"
DICT_PATH = BASELINE_DIR / "baseline_dictionary.gensim"
CORPUS_PATH = BASELINE_DIR / "baseline_corpus.mm"
DOCS_PATH = BASELINE_DIR / "baseline_documents.csv"

REPORTS_DIR = LDA_DIR / "reports"
OUT_CSV = REPORTS_DIR / "baseline_k_sweep.csv"
OUT_SUMMARY_MD = REPORTS_DIR / "baseline_k_sweep_summary.md"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K_VALUES = [10, 15, 20, 25, 30, 35, 40]
TEXT_COLUMN = "document_text_lemma"

# Frozen hyperparameters (lda/specs/lda_params.md). Identical to the
# full pipeline so the only thing that varies is k.
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

# Reference c_v from the full pipeline at k=25 (lda/reports/coherence_fine.csv,
# also recorded in lda/reports/baseline_vs_full_summary.md).
FULL_PIPELINE_CV_K25 = 0.6128
FULL_PIPELINE_K = 25

SEP = "=" * 60
SUBSEP = "-" * 60


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------
def write_summary_md(
    df: pd.DataFrame,
    best_k: int,
    best_cv: float,
) -> None:
    delta = FULL_PIPELINE_CV_K25 - best_cv
    full_still_wins = best_cv < FULL_PIPELINE_CV_K25

    lines: list[str] = []
    lines.append("# Baseline k Sweep — Fairness Check")
    lines.append("")
    lines.append(
        "k = 25 was chosen on the full-pipeline coherence curve. To check "
        "whether that choice unfairly disadvantages the baseline, this "
        "script sweeps the baseline corpus over a broad grid of k with "
        "all other LDA hyperparameters frozen (lda/specs/lda_params.md). "
        "The reported best c_v is the baseline's own best-case score."
    )
    lines.append("")
    lines.append(f"k grid: {K_VALUES}")
    lines.append("")

    # Per-k table
    lines.append("## Per-k results")
    lines.append("")
    lines.append("| k | c_v | log_perplexity |")
    lines.append("|--:|----:|---------------:|")
    for _, row in df.iterrows():
        flag = "  ★" if int(row["k"]) == best_k else ""
        lines.append(
            f"| {int(row['k'])} | {row['c_v']:.4f}{flag} "
            f"| {row['log_perplexity']:.4f} |"
        )
    lines.append("")

    # Comparison
    lines.append("## Best k vs. full pipeline")
    lines.append("")
    lines.append(f"- Baseline best k         : **{best_k}**  (c_v = {best_cv:.4f})")
    lines.append(f"- Full pipeline at k = {FULL_PIPELINE_K}  : c_v = "
                 f"{FULL_PIPELINE_CV_K25:.4f}")
    lines.append(f"- Δ (full − baseline_best): {delta:+.4f}")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    if full_still_wins:
        lines.append(
            f"At its own best k ({best_k}), the baseline still scores "
            f"c_v = {best_cv:.4f}, which is {delta:+.4f} below the full "
            f"pipeline's k = {FULL_PIPELINE_K} score of {FULL_PIPELINE_CV_K25:.4f}. "
            f"The coherence advantage reported in script 09 therefore does "
            f"not depend on disadvantaging the baseline by anchoring it at "
            f"k = {FULL_PIPELINE_K}: even when the baseline is given its own "
            f"best-case k from this grid, the full pipeline still wins. "
            f"Combined with the contamination and noise-topic counts, this "
            f"strengthens the RQ2 conclusion that archive-specific "
            f"preprocessing produces a measurably better topic model."
        )
    else:
        lines.append(
            f"At its own best k ({best_k}), the baseline scores "
            f"c_v = {best_cv:.4f}, which is {-delta:+.4f} above the full "
            f"pipeline's k = {FULL_PIPELINE_K} score of {FULL_PIPELINE_CV_K25:.4f}. "
            f"This means the headline coherence number from script 09 was "
            f"sensitive to the choice of k: anchoring the baseline at k = "
            f"{FULL_PIPELINE_K} understated its achievable c_v. The "
            f"contamination and noise-topic counts from script 09 remain "
            f"the more direct evidence for RQ2 — c_v alone does not "
            f"distinguish the two preprocessing regimes once k is allowed "
            f"to vary."
        )
    lines.append("")

    OUT_SUMMARY_MD.write_text("\n".join(lines))
    print(f"Saved → {OUT_SUMMARY_MD.relative_to(LDA_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(SEP)
    print("11 — Baseline k Sweep  (Stage 4, RQ2 fairness check)")
    print(SEP)
    print(f"  k grid: {K_VALUES}")
    print(f"  Hyperparameters: alpha=auto, eta=auto, passes=10, "
          f"iterations=400,\n                   chunksize=2000, "
          f"update_every=3, seed=42")

    for p in (DICT_PATH, CORPUS_PATH, DOCS_PATH):
        if not p.exists():
            raise SystemExit(f"ERROR: missing input — run script 07 first: {p}")

    # ------------------------------------------------------------------
    # 1. Load inputs once
    # ------------------------------------------------------------------
    print("\nLoading dictionary ...")
    dictionary = Dictionary.load(str(DICT_PATH))
    print(f"  Vocabulary  : {len(dictionary):,}")

    print("Loading corpus ...")
    mm = MmCorpus(str(CORPUS_PATH))
    corpus = list(mm)
    print(f"  Documents   : {len(corpus):,}")

    print("Loading texts for c_v coherence ...")
    df_docs = pd.read_csv(DOCS_PATH, low_memory=False)
    texts = df_docs[TEXT_COLUMN].fillna("").str.split().tolist()
    print(f"  Texts       : {len(texts):,}")
    if len(texts) != len(corpus):
        print(f"  WARNING: text/corpus length mismatch "
              f"(texts={len(texts)}, corpus={len(corpus)})")

    # ------------------------------------------------------------------
    # 2. Sweep
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("Sweeping k")
    print(SEP)

    rows: list[dict] = []
    for k in tqdm(K_VALUES, desc="k sweep"):
        print(f"\n  k = {k:>2}: training ...")
        t0 = time.perf_counter()
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            **LDA_PARAMS,
        )
        train_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        cm = CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        c_v = float(cm.get_coherence())
        coh_sec = time.perf_counter() - t1

        log_perp = float(lda.log_perplexity(corpus))

        rows.append({
            "k": k,
            "c_v": round(c_v, 6),
            "log_perplexity": round(log_perp, 4),
        })
        print(f"           c_v = {c_v:.4f}  log_perp = {log_perp:.4f}  "
              f"train = {train_sec:.1f}s  coh = {coh_sec:.1f}s")

    # ------------------------------------------------------------------
    # 3. Save CSV
    # ------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 4. Find best k
    # ------------------------------------------------------------------
    best_idx = df["c_v"].idxmax()
    best_k = int(df.loc[best_idx, "k"])
    best_cv = float(df.loc[best_idx, "c_v"])

    write_summary_md(df, best_k, best_cv)

    # ------------------------------------------------------------------
    # 5. Stdout summary
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("BASELINE k SWEEP — SUMMARY")
    print(SEP)
    print(f"  {'k':>4} {'c_v':>10} {'log_perp':>12}")
    print(f"  {'-'*4} {'-'*10} {'-'*12}")
    for _, r in df.iterrows():
        flag = "  ★" if int(r["k"]) == best_k else ""
        print(f"  {int(r['k']):>4} {r['c_v']:>10.4f} "
              f"{r['log_perplexity']:>12.4f}{flag}")

    delta = FULL_PIPELINE_CV_K25 - best_cv
    full_still_wins = best_cv < FULL_PIPELINE_CV_K25
    verdict = (
        "Full pipeline at k=25 STILL beats baseline at its best k"
        if full_still_wins
        else "Baseline at its best k BEATS full pipeline at k=25"
    )
    print(f"\n  Baseline best k       : {best_k}  (c_v = {best_cv:.4f})")
    print(f"  Full pipeline (k=25)  : c_v = {FULL_PIPELINE_CV_K25:.4f}")
    print(f"  Δ (full − baseline)   : {delta:+.4f}")
    print(f"  Verdict: {verdict}")
    print(SEP)


if __name__ == "__main__":
    main()
