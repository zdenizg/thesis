"""
10 — Multi-seed Robustness Check  (Stage 4, RQ2 follow-up)
=============================================================
Input:  lda/outputs/dictionary.gensim
        lda/outputs/corpus.mm
        pipeline/phase6/data/documents_final.csv      (full-pipeline texts)
        lda/outputs/baseline/baseline_dictionary.gensim
        lda/outputs/baseline/baseline_corpus.mm
        lda/outputs/baseline/baseline_documents.csv   (baseline texts)
Output: lda/reports/multiseed_robustness.csv
        lda/reports/multiseed_robustness_summary.md

Re-trains the full-pipeline and baseline LDA models at k = 25 across
five random seeds, with all other hyperparameters frozen. Records c_v
coherence for each (seed, model) pair so we can check whether the
single-seed gap reported in script 09 is robust or accidental.

No model files are saved — this is a robustness check, not a final
model.
"""

import time
from pathlib import Path

import numpy as np
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

FULL_DICT_PATH = LDA_DIR / "outputs" / "dictionary.gensim"
FULL_CORPUS_PATH = LDA_DIR / "outputs" / "corpus.mm"
FULL_DOCS_PATH = REPO_ROOT / "pipeline" / "phase6" / "data" / "documents_final.csv"

BASELINE_DIR = LDA_DIR / "outputs" / "baseline"
BASE_DICT_PATH = BASELINE_DIR / "baseline_dictionary.gensim"
BASE_CORPUS_PATH = BASELINE_DIR / "baseline_corpus.mm"
BASE_DOCS_PATH = BASELINE_DIR / "baseline_documents.csv"

REPORTS_DIR = LDA_DIR / "reports"
OUT_CSV = REPORTS_DIR / "multiseed_robustness.csv"
OUT_SUMMARY_MD = REPORTS_DIR / "multiseed_robustness_summary.md"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K = 25
SEEDS = [42, 123, 456, 789, 2025]
TEXT_COLUMN = "document_text_lemma"

# Frozen hyperparameters (lda/specs/lda_params.md). random_state varies.
LDA_PARAMS_BASE = dict(
    alpha="auto",
    eta="auto",
    passes=10,
    iterations=400,
    chunksize=2000,
    update_every=3,
    minimum_probability=0.01,
    eval_every=None,
)

SEP = "=" * 60
SUBSEP = "-" * 60


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_corpus_bundle(
    dict_path: Path,
    corpus_path: Path,
    docs_path: Path,
    label: str,
) -> tuple[Dictionary, list, list[list[str]]]:
    for p in (dict_path, corpus_path, docs_path):
        if not p.exists():
            raise SystemExit(f"ERROR: missing input for {label}: {p}")
    print(f"  [{label}] dictionary  : {dict_path.relative_to(LDA_DIR)}")
    dictionary = Dictionary.load(str(dict_path))
    print(f"  [{label}] corpus      : {corpus_path.relative_to(LDA_DIR)}")
    mm = MmCorpus(str(corpus_path))
    corpus = list(mm)
    print(f"  [{label}] texts       : {docs_path}")
    df = pd.read_csv(docs_path, usecols=[TEXT_COLUMN], low_memory=False)
    texts = df[TEXT_COLUMN].fillna("").str.split().tolist()
    print(f"  [{label}] vocab={len(dictionary):,}  "
          f"docs={len(corpus):,}  texts={len(texts):,}")
    if len(texts) != len(corpus):
        print(f"  [{label}] WARNING: text/corpus length mismatch")
    return dictionary, corpus, texts


# ---------------------------------------------------------------------------
# Training + coherence for one (corpus, seed) pair
# ---------------------------------------------------------------------------
def train_and_score(
    dictionary: Dictionary,
    corpus: list,
    texts: list[list[str]],
    seed: int,
    label: str,
) -> tuple[float, float]:
    t0 = time.perf_counter()
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=K,
        random_state=seed,
        **LDA_PARAMS_BASE,
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
    print(f"    [{label}] seed={seed:>4}  c_v={c_v:.4f}  "
          f"train={train_sec:.1f}s  coh={coh_sec:.1f}s")
    return c_v, train_sec + coh_sec


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------
def write_summary_md(
    df_rows: pd.DataFrame,
    full_mean: float,
    full_sd: float,
    base_mean: float,
    base_sd: float,
    delta_mean: float,
    delta_sd: float,
    consistent: bool,
) -> None:
    lines: list[str] = []
    lines.append("# Multi-seed Robustness — Full Pipeline vs. Baseline (k = 25)")
    lines.append("")
    lines.append(
        "Both models are trained at k = 25 with identical frozen LDA "
        "hyperparameters (lda/specs/lda_params.md); only `random_state` "
        "varies. The question this script answers: is the coherence gap "
        "reported in script 09 (single-seed) robust across seeds, or an "
        "artefact of the seed used there?"
    )
    lines.append("")
    lines.append(f"Seeds tested: {SEEDS}")
    lines.append("")

    # Per-seed table
    lines.append("## Per-seed results")
    lines.append("")
    lines.append("| seed | full pipeline c_v | baseline c_v | delta (full − baseline) |")
    lines.append("|-----:|------------------:|-------------:|------------------------:|")
    for _, row in df_rows.iterrows():
        lines.append(
            f"| {int(row['seed'])} | {row['full_pipeline_cv']:.4f} | "
            f"{row['baseline_cv']:.4f} | {row['delta']:+.4f} |"
        )
    lines.append("")

    # Aggregate
    lines.append("## Aggregate (n = {} seeds)".format(len(df_rows)))
    lines.append("")
    lines.append("| quantity | mean | SD |")
    lines.append("|:---------|-----:|---:|")
    lines.append(f"| full pipeline c_v | {full_mean:.4f} | {full_sd:.4f} |")
    lines.append(f"| baseline c_v | {base_mean:.4f} | {base_sd:.4f} |")
    lines.append(f"| delta (full − baseline) | {delta_mean:+.4f} | {delta_sd:.4f} |")
    lines.append("")

    # Consistency
    n_full_wins = int((df_rows["delta"] > 0).sum())
    n_total = len(df_rows)
    verdict = "Gap is robust" if consistent else "Gap is inconsistent"
    lines.append("## Consistency")
    lines.append("")
    lines.append(
        f"Full pipeline beat baseline at **{n_full_wins} of {n_total}** seeds. "
        f"Verdict: **{verdict}**."
    )
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    if consistent:
        lines.append(
            f"The full pipeline scores higher c_v than the baseline at every "
            f"seed tested, with a mean gap of {delta_mean:+.4f} c_v "
            f"(SD {delta_sd:.4f}). The within-seed advantage is stable: "
            f"per-seed deltas range from {df_rows['delta'].min():+.4f} to "
            f"{df_rows['delta'].max():+.4f}. The single-seed result reported "
            f"in script 09 is therefore not an artefact of `random_state=42` — "
            f"the archive-specific preprocessing produces a measurable "
            f"coherence improvement that survives reseeding. This addresses "
            f"the single-seed limitation flagged in the baseline comparison."
        )
    else:
        lines.append(
            f"The full pipeline does NOT consistently beat the baseline across "
            f"seeds. Of {n_total} seeds, the full pipeline won at {n_full_wins}. "
            f"Mean delta is {delta_mean:+.4f} c_v (SD {delta_sd:.4f}); per-seed "
            f"deltas range from {df_rows['delta'].min():+.4f} to "
            f"{df_rows['delta'].max():+.4f}. The single-seed gap reported in "
            f"script 09 should not be read as evidence that the full pipeline "
            f"strictly improves coherence — c_v differences at this scale "
            f"appear to be within seed-driven noise. The contamination and "
            f"noise-topic metrics from script 09 remain the more reliable "
            f"signals for RQ2."
        )
    lines.append("")

    OUT_SUMMARY_MD.write_text("\n".join(lines))
    print(f"Saved → {OUT_SUMMARY_MD.relative_to(LDA_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(SEP)
    print("10 — Multi-seed Robustness  (k = 25, Stage 4, RQ2 follow-up)")
    print(SEP)
    print(f"  Seeds: {SEEDS}")
    print(f"  Hyperparameters (frozen except seed): alpha=auto, eta=auto, "
          f"passes=10,\n                                        iterations=400, "
          f"chunksize=2000, update_every=3")

    # ------------------------------------------------------------------
    # 1. Load both corpora once
    # ------------------------------------------------------------------
    print(f"\n{SUBSEP}")
    print("Loading full-pipeline corpus ...")
    print(SUBSEP)
    full_dict, full_corpus, full_texts = load_corpus_bundle(
        FULL_DICT_PATH, FULL_CORPUS_PATH, FULL_DOCS_PATH, "full"
    )

    print(f"\n{SUBSEP}")
    print("Loading baseline corpus ...")
    print(SUBSEP)
    base_dict, base_corpus, base_texts = load_corpus_bundle(
        BASE_DICT_PATH, BASE_CORPUS_PATH, BASE_DOCS_PATH, "base"
    )

    # ------------------------------------------------------------------
    # 2. Sweep seeds
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("Training across seeds")
    print(SEP)

    rows: list[dict] = []
    for seed in tqdm(SEEDS, desc="seed sweep"):
        print(f"\n  seed = {seed}")
        cv_full, _ = train_and_score(
            full_dict, full_corpus, full_texts, seed, "full"
        )
        cv_base, _ = train_and_score(
            base_dict, base_corpus, base_texts, seed, "base"
        )
        delta = cv_full - cv_base
        rows.append({
            "seed": seed,
            "full_pipeline_cv": round(cv_full, 6),
            "baseline_cv": round(cv_base, 6),
            "delta": round(delta, 6),
        })
        print(f"    delta (full − baseline) = {delta:+.4f}")

    # ------------------------------------------------------------------
    # 3. Save CSV + aggregate
    # ------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV.relative_to(LDA_DIR)}")

    full_arr = df["full_pipeline_cv"].to_numpy(dtype=float)
    base_arr = df["baseline_cv"].to_numpy(dtype=float)
    delta_arr = df["delta"].to_numpy(dtype=float)

    full_mean, full_sd = float(full_arr.mean()), float(full_arr.std(ddof=1))
    base_mean, base_sd = float(base_arr.mean()), float(base_arr.std(ddof=1))
    delta_mean, delta_sd = float(delta_arr.mean()), float(delta_arr.std(ddof=1))

    consistent = bool(np.all(delta_arr > 0))

    write_summary_md(
        df,
        full_mean, full_sd,
        base_mean, base_sd,
        delta_mean, delta_sd,
        consistent,
    )

    # ------------------------------------------------------------------
    # 4. Stdout summary
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("MULTI-SEED ROBUSTNESS — SUMMARY")
    print(SEP)
    print(f"  {'seed':>6} {'full':>10} {'baseline':>10} {'delta':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for _, r in df.iterrows():
        print(f"  {int(r['seed']):>6} {r['full_pipeline_cv']:>10.4f} "
              f"{r['baseline_cv']:>10.4f} {r['delta']:>+10.4f}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'mean':>6} {full_mean:>10.4f} {base_mean:>10.4f} {delta_mean:>+10.4f}")
    print(f"  {'SD':>6} {full_sd:>10.4f} {base_sd:>10.4f} {delta_sd:>10.4f}")

    n_full_wins = int((delta_arr > 0).sum())
    verdict = "Gap is robust" if consistent else "Gap is inconsistent"
    print(f"\n  Full pipeline > baseline at {n_full_wins}/{len(SEEDS)} seeds.")
    print(f"  Mean delta: {delta_mean:+.4f}  (SD {delta_sd:.4f})")
    print(f"  Verdict: {verdict}")
    print(SEP)


if __name__ == "__main__":
    main()
