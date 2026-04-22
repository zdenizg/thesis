"""
02 — Broad Coherence Sweep
============================
Input:  lda/outputs/dictionary.gensim
        lda/outputs/corpus.mm
        pipeline/phase6/data/documents_final.csv  (for c_v coherence)
Output: lda/outputs/models/lda_k{k}.gensim  (one per k)
        lda/reports/coherence_broad.csv
        lda/reports/coherence_broad.png
        lda/reports/coherence_broad_summary.md

Trains LdaModel across a broad k grid, records c_v coherence,
log-perplexity, and runtime for each.  Identifies the most promising
region for the fine sweep in script 03.
"""

import argparse
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

DICT_PATH = LDA_DIR / "outputs" / "dictionary.gensim"
CORPUS_PATH = LDA_DIR / "outputs" / "corpus.mm"
INPUT_CSV = REPO_ROOT / "pipeline" / "phase6" / "data" / "documents_final.csv"
MODEL_DIR = LDA_DIR / "outputs" / "models"

CSV_PATH = LDA_DIR / "reports" / "coherence_broad.csv"
PNG_PATH = LDA_DIR / "reports" / "coherence_broad.png"
SUMMARY_PATH = LDA_DIR / "reports" / "coherence_broad_summary.md"

# ---------------------------------------------------------------------------
# k grid
# ---------------------------------------------------------------------------
K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]

# ---------------------------------------------------------------------------
# Frozen hyperparameters (from lda/specs/lda_params.md)
# ---------------------------------------------------------------------------
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

TEXT_COLUMN = "document_text_lemma"


def model_path(k: int) -> Path:
    return MODEL_DIR / f"lda_k{k}.gensim"


def main() -> None:
    parser = argparse.ArgumentParser(description="Broad LDA coherence sweep")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain all k values even if model files already exist",
    )
    args = parser.parse_args()

    separator = "=" * 60

    print(separator)
    print("02 — Broad Coherence Sweep")
    print(separator)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print("\nLoading dictionary ...")
    dictionary = Dictionary.load(str(DICT_PATH))
    print(f"  Vocabulary: {len(dictionary):,}")

    print("Loading corpus ...")
    mm = MmCorpus(str(CORPUS_PATH))
    print(f"  Documents: {mm.num_docs:,}")
    corpus = list(mm)  # materialise for multi-pass iteration

    print("Loading tokenised texts for c_v coherence ...")
    df = pd.read_csv(INPUT_CSV, usecols=[TEXT_COLUMN], low_memory=False)
    texts = df[TEXT_COLUMN].fillna("").str.split().tolist()
    print(f"  Texts loaded: {len(texts):,}")

    # ------------------------------------------------------------------
    # 2. Sweep
    # ------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for k in tqdm(K_VALUES, desc="k sweep"):
        mp = model_path(k)

        # Check for existing model
        if mp.exists() and not args.overwrite:
            print(f"\n  k={k:>2}: loading existing model (use --overwrite to retrain)")
            lda = LdaModel.load(str(mp))
            train_sec = 0.0
        else:
            print(f"\n  k={k:>2}: training ...")
            t0 = time.perf_counter()
            lda = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                **LDA_PARAMS,
            )
            train_sec = time.perf_counter() - t0
            lda.save(str(mp))
            print(f"         trained in {train_sec:.1f}s → {mp.name}")

        # Log fitted alpha / eta
        a = lda.alpha
        print(f"         alpha: min={a.min():.4f}  max={a.max():.4f}  "
              f"mean={a.mean():.4f}  (learned, {len(a)} topics)")
        eta_val = float(lda.eta.mean()) if hasattr(lda.eta, 'mean') else float(lda.eta)
        print(f"         eta:   {eta_val:.6f}")

        # Coherence
        t1 = time.perf_counter()
        cm = CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        c_v = cm.get_coherence()
        coh_sec = time.perf_counter() - t1

        # Perplexity
        log_perp = lda.log_perplexity(corpus)

        results.append({
            "k": k,
            "coherence_c_v": round(c_v, 6),
            "log_perplexity": round(log_perp, 4),
            "alpha_min": round(float(a.min()), 6),
            "alpha_max": round(float(a.max()), 6),
            "alpha_mean": round(float(a.mean()), 6),
            "eta": round(eta_val, 6),
            "train_seconds": round(train_sec, 1),
            "coherence_seconds": round(coh_sec, 1),
        })
        print(f"         c_v={c_v:.4f}  log_perp={log_perp:.4f}  "
              f"coh_time={coh_sec:.1f}s")

    # ------------------------------------------------------------------
    # 3. Save CSV
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(results)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(CSV_PATH, index=False)
    print(f"\nSaved → {CSV_PATH.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_results["k"], df_results["coherence_c_v"], marker="o")
    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel("Coherence (c_v)")
    ax.set_title("Broad coherence sweep (c_v)")
    ax.set_xticks(df_results["k"])
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved → {PNG_PATH.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 5. Top-3 and summary
    # ------------------------------------------------------------------
    top3 = df_results.nlargest(3, "coherence_c_v")

    k_vals = top3["k"].tolist()
    k_min = max(5, min(k_vals) - 5)
    k_max = max(k_vals) + 5
    recommendation = (
        f"Fine sweep recommended over k ∈ [{k_min}, {k_max}] "
        f"in steps of 1 or 2."
    )

    lines = [
        "# Broad Coherence Sweep — Summary",
        "",
        "## Top 3 k values by c_v coherence",
        "",
        "| Rank | k | c_v | log_perplexity |",
        "|---|---|---|---|",
    ]
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        lines.append(
            f"| {rank} | {int(row['k'])} | {row['coherence_c_v']:.4f} "
            f"| {row['log_perplexity']:.4f} |"
        )
    lines.append("")
    lines.append(f"## Recommendation")
    lines.append("")
    lines.append(recommendation)

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved → {SUMMARY_PATH.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("SUMMARY")
    print(separator)
    for _, row in df_results.iterrows():
        flag = " ★" if int(row["k"]) in k_vals else ""
        print(f"  k={int(row['k']):>2}  c_v={row['coherence_c_v']:.4f}  "
              f"log_perp={row['log_perplexity']:.4f}{flag}")
    print(f"\n  Top 3: k = {', '.join(str(v) for v in k_vals)}")
    print(f"  {recommendation}")
    print(separator)


if __name__ == "__main__":
    main()
