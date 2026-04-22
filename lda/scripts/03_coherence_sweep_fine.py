"""
03 — Fine Coherence Sweep
===========================
Input:  lda/outputs/dictionary.gensim
        lda/outputs/corpus.mm
        pipeline/phase6/data/documents_final.csv  (for c_v coherence)
        lda/reports/coherence_broad.csv           (verification)
Output: lda/outputs/models/lda_k{k}.gensim  (one per k)
        lda/reports/coherence_fine.csv
        lda/reports/coherence_fine.png
        lda/reports/coherence_fine_summary.md

Unit-resolution LDA coherence sweep across k ∈ [18, 40], chosen based
on the broad sweep's coherence plateau (k=20..40 with local peaks at
k=20 and k=35).  Reuses the broad-sweep models for k ∈ {20, 25, 30,
35, 40}; trains 18 new models for the intervening unit-resolution k
values.  Identifies the final k value for Stage 2 topic inspection.

Hyperparameters are identical to the broad sweep (see
lda/specs/lda_params.md).
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

BROAD_CSV_PATH = LDA_DIR / "reports" / "coherence_broad.csv"
CSV_PATH = LDA_DIR / "reports" / "coherence_fine.csv"
PNG_PATH = LDA_DIR / "reports" / "coherence_fine.png"
SUMMARY_PATH = LDA_DIR / "reports" / "coherence_fine_summary.md"

# ---------------------------------------------------------------------------
# k grid — unit resolution across the plateau identified by the broad sweep
# ---------------------------------------------------------------------------
K_VALUES = list(range(18, 41))  # 18, 19, ..., 40 inclusive

# k values that overlap with the broad sweep and should match exactly
OVERLAP_K = [20, 25, 30, 35, 40]
OVERLAP_TOLERANCE = 1e-4

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

# Tie-breaking rule for final k selection (stated explicitly per spec)
TIE_TOLERANCE = 0.02


def model_path(k: int) -> Path:
    return MODEL_DIR / f"lda_k{k}.gensim"


# ---------------------------------------------------------------------------
# Broad-sweep verification
# ---------------------------------------------------------------------------
def verify_broad_sweep_consistency(
    dictionary: Dictionary,
    corpus: list,
    texts: list,
) -> None:
    """Load each overlap-k model from disk, recompute c_v, and compare to the
    broad sweep's recorded values.  Halts if any value drifts beyond
    OVERLAP_TOLERANCE — this catches silent divergence between sweeps."""

    if not BROAD_CSV_PATH.exists():
        print("  (coherence_broad.csv not found — skipping verification.)")
        return

    print(f"Verifying consistency with {BROAD_CSV_PATH.name} ...")
    broad = pd.read_csv(BROAD_CSV_PATH)
    broad_by_k = dict(zip(broad["k"].astype(int), broad["coherence_c_v"]))

    failures = []
    for k in OVERLAP_K:
        if k not in broad_by_k:
            print(f"  k={k}: not in broad sweep (skipping)")
            continue
        mp = model_path(k)
        if not mp.exists():
            print(f"  k={k}: model file missing (skipping)")
            continue
        lda = LdaModel.load(str(mp))
        cm = CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        c_v_now = cm.get_coherence()
        c_v_broad = float(broad_by_k[k])
        diff = abs(c_v_now - c_v_broad)
        status = "OK" if diff <= OVERLAP_TOLERANCE else "DRIFT"
        print(f"  k={k:>2}: now={c_v_now:.6f}  broad={c_v_broad:.6f}  "
              f"Δ={diff:.2e}  [{status}]")
        if diff > OVERLAP_TOLERANCE:
            failures.append((k, c_v_now, c_v_broad, diff))

    if failures:
        print("\nERROR: coherence values drifted beyond tolerance "
              f"({OVERLAP_TOLERANCE}).  Halting.")
        for k, now, prev, diff in failures:
            print(f"  k={k}: now={now:.6f}  broad={prev:.6f}  Δ={diff:.2e}")
        raise SystemExit(1)

    print("  All overlap-k values match broad sweep within tolerance.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine LDA coherence sweep")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain all k values even if model files already exist",
    )
    args = parser.parse_args()

    separator = "=" * 60

    print(separator)
    print("03 — Fine Coherence Sweep")
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
    corpus = list(mm)

    print("Loading tokenised texts for c_v coherence ...")
    df = pd.read_csv(INPUT_CSV, usecols=[TEXT_COLUMN], low_memory=False)
    texts = df[TEXT_COLUMN].fillna("").str.split().tolist()
    print(f"  Texts loaded: {len(texts):,}")

    # ------------------------------------------------------------------
    # 2. Verify consistency with broad sweep (if present)
    # ------------------------------------------------------------------
    print()
    verify_broad_sweep_consistency(dictionary, corpus, texts)

    # ------------------------------------------------------------------
    # 3. Sweep
    # ------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for k in tqdm(K_VALUES, desc="k sweep"):
        mp = model_path(k)

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
    # 4. Save CSV
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(results)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(CSV_PATH, index=False)
    print(f"\nSaved → {CSV_PATH.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 5. Plot (fine curve + broad-sweep overlay)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_results["k"], df_results["coherence_c_v"],
            marker="o", label="Fine sweep (unit k)")
    if BROAD_CSV_PATH.exists():
        broad = pd.read_csv(BROAD_CSV_PATH)
        overlap = broad[broad["k"].isin(df_results["k"])]
        ax.scatter(
            overlap["k"],
            overlap["coherence_c_v"],
            marker="x", s=80, color="red", zorder=5,
            label="Broad sweep overlap (verification)",
        )
    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel("Coherence (c_v)")
    ax.set_title("Fine coherence sweep (c_v)")
    ax.set_xticks(df_results["k"])
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved → {PNG_PATH.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 6. Top-3 and final-k recommendation
    # ------------------------------------------------------------------
    top3 = df_results.nlargest(3, "coherence_c_v").reset_index(drop=True)
    best_k = int(top3.loc[0, "k"])
    best_cv = float(top3.loc[0, "coherence_c_v"])

    # Within-tolerance candidates (for tie-breaking note)
    tied = df_results[
        df_results["coherence_c_v"] >= (best_cv - TIE_TOLERANCE)
    ].sort_values("coherence_c_v", ascending=False)
    tied_ks = [int(x) for x in tied["k"].tolist()]

    lines = [
        "# Fine Coherence Sweep — Summary",
        "",
        f"Range swept: k ∈ [{min(K_VALUES)}, {max(K_VALUES)}] at unit resolution.",
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
    lines.append("## Final-k recommendation")
    lines.append("")
    lines.append(
        f"Primary signal: c_v coherence.  Top c_v = {best_cv:.4f} at k = {best_k}.  "
        f"Tie-breaking rule: candidates within {TIE_TOLERANCE} c_v of the maximum "
        "are adjudicated by the interpretability rubric applied in script 04."
    )
    lines.append("")
    lines.append(f"Candidates within the {TIE_TOLERANCE} c_v band: "
                 f"k ∈ {{{', '.join(str(k) for k in tied_ks)}}}.")
    lines.append("")
    lines.append(
        "Inspect in script 04: top-20 words per topic, three representative "
        "documents per topic, and distinctiveness between candidates before "
        "confirming the final k."
    )

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved → {SUMMARY_PATH.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    top3_ks = set(int(k) for k in top3["k"].tolist())
    print(f"\n{separator}")
    print("SUMMARY")
    print(separator)
    for _, row in df_results.iterrows():
        flag = " ★" if int(row["k"]) in top3_ks else ""
        print(f"  k={int(row['k']):>2}  c_v={row['coherence_c_v']:.4f}  "
              f"log_perp={row['log_perplexity']:.4f}{flag}")
    print(f"\n  Top 3: k = {', '.join(str(k) for k in top3['k'].tolist())}")
    print(f"  Tie band (±{TIE_TOLERANCE}): k ∈ {{{', '.join(str(k) for k in tied_ks)}}}")
    print(f"  Recommended for inspection (script 04): k = {best_k}"
          f" (plus any tied candidates)")
    print(separator)


if __name__ == "__main__":
    main()
