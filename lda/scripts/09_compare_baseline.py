"""
09 — Compare Baseline vs. Full Pipeline  (Stage 4, RQ2)
=========================================================
Input:  lda/reports/topics_k25_top_words.csv          (full pipeline)
        lda/reports/coherence_fine.csv                (full pipeline c_v)
        lda/reports/baseline_topics_top_words.csv     (baseline)
        lda/reports/baseline_coherence.csv            (baseline c_v)
Output: lda/reports/baseline_vs_full.csv
        lda/reports/baseline_vs_full_topics.md
        lda/reports/baseline_vs_full_summary.md

Compares the full-pipeline and minimal-baseline LDA models at k = 25
on three dimensions:

  1. c_v coherence — the headline RQ2 number.
  2. Metadata contamination — how many of the 26 Phase 6B blacklist
     tokens leak into each model's top-20 lists. The baseline never
     stripped them, so it should have markedly more hits.
  3. Topic interpretability — topics with >= 3 of their top-10 words
     drawn from a noise list (the 26 Phase 6B blacklist terms plus a
     short list of common archive boilerplate). These are topics an
     analyst would label "administrative noise".

The convention used throughout: delta = full_pipeline − baseline.
A positive delta on coherence means the full pipeline scores higher;
a negative delta on contamination / noise-topic counts means the full
pipeline is cleaner.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LDA_DIR = SCRIPT_DIR.parent
REPORTS_DIR = LDA_DIR / "reports"

FULL_TOPWORDS_CSV = REPORTS_DIR / "topics_k25_top_words.csv"
FULL_COHERENCE_CSV = REPORTS_DIR / "coherence_fine.csv"
BASELINE_TOPWORDS_CSV = REPORTS_DIR / "baseline_topics_top_words.csv"
BASELINE_COHERENCE_CSV = REPORTS_DIR / "baseline_coherence.csv"

OUT_CSV = REPORTS_DIR / "baseline_vs_full.csv"
OUT_TOPICS_MD = REPORTS_DIR / "baseline_vs_full_topics.md"
OUT_SUMMARY_MD = REPORTS_DIR / "baseline_vs_full_summary.md"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K = 25
TOP_N_WORDS = 20
TOP_N_FOR_NOISE = 10
NOISE_TOPIC_THRESHOLD = 3   # >= 3 of top-10 from noise list ⇒ flagged

# Phase 6B blacklist (26 terms — matches phase6b_modeling_prep.py exactly).
BLACKLIST = {
    "umbra", "noforn", "orcon", "wnintel", "moray", "limdis", "exdis",
    "nodis", "rybat", "typic", "slugs", "sensind", "mhfno", "iden",
    "docid", "nw", "decl", "drv", "originator", "cfr", "css",
    "ernment", "fpmr", "hcf", "sgswirl", "tud",
}

# Common archive boilerplate added on top of the blacklist for the
# noise-topic check. Listed in the task spec.
EXTRA_NOISE = {
    "page", "record", "document", "file", "copy", "report", "date",
    "note", "letter", "memorandum", "classification", "secret",
    "confidential", "unclassified",
    # "noforn" is already in BLACKLIST; included in the spec for emphasis
    # but kept de-duped here.
}

NOISE_TERMS = BLACKLIST | EXTRA_NOISE

SEP = "=" * 60
SUBSEP = "-" * 60


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_topwords(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ERROR: missing top-words file: {path}")
    return pd.read_csv(path)


def topic_top_words(row: pd.Series, n: int) -> list[str]:
    """Lowercased word_1..word_n for a single topic row."""
    out: list[str] = []
    for i in range(1, n + 1):
        cell = row.get(f"word_{i}")
        if isinstance(cell, str) and cell.strip():
            out.append(cell.strip().lower())
    return out


def load_coherence_full() -> float:
    df = pd.read_csv(FULL_COHERENCE_CSV)
    row = df.loc[df["k"].astype(int) == K]
    if row.empty:
        raise SystemExit(f"ERROR: k={K} not found in {FULL_COHERENCE_CSV.name}")
    return float(row.iloc[0]["coherence_c_v"])


def load_coherence_baseline() -> float:
    df = pd.read_csv(BASELINE_COHERENCE_CSV)
    row = df.loc[df["k"].astype(int) == K]
    if row.empty:
        raise SystemExit(f"ERROR: k={K} not found in {BASELINE_COHERENCE_CSV.name}")
    return float(row.iloc[0]["coherence_c_v"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metadata_contamination(df_words: pd.DataFrame) -> tuple[int, dict[str, int]]:
    """Count occurrences of any blacklist token across all topics' top-20
    words. Returns (total_hits, per_token_counts)."""
    per_token: dict[str, int] = {}
    total = 0
    for _, row in df_words.iterrows():
        for w in topic_top_words(row, TOP_N_WORDS):
            if w in BLACKLIST:
                per_token[w] = per_token.get(w, 0) + 1
                total += 1
    return total, per_token


def noise_topic_count(df_words: pd.DataFrame) -> tuple[int, list[int], list[int]]:
    """Topics where >= NOISE_TOPIC_THRESHOLD of the top-N_FOR_NOISE
    words are in NOISE_TERMS. Returns (count, flagged_topic_ids, hits_per_topic)."""
    flagged: list[int] = []
    hits_per_topic: list[int] = []
    for _, row in df_words.iterrows():
        words = topic_top_words(row, TOP_N_FOR_NOISE)
        hits = sum(1 for w in words if w in NOISE_TERMS)
        hits_per_topic.append(hits)
        if hits >= NOISE_TOPIC_THRESHOLD:
            flagged.append(int(row["topic_id"]))
    return len(flagged), flagged, hits_per_topic


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_metrics_csv(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV.relative_to(LDA_DIR)}")


def write_topics_md(
    df_full: pd.DataFrame,
    df_base: pd.DataFrame,
    flagged_full: set[int],
    flagged_base: set[int],
) -> None:
    lines: list[str] = []
    lines.append("# Baseline vs. Full Pipeline — Top-10 Words per Topic")
    lines.append("")
    lines.append(
        f"Side-by-side top-10 words for the {K} topics produced by the "
        "full preprocessing pipeline and by the minimal-preprocessing "
        "baseline, both at k = 25 and with identical LDA hyperparameters "
        "(lda/specs/lda_params.md). Topics flagged with **(noise)** had "
        f"≥ {NOISE_TOPIC_THRESHOLD} of their top-{TOP_N_FOR_NOISE} words in the "
        "blacklist + boilerplate noise list."
    )
    lines.append("")
    lines.append("Note: topic_id alignment is positional only — there is no "
                 "guarantee that topic 0 in one model corresponds to topic 0 "
                 "in the other. The two columns are listed by index for "
                 "convenience, not because they describe the same theme.")
    lines.append("")
    lines.append("| topic_id | full pipeline (top-10) | baseline (top-10) |")
    lines.append("|---------:|:-----------------------|:------------------|")

    full_by_id = {int(row["topic_id"]): topic_top_words(row, TOP_N_FOR_NOISE)
                  for _, row in df_full.iterrows()}
    base_by_id = {int(row["topic_id"]): topic_top_words(row, TOP_N_FOR_NOISE)
                  for _, row in df_base.iterrows()}

    for tid in range(K):
        full_w = full_by_id.get(tid, [])
        base_w = base_by_id.get(tid, [])
        full_str = ", ".join(full_w)
        base_str = ", ".join(base_w)
        if tid in flagged_full:
            full_str = f"**(noise)** {full_str}"
        if tid in flagged_base:
            base_str = f"**(noise)** {base_str}"
        lines.append(f"| {tid} | {full_str} | {base_str} |")
    lines.append("")
    OUT_TOPICS_MD.write_text("\n".join(lines))
    print(f"Saved → {OUT_TOPICS_MD.relative_to(LDA_DIR)}")


def write_summary_md(
    cv_full: float,
    cv_base: float,
    contam_full: int,
    contam_base: int,
    contam_full_terms: dict[str, int],
    contam_base_terms: dict[str, int],
    noise_count_full: int,
    noise_count_base: int,
    flagged_full: list[int],
    flagged_base: list[int],
) -> None:
    cv_delta = cv_full - cv_base
    contam_delta = contam_full - contam_base
    noise_delta = noise_count_full - noise_count_base

    lines: list[str] = []
    lines.append("# Baseline vs. Full Pipeline — Summary")
    lines.append("")
    lines.append(
        "Stage 4 (RQ2): does archive-specific cleaning measurably improve "
        "topic quality? Both models are trained at k = 25 with identical "
        "frozen LDA hyperparameters (lda/specs/lda_params.md). The only "
        "thing that varies is the preprocessing — Phases 2, 3, 4, 5, and "
        "6B are applied to the full-pipeline corpus and skipped for the "
        "baseline corpus."
    )
    lines.append("")
    lines.append(
        "Convention: `delta = full_pipeline − baseline`. Positive delta on "
        "coherence means the full pipeline scores higher; negative delta on "
        "contamination / noise-topic counts means the full pipeline is cleaner."
    )
    lines.append("")

    # Metric table
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | full pipeline | baseline | delta |")
    lines.append("|:-------|--------------:|---------:|------:|")
    lines.append(f"| c_v coherence (k=25) | {cv_full:.4f} | {cv_base:.4f} | {cv_delta:+.4f} |")
    lines.append(f"| metadata blacklist hits in top-20 | {contam_full} | {contam_base} | {contam_delta:+d} |")
    lines.append(f"| noise topics (≥{NOISE_TOPIC_THRESHOLD}/top-10 from noise list) | "
                 f"{noise_count_full} | {noise_count_base} | {noise_delta:+d} |")
    lines.append("")

    # Contamination breakdown
    lines.append("## Metadata contamination breakdown")
    lines.append("")
    lines.append("Counts the times each of the 26 Phase 6B blacklist tokens "
                 "appears anywhere in any topic's top-20 list. The baseline "
                 "never strips these tokens, so a non-trivial gap here is "
                 "direct evidence that Phase 6B is doing the work it was "
                 "designed for.")
    lines.append("")
    all_terms = sorted(set(contam_full_terms) | set(contam_base_terms))
    if all_terms:
        lines.append("| token | full pipeline | baseline |")
        lines.append("|:------|--------------:|---------:|")
        for t in all_terms:
            lines.append(
                f"| {t} | {contam_full_terms.get(t, 0)} | {contam_base_terms.get(t, 0)} |"
            )
    else:
        lines.append("_No blacklist tokens appeared in either model's top-20 lists._")
    lines.append("")

    # Noise topics
    lines.append("## Noise topics")
    lines.append("")
    lines.append(f"Topics with ≥ {NOISE_TOPIC_THRESHOLD} of the top-{TOP_N_FOR_NOISE} "
                 "words drawn from the blacklist + archive-boilerplate noise "
                 "list (see script 09 source for the exact list).")
    lines.append("")
    lines.append(f"- Full pipeline noise topics (n = {noise_count_full}): "
                 f"{flagged_full if flagged_full else '—'}")
    lines.append(f"- Baseline noise topics      (n = {noise_count_base}): "
                 f"{flagged_base if flagged_base else '—'}")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    cleaner_phrase = (
        "the full pipeline produces a measurably cleaner topic model"
        if (cv_delta > 0 and contam_delta < 0 and noise_delta < 0)
        else "the comparison is mixed and warrants a closer look"
    )
    lines.append(
        f"At identical hyperparameters and identical k, {cleaner_phrase}. "
        f"Coherence shifts by {cv_delta:+.4f} c_v from baseline to full pipeline. "
        f"Phase 6B blacklist tokens appear in the top-20 lists "
        f"{contam_full} times in the full-pipeline model versus "
        f"{contam_base} times in the baseline (delta {contam_delta:+d}). "
        f"Topics flagged as administrative noise (≥{NOISE_TOPIC_THRESHOLD} of "
        f"top-{TOP_N_FOR_NOISE} from the noise list) drop from "
        f"{noise_count_base} in the baseline to {noise_count_full} in the "
        f"full pipeline (delta {noise_delta:+d}). Because the only thing that "
        "varies between the two conditions is the archive-specific "
        "preprocessing, these differences are attributable to that pipeline. "
        "The contamination and noise-topic counts speak more directly than "
        "coherence to RQ2 — c_v is a relative quality measure, but a topic "
        "whose top words are `noforn`, `docid`, and `decl` is unambiguously "
        "less useful to a historian than one whose top words name agencies "
        "and operations."
    )
    lines.append("")

    OUT_SUMMARY_MD.write_text("\n".join(lines))
    print(f"Saved → {OUT_SUMMARY_MD.relative_to(LDA_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(SEP)
    print("09 — Compare Baseline vs. Full Pipeline  (Stage 4, RQ2)")
    print(SEP)
    print(f"  k = {K}, top-N = {TOP_N_WORDS}, noise threshold = "
          f"{NOISE_TOPIC_THRESHOLD}/top-{TOP_N_FOR_NOISE}")
    print(f"  Blacklist size: {len(BLACKLIST)}    "
          f"Noise list size (blacklist + boilerplate): {len(NOISE_TERMS)}")

    print("\nLoading inputs ...")
    df_full = load_topwords(FULL_TOPWORDS_CSV)
    df_base = load_topwords(BASELINE_TOPWORDS_CSV)
    cv_full = load_coherence_full()
    cv_base = load_coherence_baseline()
    print(f"  Full-pipeline topics    : {len(df_full)}  c_v = {cv_full:.4f}")
    print(f"  Baseline      topics    : {len(df_base)}  c_v = {cv_base:.4f}")

    # ------------------------------------------------------------------
    # Metric 2 — metadata contamination
    # ------------------------------------------------------------------
    print("\nCounting metadata contamination ...")
    contam_full, contam_full_terms = metadata_contamination(df_full)
    contam_base, contam_base_terms = metadata_contamination(df_base)
    print(f"  Full pipeline: {contam_full} blacklist hits across all top-20 "
          f"({len(contam_full_terms)} distinct tokens)")
    print(f"  Baseline     : {contam_base} blacklist hits across all top-20 "
          f"({len(contam_base_terms)} distinct tokens)")

    # ------------------------------------------------------------------
    # Metric 3 — noise topics
    # ------------------------------------------------------------------
    print("\nCounting noise topics ...")
    noise_count_full, flagged_full, _ = noise_topic_count(df_full)
    noise_count_base, flagged_base, _ = noise_topic_count(df_base)
    print(f"  Full pipeline: {noise_count_full} noise topics  "
          f"→ topic_ids {flagged_full or '—'}")
    print(f"  Baseline     : {noise_count_base} noise topics  "
          f"→ topic_ids {flagged_base or '—'}")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    print()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows = [
        {
            "metric": "c_v_coherence_k25",
            "full_pipeline": round(cv_full, 6),
            "baseline": round(cv_base, 6),
            "delta": round(cv_full - cv_base, 6),
        },
        {
            "metric": "metadata_blacklist_hits_top20",
            "full_pipeline": contam_full,
            "baseline": contam_base,
            "delta": contam_full - contam_base,
        },
        {
            "metric": "noise_topics_top10_ge3",
            "full_pipeline": noise_count_full,
            "baseline": noise_count_base,
            "delta": noise_count_full - noise_count_base,
        },
    ]
    write_metrics_csv(metric_rows)
    write_topics_md(df_full, df_base, set(flagged_full), set(flagged_base))
    write_summary_md(
        cv_full, cv_base,
        contam_full, contam_base,
        contam_full_terms, contam_base_terms,
        noise_count_full, noise_count_base,
        flagged_full, flagged_base,
    )

    # ------------------------------------------------------------------
    # Stdout comparison table
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("BASELINE vs. FULL PIPELINE — COMPARISON")
    print(SEP)
    print(f"  {'metric':<40} {'full':>10} {'baseline':>10} {'delta':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
    for r in metric_rows:
        full_v = r["full_pipeline"]
        base_v = r["baseline"]
        d = r["delta"]
        full_str = f"{full_v:.4f}" if isinstance(full_v, float) else f"{full_v:d}"
        base_str = f"{base_v:.4f}" if isinstance(base_v, float) else f"{base_v:d}"
        delta_str = f"{d:+.4f}" if isinstance(d, float) else f"{d:+d}"
        print(f"  {r['metric']:<40} {full_str:>10} {base_str:>10} {delta_str:>10}")

    # Per-token contamination breakdown
    if contam_full_terms or contam_base_terms:
        print(f"\n{SUBSEP}")
        print("Metadata contamination — per-token counts")
        print(SUBSEP)
        all_terms = sorted(set(contam_full_terms) | set(contam_base_terms))
        print(f"  {'token':<14} {'full':>6} {'baseline':>10}")
        print(f"  {'-'*14} {'-'*6} {'-'*10}")
        for t in all_terms:
            print(f"  {t:<14} {contam_full_terms.get(t, 0):>6} "
                  f"{contam_base_terms.get(t, 0):>10}")

    print(f"\n{SUBSEP}")
    print("Noise topics (≥3 of top-10 from noise list)")
    print(SUBSEP)
    print(f"  Full pipeline ({noise_count_full}): "
          f"{flagged_full if flagged_full else '—'}")
    print(f"  Baseline      ({noise_count_base}): "
          f"{flagged_base if flagged_base else '—'}")
    print(SEP)


if __name__ == "__main__":
    main()
