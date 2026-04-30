"""
05 — Cold War Relevance (First-Pass Heuristic)
==============================================
Input:  lda/reports/topics_k{K}_top_words.csv   (from script 04)
        lda/specs/cold_war_vocabulary.yml       (5 categories, 30 terms)
Output: lda/reports/cold_war_relevance_k{K}.csv
        lda/reports/cold_war_relevance_k{K}_summary.md

For a given k, compute per-topic Cold War relevance scores using the
pre-registered vocabulary. This is a STRUCTURED FIRST-PASS HEURISTIC,
NOT AN OBJECTIVE CLASSIFIER — it surfaces candidate Cold-War-core and
Cold-War-adjacent topics for the analyst to confirm via the manual
interpretability rubric (lda/specs/interpretability_rubric.md).

Scoring:
  - Per category: fraction of category reference terms that appear
    in the topic's top-20 words (in [0, 1]).
  - Overall: unweighted mean of the five category scores.

Tier assignment (by overall score):
  - Cold-War-core     : overall >= 0.15
  - Cold-War-adjacent : 0.05 <= overall < 0.15
  - Low-overall        : overall < 0.05
    (labelled "Administrative" only after the analyst confirms the
    top words suggest admin content — this script does NOT auto-label.)

Methodological caveat
---------------------
The vocabulary term "source" (tradecraft) is a known limitation
(see deferred_items.md #10): it is also a generic English word and
will inflate the tradecraft category in non-intelligence contexts.
This caveat is acknowledged in the thesis writeup; the raw per-term
match list in the CSV allows readers to inspect how much of the
tradecraft score comes from "source" alone.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LDA_DIR = SCRIPT_DIR.parent

VOCAB_YAML = LDA_DIR / "specs" / "cold_war_vocabulary.yml"
REPORTS_DIR = LDA_DIR / "reports"

TOP_N_WORDS = 20

CATEGORIES = ["geopolitical", "agency", "tradecraft", "diplomatic", "named_actor"]

# Tier thresholds on overall score
TIER_CORE = 0.15
TIER_ADJACENT = 0.05

SEPARATOR = "=" * 60


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_vocabulary() -> dict[str, list[str]]:
    """Return {category: [term, ...]} with lowercased terms."""
    with open(VOCAB_YAML, "r") as f:
        data = yaml.safe_load(f)
    vocab: dict[str, list[str]] = {}
    for category in CATEGORIES:
        if category not in data:
            raise KeyError(f"Category '{category}' missing from {VOCAB_YAML.name}")
        vocab[category] = [entry["term"].lower() for entry in data[category]]
    return vocab


def load_top_words(k: int) -> pd.DataFrame:
    path = REPORTS_DIR / f"topics_k{k}_top_words.csv"
    if not path.exists():
        print(f"ERROR: {path} not found — run script 04 for k={k} first.",
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    if len(df) != k:
        print(f"WARNING: expected {k} topics in {path.name}, got {len(df)}",
              file=sys.stderr)
    return df


def topic_top_words(row: pd.Series) -> list[str]:
    """Extract lowercased word_1..word_20 from a row of topics_k*_top_words.csv."""
    words = []
    for i in range(1, TOP_N_WORDS + 1):
        cell = row.get(f"word_{i}")
        if isinstance(cell, str) and cell.strip():
            words.append(cell.strip().lower())
    return words


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_topic(
    top_words: list[str],
    vocab: dict[str, list[str]],
) -> tuple[dict[str, float], dict[str, list[str]], float]:
    """Return (per-category scores, per-category matched terms, overall)."""
    word_set = set(top_words)
    scores: dict[str, float] = {}
    matches: dict[str, list[str]] = {}
    for category, terms in vocab.items():
        hits = [t for t in terms if t in word_set]
        matches[category] = hits
        scores[category] = len(hits) / len(terms) if terms else 0.0
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    return scores, matches, overall


def format_matched_terms(matches: dict[str, list[str]]) -> str:
    """'oswald (named_actor); cia (agency); ...' — empty categories omitted."""
    parts = []
    for category in CATEGORIES:
        for term in matches.get(category, []):
            parts.append(f"{term} ({category})")
    return "; ".join(parts)


def assign_tier(overall: float) -> str:
    if overall >= TIER_CORE:
        return "Cold-War-core"
    if overall >= TIER_ADJACENT:
        return "Cold-War-adjacent"
    return "Low-overall"


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------
def build_relevance_table(
    df_words: pd.DataFrame,
    vocab: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    for _, row in df_words.iterrows():
        topic_id = int(row["topic_id"])
        top_words = topic_top_words(row)
        scores, matches, overall = score_topic(top_words, vocab)
        rows.append({
            "topic_id": topic_id,
            **{c: round(scores[c], 4) for c in CATEGORIES},
            "overall": round(overall, 4),
            "matched_terms": format_matched_terms(matches),
        })
    return pd.DataFrame(rows)


def category_leaders(df_relevance: pd.DataFrame, n: int = 3) -> dict[str, list[int]]:
    """Topic IDs with the highest score in each category (ties broken by topic_id)."""
    leaders: dict[str, list[int]] = {}
    for category in CATEGORIES:
        ranked = (
            df_relevance[df_relevance[category] > 0]
            .sort_values([category, "topic_id"], ascending=[False, True])
        )
        leaders[category] = ranked["topic_id"].head(n).tolist()
    return leaders


def build_summary_markdown(
    df_relevance: pd.DataFrame,
    k: int,
) -> str:
    ranked = df_relevance.sort_values(["overall", "topic_id"], ascending=[False, True])

    lines: list[str] = []
    lines.append(f"# Cold War Relevance — k = {k}")
    lines.append("")
    lines.append("First-pass heuristic scores from `lda/scripts/05_cold_war_relevance.py`. "
                 "Scores are fractions of the pre-registered Cold War vocabulary "
                 "(`lda/specs/cold_war_vocabulary.yml`) matched in each topic's top-20 "
                 "words. **Not an objective classifier** — use together with the manual "
                 "interpretability rubric.")
    lines.append("")
    lines.append(f"- Cold-War-core       : overall ≥ {TIER_CORE:.2f}")
    lines.append(f"- Cold-War-adjacent   : {TIER_ADJACENT:.2f} ≤ overall < {TIER_CORE:.2f}")
    lines.append(f"- Low-overall         : overall < {TIER_ADJACENT:.2f} "
                 "(analyst assigns an 'Administrative' label only after "
                 "confirming top-word content)")
    lines.append("")

    # Ranked table
    lines.append("## Ranked topics (highest overall first)")
    lines.append("")
    lines.append("| rank | topic_id | overall | tier | geo | agency | trade | diplo | actor | matched |")
    lines.append("|-----:|---------:|--------:|:-----|----:|-------:|------:|------:|------:|:--------|")
    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        tier = assign_tier(row["overall"])
        matched = row["matched_terms"] or "—"
        lines.append(
            f"| {rank} | {int(row['topic_id'])} | {row['overall']:.3f} | {tier} "
            f"| {row['geopolitical']:.2f} | {row['agency']:.2f} "
            f"| {row['tradecraft']:.2f} | {row['diplomatic']:.2f} "
            f"| {row['named_actor']:.2f} | {matched} |"
        )
    lines.append("")

    # Tier counts
    tier_counts = ranked["overall"].apply(assign_tier).value_counts().to_dict()
    lines.append("## Tier counts")
    lines.append("")
    for tier in ("Cold-War-core", "Cold-War-adjacent", "Low-overall"):
        lines.append(f"- {tier}: {tier_counts.get(tier, 0)}")
    lines.append("")

    # Category leaders
    leaders = category_leaders(df_relevance, n=3)
    lines.append("## Category-level breakdown")
    lines.append("")
    lines.append("Top-3 topic IDs per category (non-zero scores only, ties broken by topic_id):")
    lines.append("")
    for category in CATEGORIES:
        ids = leaders[category]
        if not ids:
            lines.append(f"- **{category}** : no topic scored above zero")
        else:
            lines.append(f"- **{category}** : {', '.join(str(i) for i in ids)}")
    lines.append("")

    # Caveat
    lines.append("## Methodological caveat")
    lines.append("")
    lines.append("The tradecraft term `source` is also a generic English word "
                 "(deferred item #10). Inspect `matched_terms` in the CSV to see how "
                 "much of a topic's tradecraft score is driven by `source` alone before "
                 "drawing interpretive conclusions.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cold War relevance scores for an LDA model (first-pass heuristic)"
    )
    parser.add_argument("--k", type=int, required=True,
                        help="Number of topics (top-words CSV must exist under lda/reports/)")
    args = parser.parse_args()
    k = args.k

    print(SEPARATOR)
    print(f"05 — Cold War Relevance  (k = {k})")
    print(SEPARATOR)
    print("NOTE: structured first-pass heuristic, NOT an objective classifier.")
    print("      Use with the manual interpretability rubric before drawing conclusions.")

    print("\nLoading vocabulary ...")
    vocab = load_vocabulary()
    total_terms = sum(len(v) for v in vocab.values())
    print(f"  Categories: {len(vocab)}  |  Total terms: {total_terms}")
    for category in CATEGORIES:
        print(f"    {category:<13} : {len(vocab[category])} terms")

    print(f"\nLoading top-words table for k = {k} ...")
    df_words = load_top_words(k)
    print(f"  Topics: {len(df_words)}")

    print("\nScoring topics ...")
    df_relevance = build_relevance_table(df_words, vocab)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / f"cold_war_relevance_k{k}.csv"
    df_relevance.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path.relative_to(LDA_DIR)}")

    md_path = REPORTS_DIR / f"cold_war_relevance_k{k}_summary.md"
    md_path.write_text(build_summary_markdown(df_relevance, k))
    print(f"  Saved → {md_path.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # Per-topic stdout
    # ------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print(f"PER-TOPIC SCORES  (k = {k}, sorted by overall desc)")
    print(SEPARATOR)
    ranked = df_relevance.sort_values(["overall", "topic_id"], ascending=[False, True])
    for _, row in ranked.iterrows():
        tid = int(row["topic_id"])
        overall = row["overall"]
        tier = assign_tier(overall)
        matched = row["matched_terms"] or "(no matches)"
        print(f"  topic {tid:>3}  overall={overall:.3f}  [{tier:<18}]  {matched}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    tier_counts = ranked["overall"].apply(assign_tier).value_counts().to_dict()
    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  k                        : {k}")
    print(f"  Cold-War-core (>= {TIER_CORE:.2f}) : {tier_counts.get('Cold-War-core', 0)}")
    print(f"  Cold-War-adjacent        : {tier_counts.get('Cold-War-adjacent', 0)}")
    print(f"  Low-overall (< {TIER_ADJACENT:.2f})    : {tier_counts.get('Low-overall', 0)}")
    print()
    print("  REMINDER: overall < 0.05 topics are 'Low-overall' here — do NOT")
    print("  auto-label them 'Administrative'. Confirm with the manual")
    print("  interpretability rubric before assigning tier labels in the thesis.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
