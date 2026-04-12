"""
00 — Vocabulary Audit
======================
Input:  pipeline/phase6/data/documents_final.csv
        lda/outputs/dictionary.gensim  (filtered)
Output: lda/reports/vocab_audit.md

One-time diagnostic: identifies which terms were dropped by
filter_extremes(no_below=5, no_above=0.5) and samples them to confirm
the 89.8% drop is removing OCR noise rather than substantive vocabulary.
"""

import random
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LDA_DIR = SCRIPT_DIR.parent
REPO_ROOT = LDA_DIR.parent

INPUT_CSV = REPO_ROOT / "pipeline" / "phase6" / "data" / "documents_final.csv"
FILTERED_DICT_PATH = LDA_DIR / "outputs" / "dictionary.gensim"
REPORT_PATH = LDA_DIR / "reports" / "vocab_audit.md"

TEXT_COLUMN = "document_text_lemma"

# Thresholds (for reporting only — this script does NOT call filter_extremes)
NO_BELOW = 5
NO_ABOVE = 0.5


def main() -> None:
    separator = "=" * 60

    print(separator)
    print("00 — Vocabulary Audit")
    print(separator)

    # ------------------------------------------------------------------
    # 1. Build unfiltered dictionary from scratch
    # ------------------------------------------------------------------
    print("\nLoading documents ...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  Documents loaded: {len(df):,}")

    print("Tokenising by whitespace split ...")
    texts = df[TEXT_COLUMN].fillna("").str.split().tolist()

    print("Building unfiltered dictionary ...")
    dict_unfiltered = Dictionary(texts)
    vocab_unfiltered = len(dict_unfiltered)
    print(f"  Unfiltered vocabulary: {vocab_unfiltered:,}")

    # ------------------------------------------------------------------
    # 2. Load filtered dictionary
    # ------------------------------------------------------------------
    print(f"\nLoading filtered dictionary from {FILTERED_DICT_PATH.name} ...")
    dict_filtered = Dictionary.load(str(FILTERED_DICT_PATH))
    vocab_filtered = len(dict_filtered)
    print(f"  Filtered vocabulary: {vocab_filtered:,}")

    # ------------------------------------------------------------------
    # 3. Identify dropped terms
    # ------------------------------------------------------------------
    filtered_terms = set(dict_filtered.values())
    dropped = []
    for token_id, term in dict_unfiltered.items():
        if term not in filtered_terms:
            dropped.append({
                "term": term,
                "doc_freq": dict_unfiltered.dfs[token_id],
                "corpus_freq": dict_unfiltered.cfs[token_id],
            })

    total_dropped = len(dropped)
    print(f"\n  Total terms dropped: {total_dropped:,}")

    # ------------------------------------------------------------------
    # 4. Breakdown by doc_freq bucket
    # ------------------------------------------------------------------
    df_dropped = pd.DataFrame(dropped)

    n_df1 = int((df_dropped["doc_freq"] == 1).sum())
    n_df2_4 = int(df_dropped["doc_freq"].between(2, 4).sum())
    n_df5plus = int((df_dropped["doc_freq"] >= 5).sum())

    print(f"    doc_freq == 1:   {n_df1:>10,}  (hapax legomena)")
    print(f"    doc_freq 2–4:    {n_df2_4:>10,}  (below no_below=5)")
    print(f"    doc_freq >= 5:   {n_df5plus:>10,}  (dropped by no_above=0.5)")

    # ------------------------------------------------------------------
    # 5. Build sample tables
    # ------------------------------------------------------------------
    print("\nBuilding sample tables ...")

    # Table A: top 50 most frequent dropped terms by corpus frequency
    table_a = (
        df_dropped
        .sort_values("corpus_freq", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )

    # Table B: top 50 most frequent dropped terms with doc_freq 2–4
    table_b = (
        df_dropped[df_dropped["doc_freq"].between(2, 4)]
        .sort_values("corpus_freq", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )

    # Table C: random 100 hapax legomena
    hapax = df_dropped[df_dropped["doc_freq"] == 1]
    rng = random.Random(42)
    if len(hapax) > 100:
        sample_idx = rng.sample(range(len(hapax)), 100)
        table_c = hapax.iloc[sorted(sample_idx)].reset_index(drop=True)
    else:
        table_c = hapax.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6. Write report
    # ------------------------------------------------------------------
    print(f"Writing report → {REPORT_PATH.relative_to(LDA_DIR)}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # Header
    lines.append("# Vocabulary Audit Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Vocabulary before filter | {vocab_unfiltered:,} |")
    lines.append(f"| Vocabulary after filter | {vocab_filtered:,} |")
    lines.append(f"| Total terms dropped | {total_dropped:,} |")
    lines.append(f"| `no_below` | {NO_BELOW} |")
    lines.append(f"| `no_above` | {NO_ABOVE} |")
    lines.append("")
    lines.append("### Dropped terms by document frequency")
    lines.append("")
    lines.append(f"| Bucket | Count | Notes |")
    lines.append(f"|---|---|---|")
    lines.append(f"| doc_freq == 1 | {n_df1:,} | Hapax legomena |")
    lines.append(f"| doc_freq 2–4 | {n_df2_4:,} | Below no_below=5 cutoff |")
    lines.append(f"| doc_freq >= 5 | {n_df5plus:,} | Dropped by no_above=0.5 rule |")
    lines.append("")

    def write_table(title, table):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| term | doc_freq | corpus_freq |")
        lines.append("|---|---|---|")
        for _, row in table.iterrows():
            lines.append(f"| {row['term']} | {row['doc_freq']:,} | {row['corpus_freq']:,} |")
        lines.append("")

    write_table(
        "Table A — Top 50 most frequent dropped terms by corpus frequency",
        table_a,
    )
    write_table(
        "Table B — Top 50 most frequent dropped terms with doc_freq 2–4",
        table_b,
    )
    write_table(
        "Table C — Random sample of 100 dropped terms with doc_freq == 1 (seed=42)",
        table_c,
    )

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print("DONE")
    print(separator)
    print(f"  Unfiltered vocab:  {vocab_unfiltered:>10,}")
    print(f"  Filtered vocab:    {vocab_filtered:>10,}")
    print(f"  Dropped terms:     {total_dropped:>10,}")
    print(f"  Report: {REPORT_PATH.relative_to(LDA_DIR)}")
    print(separator)


if __name__ == "__main__":
    main()
