"""
Per-phase text change report
=============================
Reads the frozen CSV outputs of phases 1, 2, 3, 4, 5, and 6B and
measures how much each phase modifies the text of each document.
For each phase transition, matched documents are compared on:

    pct_word_change   — (words_after - words_before) / words_before
    pct_char_change   — (chars_after - chars_before) / chars_before
    jaccard_words     — |A ∩ B| / |A ∪ B| on lowercased alphabetic word sets

Mean ± std is reported per transition, along with `n_matched` and
`n_dropped`.  Documents are matched by (file_id, page_number) for
page-level transitions, and by `file_id` for the final aggregation
transition (Phase 5 → Phase 6B).

Phase 5 → Phase 6B aggregation: Phase 6B first groups pages into
documents (one per `file_id`) and then chunks documents longer than
5 000 tokens into `<file_id>_chunk_NNN` sub-documents.  For a fair
comparison, all Phase 5 pages belonging to one `file_id` are
concatenated (space-joined) and all Phase 6B rows sharing the same
base `file_id` (chunk suffix stripped) are likewise concatenated
before the metrics are computed.

Inputs (read-only):
    pipeline/phase1/JFK_Pages_Merged.csv
    pipeline/phase2/data/pages_phase2_cleaned.csv
    pipeline/phase3/data/pages_phase3_linefiltered.csv
    pipeline/phase4/data/pages_phase4_modeltext.csv
    pipeline/phase5/data/pages_for_modeling.csv
    pipeline/phase6/data/documents_final.csv

Output:
    pipeline/evaluation/change_report.json
    printed summary table

Reference: evaluation metrics adapted from Zimmermann et al. (2024),
"Approaches to improve preprocessing for Latent Dirichlet Allocation
topic modeling," Decision Support Systems.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_DIR.parent

P1_CSV = PIPELINE_DIR / "phase1" / "JFK_Pages_Merged.csv"
P2_CSV = PIPELINE_DIR / "phase2" / "data" / "pages_phase2_cleaned.csv"
P3_CSV = PIPELINE_DIR / "phase3" / "data" / "pages_phase3_linefiltered.csv"
P4_CSV = PIPELINE_DIR / "phase4" / "data" / "pages_phase4_modeltext.csv"
P5_CSV = PIPELINE_DIR / "phase5" / "data" / "pages_for_modeling.csv"
P6_CSV = PIPELINE_DIR / "phase6" / "data" / "documents_final.csv"

OUTPUT_JSON = SCRIPT_DIR / "change_report.json"

# ---------------------------------------------------------------------------
# Column bindings (verified against CSV headers)
# ---------------------------------------------------------------------------
TEXT_COLUMNS = {
    "phase1": "content",
    "phase2": "content_clean_ocr",
    "phase3": "content_clean_lines",
    "phase4": "content_model_lemma",
    "phase5": "content_model_lemma",
    "phase6": "document_text_lemma",
}

WORD_RE = re.compile(r"[a-z]+")
CHUNK_SUFFIX_RE = re.compile(r"_chunk_\d+$")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def word_set(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower())) if text else set()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def pct_change(before: int, after: int) -> float:
    if before == 0:
        return 0.0 if after == 0 else float("inf")
    return (after - before) / before


def metrics_for_pair(text_before: str, text_after: str) -> dict:
    text_before = text_before or ""
    text_after = text_after or ""
    w_before = len(text_before.split())
    w_after = len(text_after.split())
    c_before = len(text_before)
    c_after = len(text_after)
    return {
        "pct_word_change": pct_change(w_before, w_after),
        "pct_char_change": pct_change(c_before, c_after),
        "jaccard_words": jaccard(word_set(text_before), word_set(text_after)),
    }


def aggregate(pairs: list[dict]) -> dict:
    if not pairs:
        return {"n_matched": 0}
    arr = {k: np.array([p[k] for p in pairs], dtype=float) for k in pairs[0]}
    # Replace inf with nan for stats
    out = {"n_matched": len(pairs)}
    for k, v in arr.items():
        finite = v[np.isfinite(v)]
        out[f"{k}_mean"] = float(finite.mean()) if finite.size else float("nan")
        out[f"{k}_std"] = float(finite.std(ddof=0)) if finite.size else float("nan")
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _read_csv_robust(csv_path: Path, usecols: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except pd.errors.ParserError:
        return pd.read_csv(csv_path, usecols=usecols, engine="python")


def load_page(csv_path: Path, text_col: str) -> pd.DataFrame:
    df = _read_csv_robust(csv_path, ["file_id", "page_number", text_col])
    df[text_col] = df[text_col].fillna("")
    return df


def merge_pages(a: pd.DataFrame, col_a: str, b: pd.DataFrame, col_b: str) -> pd.DataFrame:
    merged = a.merge(
        b[["file_id", "page_number", col_b]],
        on=["file_id", "page_number"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    return merged


def compute_page_transition(
    df_before: pd.DataFrame,
    col_before: str,
    df_after: pd.DataFrame,
    col_after: str,
    desc: str,
) -> dict:
    left = df_before[["file_id", "page_number", col_before]].rename(
        columns={col_before: "__before"})
    right = df_after[["file_id", "page_number", col_after]].rename(
        columns={col_after: "__after"})
    merged = left.merge(right, on=["file_id", "page_number"], how="inner")
    n_dropped = len(df_before) - len(merged)

    pairs = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc=desc, unit="page"):
        pairs.append(metrics_for_pair(row["__before"], row["__after"]))

    result = aggregate(pairs)
    result["n_before"] = int(len(df_before))
    result["n_after"] = int(len(df_after))
    result["n_dropped"] = int(n_dropped)
    return result


def strip_chunk_suffix(fid: str) -> str:
    return CHUNK_SUFFIX_RE.sub("", fid)


def compute_aggregation_transition(
    df_pages: pd.DataFrame,
    page_text_col: str,
    df_docs: pd.DataFrame,
    doc_text_col: str,
    desc: str,
) -> dict:
    # Concatenate page-level text by file_id
    pages_agg = (
        df_pages.sort_values(["file_id", "page_number"])
        .groupby("file_id")[page_text_col]
        .apply(lambda s: " ".join(s.astype(str)))
        .rename("text_before")
        .reset_index()
    )

    # Concatenate document-level text by base file_id (strip chunk suffix)
    df_docs = df_docs.copy()
    df_docs["base_file_id"] = df_docs["file_id"].apply(strip_chunk_suffix)
    docs_agg = (
        df_docs.sort_values(["base_file_id", "file_id"])
        .groupby("base_file_id")[doc_text_col]
        .apply(lambda s: " ".join(s.astype(str)))
        .rename("text_after")
        .reset_index()
        .rename(columns={"base_file_id": "file_id"})
    )

    merged = pages_agg.merge(docs_agg, on="file_id", how="inner")
    n_dropped = len(pages_agg) - len(merged)

    pairs = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc=desc, unit="doc"):
        pairs.append(metrics_for_pair(row["text_before"], row["text_after"]))

    result = aggregate(pairs)
    result["n_before"] = int(len(pages_agg))
    result["n_after"] = int(len(docs_agg))
    result["n_dropped"] = int(n_dropped)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    separator = "=" * 70
    print(separator)
    print("Per-phase text change report")
    print(separator)

    print("\nLoading CSVs ...")
    df1 = load_page(P1_CSV, TEXT_COLUMNS["phase1"])
    print(f"  phase1: {len(df1):,} pages")
    df2 = load_page(P2_CSV, TEXT_COLUMNS["phase2"])
    print(f"  phase2: {len(df2):,} pages")
    df3 = load_page(P3_CSV, TEXT_COLUMNS["phase3"])
    print(f"  phase3: {len(df3):,} pages")
    df4 = load_page(P4_CSV, TEXT_COLUMNS["phase4"])
    print(f"  phase4: {len(df4):,} pages")
    df5 = load_page(P5_CSV, TEXT_COLUMNS["phase5"])
    print(f"  phase5: {len(df5):,} pages")
    df6 = _read_csv_robust(P6_CSV, ["file_id", TEXT_COLUMNS["phase6"]])
    df6[TEXT_COLUMNS["phase6"]] = df6[TEXT_COLUMNS["phase6"]].fillna("")
    print(f"  phase6: {len(df6):,} documents")

    print("\nComputing transitions ...")
    transitions: dict[str, dict] = {}

    transitions["raw_to_phase2"] = compute_page_transition(
        df1, TEXT_COLUMNS["phase1"], df2, TEXT_COLUMNS["phase2"],
        desc="Raw → Phase 2",
    )
    transitions["phase2_to_phase3"] = compute_page_transition(
        df2, TEXT_COLUMNS["phase2"], df3, TEXT_COLUMNS["phase3"],
        desc="Phase 2 → Phase 3",
    )
    transitions["phase3_to_phase4"] = compute_page_transition(
        df3, TEXT_COLUMNS["phase3"], df4, TEXT_COLUMNS["phase4"],
        desc="Phase 3 → Phase 4",
    )
    transitions["phase4_to_phase5"] = compute_page_transition(
        df4, TEXT_COLUMNS["phase4"], df5, TEXT_COLUMNS["phase5"],
        desc="Phase 4 → Phase 5",
    )
    transitions["phase5_to_phase6b"] = compute_aggregation_transition(
        df5, TEXT_COLUMNS["phase5"], df6, TEXT_COLUMNS["phase6"],
        desc="Phase 5 → Phase 6B",
    )

    # --------------------------------------------------------------
    # Save and summarise
    # --------------------------------------------------------------
    report = {
        "columns_used": TEXT_COLUMNS,
        "transitions": transitions,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {OUTPUT_JSON.relative_to(REPO_ROOT)}")

    # Summary table
    print(f"\n{separator}")
    print("SUMMARY")
    print(separator)
    header = f"{'transition':<22} {'n_match':>8} {'drop':>6} " \
             f"{'Δwords':>10} {'Δchars':>10} {'Jaccard':>10}"
    print(header)
    print("-" * len(header))
    for name, r in transitions.items():
        dw = r.get("pct_word_change_mean", float("nan"))
        dc = r.get("pct_char_change_mean", float("nan"))
        jac = r.get("jaccard_words_mean", float("nan"))
        print(f"{name:<22} {r['n_matched']:>8,} {r['n_dropped']:>6,} "
              f"{dw:>9.2%} {dc:>9.2%} {jac:>9.4f}")
    print(separator)


if __name__ == "__main__":
    main()
