"""
Per-stage text cleanliness report
==================================
Reads the frozen CSV outputs of each pipeline stage and reports five
cleanliness metrics per stage, aggregated across all documents/pages
in that stage's CSV:

    dict_hit_rate     — fraction of alphabetic tokens found in the
                        system word list (/usr/share/dict/words,
                        falling back to NLTK's words corpus if the
                        system list is unavailable).
    type_token_ratio  — unique_types / total_tokens
    hapax_ratio       — fraction of types appearing exactly once
    alpha_char_ratio  — alphabetic characters / total characters
    mean_token_len    — average token character length

For page-level stages (0, 2, 3, 4, 5) metrics are computed per page
and averaged.  For the document-level stage (6B) metrics are computed
per document.  The report also prints a "cleaning gradient" — the
delta between stage 0 (raw) and stage 6B (final) for each metric.

Inputs (read-only):
    pipeline/phase1/JFK_Pages_Merged.csv          (stage 0, raw)
    pipeline/phase2/data/pages_phase2_cleaned.csv (stage 2)
    pipeline/phase3/data/pages_phase3_linefiltered.csv (stage 3)
    pipeline/phase4/data/pages_phase4_modeltext.csv    (stage 4)
    pipeline/phase5/data/pages_for_modeling.csv        (stage 5)
    pipeline/phase6/data/documents_final.csv           (stage 6B)

Output:
    pipeline/evaluation/cleanliness_report.json
    printed comparison table + cleaning gradient

Reference: evaluation metrics adapted from Zimmermann et al. (2024),
"Approaches to improve preprocessing for Latent Dirichlet Allocation
topic modeling," Decision Support Systems.
"""

import json
import re
import sys
from collections import Counter
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

STAGES = [
    ("stage0_raw",  PIPELINE_DIR / "phase1" / "JFK_Pages_Merged.csv",               "content",              "page"),
    ("stage2",      PIPELINE_DIR / "phase2" / "data" / "pages_phase2_cleaned.csv",   "content_clean_ocr",    "page"),
    ("stage3",      PIPELINE_DIR / "phase3" / "data" / "pages_phase3_linefiltered.csv", "content_clean_lines", "page"),
    ("stage4",      PIPELINE_DIR / "phase4" / "data" / "pages_phase4_modeltext.csv", "content_model_lemma",  "page"),
    ("stage5",      PIPELINE_DIR / "phase5" / "data" / "pages_for_modeling.csv",     "content_model_lemma",  "page"),
    ("stage6b",     PIPELINE_DIR / "phase6" / "data" / "documents_final.csv",        "document_text_lemma",  "doc"),
]

OUTPUT_JSON = SCRIPT_DIR / "cleanliness_report.json"

SYSTEM_DICT = Path("/usr/share/dict/words")

WORD_RE = re.compile(r"[a-z]+")


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------
def load_dictionary() -> tuple[set[str], str]:
    """Return (word set, source label)."""
    if SYSTEM_DICT.exists():
        with open(SYSTEM_DICT, "r", encoding="utf-8", errors="ignore") as f:
            words = {line.strip().lower() for line in f if line.strip()}
        return words, str(SYSTEM_DICT)

    try:
        from nltk.corpus import words as nltk_words
        return {w.lower() for w in nltk_words.words()}, "nltk.corpus.words"
    except Exception as exc:
        print(f"ERROR: no word dictionary available: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-text metrics
# ---------------------------------------------------------------------------
def metrics_for_text(text: str, word_dict: set[str]) -> dict:
    if not isinstance(text, str):
        text = ""
    tokens = text.split()
    n_tokens = len(tokens)
    alpha_tokens = WORD_RE.findall(text.lower())
    n_alpha = len(alpha_tokens)

    if n_alpha > 0:
        dict_hit_rate = sum(1 for t in alpha_tokens if t in word_dict) / n_alpha
    else:
        dict_hit_rate = 0.0

    if n_tokens > 0:
        type_counter = Counter(tokens)
        n_types = len(type_counter)
        type_token_ratio = n_types / n_tokens
        hapax_ratio = sum(1 for c in type_counter.values() if c == 1) / n_types
        mean_token_len = sum(len(t) for t in tokens) / n_tokens
    else:
        type_token_ratio = 0.0
        hapax_ratio = 0.0
        mean_token_len = 0.0

    n_chars = len(text)
    if n_chars > 0:
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        alpha_char_ratio = alpha_chars / n_chars
    else:
        alpha_char_ratio = 0.0

    return {
        "dict_hit_rate": dict_hit_rate,
        "type_token_ratio": type_token_ratio,
        "hapax_ratio": hapax_ratio,
        "alpha_char_ratio": alpha_char_ratio,
        "mean_token_len": mean_token_len,
    }


def aggregate(per_text: list[dict]) -> dict:
    if not per_text:
        return {}
    keys = per_text[0].keys()
    return {k: float(np.mean([r[k] for r in per_text])) for k in keys}


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------
def _read_csv_robust(csv_path: Path, usecols: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except pd.errors.ParserError:
        return pd.read_csv(csv_path, usecols=usecols, engine="python")


def compute_stage(
    csv_path: Path,
    text_col: str,
    unit: str,
    word_dict: set[str],
    desc: str,
) -> dict:
    df = _read_csv_robust(csv_path, [text_col])
    df[text_col] = df[text_col].fillna("")
    per_text = []
    for text in tqdm(df[text_col], desc=desc, unit=unit):
        per_text.append(metrics_for_text(text, word_dict))
    result = aggregate(per_text)
    result["n_units"] = len(per_text)
    result["unit"] = unit
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    separator = "=" * 78
    print(separator)
    print("Per-stage text cleanliness report")
    print(separator)

    print("\nLoading word dictionary ...")
    word_dict, dict_source = load_dictionary()
    print(f"  Source: {dict_source}")
    print(f"  Entries: {len(word_dict):,}")

    print("\nComputing per-stage metrics ...")
    results: dict[str, dict] = {}
    for name, csv_path, col, unit in STAGES:
        print(f"\n  {name}: {csv_path.name}  (col={col}, unit={unit})")
        results[name] = compute_stage(csv_path, col, unit, word_dict, desc=name)

    # --------------------------------------------------------------
    # Save report
    # --------------------------------------------------------------
    report = {
        "dictionary_source": dict_source,
        "dictionary_size": len(word_dict),
        "stages": results,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {OUTPUT_JSON.relative_to(REPO_ROOT)}")

    # --------------------------------------------------------------
    # Summary table
    # --------------------------------------------------------------
    metric_names = [
        "dict_hit_rate",
        "type_token_ratio",
        "hapax_ratio",
        "alpha_char_ratio",
        "mean_token_len",
    ]

    print(f"\n{separator}")
    print("SUMMARY (per stage)")
    print(separator)
    hdr = f"{'stage':<12} {'unit':>4} {'n':>8} " + \
          " ".join(f"{m:>18}" for m in metric_names)
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        row = f"{name:<12} {r['unit']:>4} {r['n_units']:>8,}"
        for m in metric_names:
            row += f" {r[m]:>18.4f}"
        print(row)

    # Cleaning gradient: stage 0 → stage 6B
    print(f"\n{separator}")
    print("CLEANING GRADIENT (stage 0 → stage 6B)")
    print(separator)
    s0 = results["stage0_raw"]
    s6 = results["stage6b"]
    for m in metric_names:
        delta = s6[m] - s0[m]
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  {m:<20}  {s0[m]:>8.4f}  →  {s6[m]:>8.4f}  "
              f"(Δ = {delta:+.4f} {arrow})")
    print(separator)


if __name__ == "__main__":
    main()
