"""
Phase 5 — Corpus Filtering
===========================
Input:  phase4/data/pages_phase4_modeltext.csv
Output: phase5/data/pages_for_modeling.csv
        phase5/data/pages_excluded.csv
        phase5/data/phase5_summary.json

Filters pages into retained vs excluded based on page-level quality
criteria.  No text transformation is applied — only row selection.

Exclusion criteria
------------------
  sparse       — fewer than MIN_TOKENS model tokens
  low_content  — flagged as low-content in Phase 1
  cover        — flagged as likely cover page in Phase 1
  non_english  — non-ASCII character ratio exceeds threshold

Dependencies: pandas, tqdm, json, pathlib
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE5_DIR = SCRIPT_DIR.parent
PHASE4_DIR = PHASE5_DIR.parent / "phase4"

INPUT_CSV = PHASE4_DIR / "data" / "pages_phase4_modeltext.csv"
OUTPUT_DIR = PHASE5_DIR / "data"
RETAINED_PATH = OUTPUT_DIR / "pages_for_modeling.csv"
EXCLUDED_PATH = OUTPUT_DIR / "pages_excluded.csv"
SUMMARY_PATH = OUTPUT_DIR / "phase5_summary.json"

SEPARATOR = "=" * 60
REQUIRED_COLUMNS = {"token_count_model_lemma", "is_low_content_page",
                    "is_likely_cover_page", "content", "file_id"}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_TOKENS = 15
NON_ASCII_THRESH = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def non_ascii_ratio(text: object) -> float:
    """Return the share of non-ASCII characters in a text value."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def print_section(title: str) -> None:
    """Print a consistently formatted section header."""
    print(f"\n{title}")


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print_section("Loading data...")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    validate_columns(df, REQUIRED_COLUMNS, INPUT_CSV.name)
    print(f"  Input:  {INPUT_CSV.name:<35} {len(df):,} rows")
    return df


def build_exclusion_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    print_section("Evaluating exclusion criteria...")

    tqdm.pandas(desc="Non-ASCII ratio")
    mask_non_english = df["content"].progress_apply(non_ascii_ratio) > NON_ASCII_THRESH

    return {
        "sparse": df["token_count_model_lemma"] < MIN_TOKENS,
        "low_content": df["is_low_content_page"] == True,
        "cover": df["is_likely_cover_page"] == True,
        "non_english": mask_non_english,
    }


def apply_exclusions(
    df: pd.DataFrame, criteria: dict[str, pd.Series]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into retained and excluded DataFrames."""
    # Build the pipe-delimited exclusion reason without row-wise apply.
    reason_parts = pd.DataFrame(criteria, index=df.index).fillna(False).astype(bool)
    reasons = pd.Series("", index=df.index, dtype="object")
    for label in reason_parts.columns:
        mask = reason_parts[label]
        empty_mask = mask & reasons.eq("")
        append_mask = mask & reasons.ne("")
        reasons = reasons.mask(empty_mask, label)
        reasons = reasons.where(~append_mask, reasons + "|" + label)

    df["exclusion_reason"] = reasons

    excluded_mask = df["exclusion_reason"] != ""
    retained = df[~excluded_mask].drop(columns=["exclusion_reason"]).copy()
    excluded = df[excluded_mask].copy()

    return retained, excluded


def save_output(
    retained: pd.DataFrame,
    excluded: pd.DataFrame,
    total: int,
    criteria: dict[str, pd.Series],
) -> tuple[dict[str, int], int]:
    print_section("Saving output...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    retained.to_csv(RETAINED_PATH, index=False)
    print(f"  → {RETAINED_PATH.relative_to(PHASE5_DIR)}  {len(retained):>10,} rows")

    excluded.to_csv(EXCLUDED_PATH, index=False)
    print(f"  → {EXCLUDED_PATH.relative_to(PHASE5_DIR)}  {len(excluded):>10,} rows")

    # Summary JSON
    per_criterion = {label: int(mask.sum()) for label, mask in criteria.items()}
    multi_flag = int((sum(criteria.values()) > 1).sum())

    summary = {
        "total_input_pages": total,
        "total_retained": len(retained),
        "total_excluded": len(excluded),
        "exclusion_count_per_criterion": per_criterion,
        "multi_flag_pages": multi_flag,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {SUMMARY_PATH.relative_to(PHASE5_DIR)}")

    return per_criterion, multi_flag


def print_summary(
    total: int,
    retained: pd.DataFrame,
    excluded: pd.DataFrame,
    per_criterion: dict[str, int],
    multi_flag: int,
) -> None:
    n_retained = len(retained)
    n_excluded = len(excluded)

    def fmt_pct(count: int) -> str:
        if total == 0:
            return "n/a"
        return f"{count / total:.1%}"

    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  Input pages              : {total:>8,}")
    print(f"  Retained                 : {n_retained:>8,}  ({fmt_pct(n_retained)})")
    print(f"  Excluded                 : {n_excluded:>8,}  ({fmt_pct(n_excluded)})")
    print()
    print("  Exclusion counts per criterion:")
    for label, count in per_criterion.items():
        print(f"    {label:<14}: {count:>6,}  ({fmt_pct(count)})")
    print()
    print(f"  Pages flagged by >1 criterion: {multi_flag:,}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEPARATOR)
    print("PHASE 5 — Corpus Filtering")
    print(SEPARATOR)

    df = load_data()
    total = len(df)
    criteria = build_exclusion_masks(df)
    retained, excluded = apply_exclusions(df, criteria)
    per_criterion, multi_flag = save_output(retained, excluded, total, criteria)
    print_summary(total, retained, excluded, per_criterion, multi_flag)


if __name__ == "__main__":
    main()
