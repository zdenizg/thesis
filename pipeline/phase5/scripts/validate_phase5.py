"""
Phase 5 validation script.
Checks retained pages, exclusion overlap, and column integrity.
Read-only - no modifications to any dataset.
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE5_DIR = SCRIPT_DIR.parent
PHASE4_DIR = PHASE5_DIR.parent / "phase4"

PHASE4_INPUT = PHASE4_DIR / "data" / "pages_phase4_modeltext.csv"
RETAINED_PATH = PHASE5_DIR / "data" / "pages_for_modeling.csv"
EXCLUDED_PATH = PHASE5_DIR / "data" / "pages_excluded.csv"

SEPARATOR = "=" * 60
EXCLUSION_LABELS = ["sparse", "low_content", "cover", "non_english"]
RETAINED_REQUIRED = {"filename", "page_number", "content"}
EXCLUDED_REQUIRED = {"exclusion_reason"}


def ensure_exists(path: Path) -> None:
    """Raise a clear error if an expected file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def preview_text(text: object, limit: int = 300) -> str:
    """Return a compact single-line preview for terminal output."""
    if not isinstance(text, str) or not text:
        return "<NaN>"
    return text[:limit].replace("\n", " <-> ")


def compare_columns(actual: list[str], expected: list[str]) -> tuple[list[str], list[str]]:
    """Return columns missing from actual and extra beyond expected."""
    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]
    return missing, extra


def main() -> None:
    """Run the Phase 5 validation checks."""
    print(SEPARATOR)
    print("PHASE 5 VALIDATION")
    print(SEPARATOR)
    print(f"  Phase 4 input : {PHASE4_INPUT}")
    print(f"  Retained file : {RETAINED_PATH}")
    print(f"  Excluded file : {EXCLUDED_PATH}")

    ensure_exists(PHASE4_INPUT)
    ensure_exists(RETAINED_PATH)
    ensure_exists(EXCLUDED_PATH)

    retained = pd.read_csv(RETAINED_PATH, low_memory=False)
    excluded = pd.read_csv(EXCLUDED_PATH, low_memory=False)
    validate_columns(retained, RETAINED_REQUIRED, RETAINED_PATH.name)
    validate_columns(excluded, EXCLUDED_REQUIRED, EXCLUDED_PATH.name)

    # -----------------------------------------------------------------
    # Check 1 - Spot-check retained pages
    # -----------------------------------------------------------------
    print()
    print(SEPARATOR)
    print("CHECK 1 - SPOT-CHECK RETAINED PAGES")
    print(SEPARATOR)

    sample_size = min(20, len(retained))
    if sample_size == 0:
        print("No retained pages available for spot-checking.")
    else:
        print(f"Sample size: {sample_size} pages (random_state=42)")
        sample = retained.sample(sample_size, random_state=42)
        for index, (_, row) in enumerate(sample.iterrows(), 1):
            print(f"\n--- {index} ---")
            print(f"filename    : {row['filename']}")
            print(f"page_number : {row['page_number']}")
            print("content     :")
            print(preview_text(row.get("content", "")))

    # -----------------------------------------------------------------
    # Check 2 - Overlap between exclusion criteria
    # -----------------------------------------------------------------
    print()
    print(SEPARATOR)
    print("CHECK 2 - EXCLUSION CRITERIA OVERLAP")
    print(SEPARATOR)

    reasons = excluded["exclusion_reason"].fillna("")

    single_flag = (reasons.str.count(r"\|") == 0) & (reasons != "")
    multi_flag = reasons.str.count(r"\|") >= 1

    print(f"\n  Flagged by exactly one criterion : {int(single_flag.sum()):,}")
    print(f"  Flagged by more than one         : {int(multi_flag.sum()):,}")

    print("\n  Per-criterion counts:")
    for label in EXCLUSION_LABELS:
        count = int(reasons.str.contains(label, regex=False).sum())
        print(f"    {label:<14}: {count:,}")

    # -----------------------------------------------------------------
    # Check 3 - Column integrity
    # -----------------------------------------------------------------
    print()
    print(SEPARATOR)
    print("CHECK 3 - COLUMN INTEGRITY")
    print(SEPARATOR)

    phase4_cols = pd.read_csv(PHASE4_INPUT, low_memory=False, nrows=0).columns.tolist()
    retained_cols = retained.columns.tolist()
    excluded_cols = excluded.columns.tolist()

    retained_missing, retained_extra = compare_columns(retained_cols, phase4_cols)
    excluded_expected = phase4_cols + ["exclusion_reason"]
    excluded_missing, excluded_extra = compare_columns(excluded_cols, excluded_expected)

    if not retained_missing and not retained_extra:
        print("\n  PASS - retained CSV has exactly the same columns as Phase 4 input.")
    else:
        if retained_missing:
            print(f"\n  FAIL - retained is missing columns: {retained_missing}")
        if retained_extra:
            print(f"  NOTE - retained has extra columns: {retained_extra}")

    if not excluded_missing and not excluded_extra:
        print("  PASS - excluded CSV matches Phase 4 columns plus exclusion_reason.")
    else:
        if excluded_missing:
            print(f"  FAIL - excluded is missing columns: {excluded_missing}")
        if excluded_extra:
            print(f"  NOTE - excluded has unexpected extra columns: {excluded_extra}")


if __name__ == "__main__":
    main()
