"""Phase 3 baseline validation checks."""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parent
PHASE2_DIR = PHASE3_DIR.parent / "phase2"

INPUT_PATH = PHASE2_DIR / "data" / "pages_phase2_cleaned.csv"
OUTPUT_PATH = PHASE3_DIR / "data" / "pages_phase3_linefiltered.csv"

SEPARATOR = "=" * 60
REQUIRED_COLS = [
    "id",
    "filename",
    "page_number",
    "content_clean_lines",
    "lines_before",
    "lines_after",
    "lines_removed",
    "line_removal_ratio",
]
NOTNULL_COLS = ["id", "filename", "page_number"]


def ensure_exists(path: Path) -> None:
    """Raise a clear error if an expected CSV file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def main() -> None:
    """Run the Phase 3 baseline validation checks."""
    print(SEPARATOR)
    print("PHASE 3 BASELINE VALIDATION")
    print(SEPARATOR)
    print(f"  Phase 2 input : {INPUT_PATH}")
    print(f"  Phase 3 output: {OUTPUT_PATH}")

    ensure_exists(INPUT_PATH)
    ensure_exists(OUTPUT_PATH)

    inp = pd.read_csv(INPUT_PATH, low_memory=False)
    out = pd.read_csv(OUTPUT_PATH, low_memory=False)

    results: list[bool] = []

    # Check 1 — Row count
    if len(out) == len(inp):
        print(f"PASS  row count: {len(out):,} rows (matches input)")
        results.append(True)
    else:
        print(f"FAIL  row count: input={len(inp):,}, output={len(out):,}")
        results.append(False)

    # Check 2 — Required columns
    missing = [column for column in REQUIRED_COLS if column not in out.columns]
    if not missing:
        print(f"PASS  required columns: all {len(REQUIRED_COLS)} present")
        results.append(True)
    else:
        print(f"FAIL  required columns missing: {missing}")
        results.append(False)

    # Check 3 — No nulls in key columns
    available_notnull_cols = [column for column in NOTNULL_COLS if column in out.columns]
    missing_notnull_cols = [column for column in NOTNULL_COLS if column not in out.columns]
    if missing_notnull_cols:
        print(f"INFO  null check skipped missing columns: {missing_notnull_cols}")
    null_counts = {
        column: int(out[column].isnull().sum()) for column in available_notnull_cols
    }
    any_nulls = any(count > 0 for count in null_counts.values())
    if not any_nulls:
        print(f"PASS  no nulls in {NOTNULL_COLS}")
        results.append(True)
    else:
        print(f"FAIL  nulls found: {null_counts}")
        results.append(False)

    # Check 4 — lines_after <= lines_before
    if {"lines_before", "lines_after"}.issubset(out.columns):
        violations = int((out["lines_after"] > out["lines_before"]).sum())
        if violations == 0:
            print("PASS  lines_after <= lines_before everywhere")
            results.append(True)
        else:
            print(f"FAIL  {violations:,} rows where lines_after > lines_before")
            results.append(False)
    else:
        print("SKIP  lines_after/lines_before columns not found")
        results.append(True)

    # Check 5 — line_removal_ratio between 0 and 1
    if "line_removal_ratio" in out.columns:
        below = int((out["line_removal_ratio"] < 0).sum())
        above = int((out["line_removal_ratio"] > 1).sum())
        if below == 0 and above == 0:
            print("PASS  line_removal_ratio is between 0 and 1 everywhere")
            results.append(True)
        else:
            print(f"FAIL  line_removal_ratio out of range: <0={below}, >1={above}")
            results.append(False)
    else:
        print("SKIP  line_removal_ratio column not found")
        results.append(True)

    # Info — line_removal_ratio distribution
    if "line_removal_ratio" in out.columns:
        print()
        print("  line_removal_ratio distribution:")
        print(f"    mean   : {out['line_removal_ratio'].mean():.3f}")
        print(f"    median : {out['line_removal_ratio'].median():.3f}")

    # Summary
    passed = sum(results)
    total = len(results)
    summary_mark = "✓" if passed == total else "✗"
    print()
    print(SEPARATOR)
    print(f"SUMMARY: {passed}/{total} checks passed {summary_mark}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
