"""Phase 2 baseline validation checks."""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE2_DIR = SCRIPT_DIR.parent
PHASE1_DIR = PHASE2_DIR.parent / "phase1"

INPUT_PATH = PHASE1_DIR / "data" / "pages_phase1_structural.csv"
OUTPUT_PATH = PHASE2_DIR / "data" / "pages_phase2_cleaned.csv"

SEPARATOR = "=" * 60
REQUIRED_COLS = [
    "id",
    "filename",
    "page_number",
    "content",
    "content_clean_ocr",
    "word_count_clean",
    "char_count_clean",
]
NOTNULL_COLS = ["id", "filename", "page_number"]


def ensure_exists(path: Path) -> None:
    """Raise a clear error if an expected CSV file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def main() -> None:
    """Run the Phase 2 baseline validation checks."""
    print(SEPARATOR)
    print("PHASE 2 BASELINE VALIDATION")
    print(SEPARATOR)
    print(f"  Phase 1 input : {INPUT_PATH}")
    print(f"  Phase 2 output: {OUTPUT_PATH}")

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

    # Check 4 — Non-negative word_count_clean and char_count_clean
    neg_word = (
        int((out["word_count_clean"] < 0).sum())
        if "word_count_clean" in out.columns
        else -1
    )
    neg_char = (
        int((out["char_count_clean"] < 0).sum())
        if "char_count_clean" in out.columns
        else -1
    )
    if neg_word == 0 and neg_char == 0:
        print("PASS  word_count_clean and char_count_clean are non-negative everywhere")
        results.append(True)
    else:
        print(
            "FAIL  negative counts: "
            f"word_count_clean={neg_word}, char_count_clean={neg_char}"
        )
        results.append(False)

    # Check 5 — char_count_clean <= char_count from Phase 1
    if {"char_count", "id"}.issubset(inp.columns) and {"char_count_clean", "id"}.issubset(
        out.columns
    ):
        merged = out[["id", "char_count_clean"]].merge(
            inp[["id", "char_count"]], on="id", how="inner"
        )
        unmatched = len(out) - len(merged)
        if unmatched > 0:
            print(f"INFO  char_count comparison: {unmatched:,} output rows were unmatched on id")
        violations = int((merged["char_count_clean"] > merged["char_count"]).sum())
        if violations == 0:
            print(
                "PASS  char_count_clean <= char_count (Phase 1) "
                f"for all {len(merged):,} matched rows"
            )
            results.append(True)
        else:
            print(
                "FAIL  "
                f"{violations:,} rows where char_count_clean > char_count (Phase 1)"
            )
            results.append(False)
    else:
        print("SKIP  char_count comparison — required columns not available")
        results.append(True)

    # Info — word_count_clean distribution
    if "word_count_clean" in out.columns:
        print()
        print("  word_count_clean distribution:")
        print(f"    mean   : {out['word_count_clean'].mean():.1f}")
        print(f"    median : {out['word_count_clean'].median():.1f}")

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
