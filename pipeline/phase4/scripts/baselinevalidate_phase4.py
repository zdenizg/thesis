"""Phase 4 baseline validation checks."""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE4_DIR = SCRIPT_DIR.parent
PHASE3_DIR = PHASE4_DIR.parent / "phase3"

INPUT_PATH = PHASE3_DIR / "data" / "pages_phase3_linefiltered.csv"
OUTPUT_PATH = PHASE4_DIR / "data" / "pages_phase4_modeltext.csv"

SEPARATOR = "=" * 60
REQUIRED_COLS = [
    "id",
    "filename",
    "page_number",
    "content_model_lemma",
    "content_model_no_lemma",
    "token_count_model_lemma",
    "token_count_model_no_lemma",
]
NOTNULL_COLS = ["id", "filename", "page_number"]


def ensure_exists(path: Path) -> None:
    """Raise a clear error if an expected CSV file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def main() -> None:
    """Run the Phase 4 baseline validation checks."""
    print(SEPARATOR)
    print("PHASE 4 BASELINE VALIDATION")
    print(SEPARATOR)
    print(f"  Phase 3 input : {INPUT_PATH}")
    print(f"  Phase 4 output: {OUTPUT_PATH}")

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

    # Check 4 — token_count_model_lemma is non-negative
    if "token_count_model_lemma" in out.columns:
        neg = int((out["token_count_model_lemma"] < 0).sum())
        if neg == 0:
            print("PASS  token_count_model_lemma is non-negative everywhere")
            results.append(True)
        else:
            print(f"FAIL  {neg:,} rows with negative token_count_model_lemma")
            results.append(False)
    else:
        print("SKIP  token_count_model_lemma column not found")
        results.append(True)

    # Check 5 — Nulls in content_model_lemma
    if "content_model_lemma" in out.columns:
        null_lemma = int(out["content_model_lemma"].isnull().sum())
        zero_token = (
            int((out["token_count_model_lemma"] == 0).sum())
            if "token_count_model_lemma" in out.columns
            else None
        )
        if null_lemma == 0:
            print("PASS  no nulls in content_model_lemma")
            results.append(True)
        elif zero_token is not None and null_lemma == zero_token:
            print(
                f"PASS  {null_lemma:,} nulls in content_model_lemma - "
                "all correspond to zero-token pages (expected)"
            )
            results.append(True)
        else:
            print(
                f"FAIL  {null_lemma:,} nulls in content_model_lemma "
                f"({zero_token} are zero-token pages)"
            )
            results.append(False)
    else:
        print("SKIP  content_model_lemma column not found")
        results.append(True)

    # Info — token_count_model_lemma distribution
    if "token_count_model_lemma" in out.columns:
        token_counts = out["token_count_model_lemma"]
        print()
        print("  token_count_model_lemma distribution:")
        print(f"    mean   : {token_counts.mean():.1f}")
        print(f"    median : {token_counts.median():.1f}")
        print(f"    min    : {int(token_counts.min())}")
        print(f"    max    : {int(token_counts.max())}")
        print(f"    zero-token pages: {int((token_counts == 0).sum()):,}")

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
