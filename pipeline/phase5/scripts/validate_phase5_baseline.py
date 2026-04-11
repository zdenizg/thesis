"""Phase 5 baseline validation checks."""

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE5_DIR = SCRIPT_DIR.parent
PHASE4_DIR = PHASE5_DIR.parent / "phase4"

INPUT_PATH = PHASE4_DIR / "data" / "pages_phase4_modeltext.csv"
RETAINED_PATH = PHASE5_DIR / "data" / "pages_for_modeling.csv"
EXCLUDED_PATH = PHASE5_DIR / "data" / "pages_excluded.csv"
SUMMARY_PATH = PHASE5_DIR / "data" / "phase5_summary.json"

SEPARATOR = "=" * 60
VALID_LABELS = {"sparse", "low_content", "cover", "non_english"}
NOTNULL_COLS = ["id", "filename", "page_number"]


def ensure_exists(path: Path) -> None:
    """Raise a clear error if an expected output file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def safe_pct(part: int, whole: int) -> str:
    """Format a percentage while handling zero denominators."""
    if whole == 0:
        return "n/a"
    return f"{(100 * part / whole):.1f}%"


def main() -> None:
    """Run the Phase 5 baseline validation checks."""
    print(SEPARATOR)
    print("PHASE 5 BASELINE VALIDATION")
    print(SEPARATOR)
    print(f"  Phase 4 input : {INPUT_PATH}")
    print(f"  Retained file : {RETAINED_PATH}")
    print(f"  Excluded file : {EXCLUDED_PATH}")
    print(f"  Summary file  : {SUMMARY_PATH}")

    ensure_exists(INPUT_PATH)
    ensure_exists(RETAINED_PATH)
    ensure_exists(EXCLUDED_PATH)
    ensure_exists(SUMMARY_PATH)

    inp = pd.read_csv(INPUT_PATH, low_memory=False)
    retained = pd.read_csv(RETAINED_PATH, low_memory=False)
    excluded = pd.read_csv(EXCLUDED_PATH, low_memory=False)
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        summary = json.load(f)

    results: list[bool] = []

    # Check 1 - retained + excluded == input
    total_out = len(retained) + len(excluded)
    if total_out == len(inp):
        print(
            f"PASS  retained ({len(retained):,}) + excluded ({len(excluded):,}) = "
            f"input ({len(inp):,})"
        )
        results.append(True)
    else:
        print(f"FAIL  retained + excluded = {total_out:,}, expected {len(inp):,}")
        results.append(False)

    # Check 2 - No nulls in key columns in both files
    all_null_ok = True
    for label, df in [("retained", retained), ("excluded", excluded)]:
        available_cols = [column for column in NOTNULL_COLS if column in df.columns]
        missing_cols = [column for column in NOTNULL_COLS if column not in df.columns]
        if missing_cols:
            print(f"INFO  null check skipped missing columns in {label}: {missing_cols}")
        for column in available_cols:
            n_null = int(df[column].isnull().sum())
            if n_null > 0:
                print(f"FAIL  {n_null:,} nulls in {column} ({label})")
                all_null_ok = False
    if all_null_ok:
        print(f"PASS  no nulls in {NOTNULL_COLS} in either output file")
    results.append(all_null_ok)

    # Check 3 - exclusion_reason has no nulls in excluded file
    if "exclusion_reason" in excluded.columns:
        null_reason = int(excluded["exclusion_reason"].isnull().sum())
        if null_reason == 0:
            print("PASS  no nulls in exclusion_reason")
            results.append(True)
        else:
            print(f"FAIL  {null_reason:,} nulls in exclusion_reason")
            results.append(False)
    else:
        print("FAIL  exclusion_reason column missing from excluded file")
        results.append(False)

    # Check 4 - All labels in exclusion_reason are valid
    if "exclusion_reason" in excluded.columns:
        all_labels_ok = True
        for reason in excluded["exclusion_reason"].dropna().unique():
            invalid = set(str(reason).split("|")) - VALID_LABELS
            if invalid:
                print(f"FAIL  invalid label(s) in exclusion_reason: {sorted(invalid)}")
                all_labels_ok = False
                break
        if all_labels_ok:
            print("PASS  all exclusion_reason values use valid labels")
        results.append(all_labels_ok)
    else:
        results.append(False)

    # Check 5 - Summary JSON matches actual counts
    json_ok = (
        summary.get("total_retained") == len(retained)
        and summary.get("total_excluded") == len(excluded)
        and summary.get("total_input_pages") == len(inp)
    )
    if json_ok:
        print("PASS  phase5_summary.json counts match actual file row counts")
        results.append(True)
    else:
        print("FAIL  summary mismatch")
        print(
            "       json:   "
            f"retained={summary.get('total_retained')}, "
            f"excluded={summary.get('total_excluded')}, "
            f"total={summary.get('total_input_pages')}"
        )
        print(
            "       actual: "
            f"retained={len(retained)}, "
            f"excluded={len(excluded)}, "
            f"total={len(inp)}"
        )
        results.append(False)

    # Check 6 - No id in both retained and excluded
    if "id" in retained.columns and "id" in excluded.columns:
        overlap = set(retained["id"]).intersection(set(excluded["id"]))
        if not overlap:
            print("PASS  no id appears in both retained and excluded")
            results.append(True)
        else:
            print(f"FAIL  {len(overlap):,} id(s) appear in both files: {list(overlap)[:5]}")
            results.append(False)
    else:
        print("SKIP  id column not available for overlap check")
        results.append(True)

    # Info - retention rate and exclusion breakdown
    total = len(inp)
    print()
    print(f"  Retention rate : {safe_pct(len(retained), total)}  ({len(retained):,} of {total:,})")
    print("  Exclusion breakdown by criterion:")
    if "exclusion_reason" in excluded.columns:
        for label in sorted(VALID_LABELS):
            count = int(excluded["exclusion_reason"].fillna("").str.contains(label, regex=False).sum())
            print(f"    {label:<14}: {count:,}  ({safe_pct(count, total)})")

    # Summary
    passed = sum(results)
    total_checks = len(results)
    summary_mark = "✓" if passed == total_checks else "✗"
    print()
    print(SEPARATOR)
    print(f"SUMMARY: {passed}/{total_checks} checks passed {summary_mark}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
