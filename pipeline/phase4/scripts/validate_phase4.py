"""
Phase 4 Validation Script
Validates stopword removal and token normalisation output.
Read-only - no modifications to any dataset.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SEPARATOR = "=" * 60
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE4_DIR = SCRIPT_DIR.parent
INPUT_PATH = PHASE4_DIR / "data" / "pages_phase4_modeltext.csv"
REQUIRED_COLUMNS = {
    "file_id",
    "page_number",
    "word_count_clean",
    "content_clean_lines",
    "content_model_no_lemma",
    "token_count_model_no_lemma",
    "token_count_model_lemma",
}
ENTITIES = ["oswald", "soviet", "mexico", "cia", "embassy"]


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise a clear error if an expected dataset column is missing."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def safe_percent(part: int, whole: int, decimals: int = 1) -> str:
    """Format a percentage while handling zero denominators."""
    if whole == 0:
        return "n/a"
    return f"{(100 * part / whole):.{decimals}f}%"


def preview_text(text: object, limit: int = 500) -> str:
    """Return a single-line preview suitable for terminal output."""
    if not isinstance(text, str):
        return "<NaN>"
    return text[:limit].replace("\n", " <-> ")


def build_token_counter(series: pd.Series, *, desc: str) -> Counter[str]:
    """Count lowercased whitespace-delimited tokens across a Series."""
    counter: Counter[str] = Counter()
    for text in tqdm(series.dropna(), desc=desc, unit="row"):
        counter.update(token.lower() for token in str(text).split())
    return counter


def main() -> None:
    """Run the Phase 4 validation report."""
    print(SEPARATOR)
    print("SECTION 1 - LOAD DATASET")
    print(SEPARATOR)
    print(f"Input path          : {INPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    validate_columns(df, REQUIRED_COLUMNS, INPUT_PATH.name)

    # -----------------------------------------------------------------------
    # 2. Basic dataset information
    # -----------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("SECTION 2 - BASIC DATASET INFORMATION")
    print(SEPARATOR)
    print(f"Rows               : {len(df):,}")
    print(f"Unique file_ids    : {df['file_id'].nunique():,}")
    print(f"Columns            : {list(df.columns)}")
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    missing_rows = missing[missing > 0]
    if missing_rows.empty:
        print("None")
    else:
        print(missing_rows.to_string())

    # -----------------------------------------------------------------------
    # 3. Descriptive statistics
    # -----------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("SECTION 3 - DESCRIPTIVE STATISTICS")
    print(SEPARATOR)
    for col in ["word_count_clean", "token_count_model_no_lemma", "token_count_model_lemma"]:
        print(f"\n--- {col} ---")
        print(df[col].describe().to_string())

    # -----------------------------------------------------------------------
    # 4. Token retention ratio
    # -----------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("SECTION 4 - TOKEN RETENTION RATIO")
    print(SEPARATOR)

    df["token_ratio"] = df["token_count_model_no_lemma"] / df["word_count_clean"].replace(0, np.nan)

    print("\nDescriptive statistics for token_ratio:")
    print(df["token_ratio"].describe().to_string())

    low_ratio = df[df["token_ratio"] < 0.3].copy()
    print(f"\nPages with token_ratio < 0.3: {len(low_ratio):,}")

    if not low_ratio.empty:
        sample = low_ratio.sample(min(10, len(low_ratio)), random_state=42)
        for _, row in sample.iterrows():
            clean_preview = preview_text(row.get("content_clean_lines", ""), 300)
            model_preview = preview_text(row.get("content_model_no_lemma", ""), 300)
            print(f"\n  file_id        : {row['file_id']}")
            print(f"  page_number    : {row['page_number']}")
            print(f"  word_count_clean          : {row['word_count_clean']}")
            print(f"  token_count_model_no_lemma: {row['token_count_model_no_lemma']}")
            print(f"  content_clean_lines preview:\n    {clean_preview}")
            print(f"  content_model_no_lemma preview:\n    {model_preview}")

    # -----------------------------------------------------------------------
    # 5. Vocabulary size comparison
    # -----------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("SECTION 5 - VOCABULARY SIZE COMPARISON")
    print(SEPARATOR)

    print("Building token counters...")
    clean_counter = build_token_counter(df["content_clean_lines"], desc="content_clean_lines")
    model_counter = build_token_counter(df["content_model_no_lemma"], desc="content_model_no_lemma")

    vocab_clean = set(clean_counter.keys())
    vocab_model = set(model_counter.keys())
    reduction = np.nan
    if len(vocab_clean) > 0:
        reduction = (1 - len(vocab_model) / len(vocab_clean)) * 100

    print(f"Vocabulary size - content_clean_lines       : {len(vocab_clean):,}")
    print(f"Vocabulary size - content_model_no_lemma    : {len(vocab_model):,}")
    if pd.isna(reduction):
        print("Percentage reduction                        : n/a")
    else:
        print(f"Percentage reduction                        : {reduction:.1f}%")

    # -----------------------------------------------------------------------
    # 6. Top 30 tokens
    # -----------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("SECTION 6 - TOP 30 TOKENS")
    print(SEPARATOR)

    print("\nTop 30 - content_clean_lines:")
    for rank, (token, count) in enumerate(clean_counter.most_common(30), 1):
        print(f"  {rank:>2}. {token:<25} {count:,}")

    print("\nTop 30 - content_model_no_lemma:")
    for rank, (token, count) in enumerate(model_counter.most_common(30), 1):
        print(f"  {rank:>2}. {token:<25} {count:,}")

    # -----------------------------------------------------------------------
    # 7. Important entity counts
    # -----------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("SECTION 7 - IMPORTANT ENTITY VERIFICATION")
    print(SEPARATOR)

    print(f"\n{'Token':<15} {'In content_clean_lines':>22} {'In content_model_no_lemma':>26}")
    print("-" * 65)
    entity_counts_clean = {entity: clean_counter.get(entity, 0) for entity in ENTITIES}
    entity_counts_model = {entity: model_counter.get(entity, 0) for entity in ENTITIES}
    for entity in ENTITIES:
        print(
            f"{entity:<15} {entity_counts_clean[entity]:>22,} "
            f"{entity_counts_model[entity]:>26,}"
        )

    # -----------------------------------------------------------------------
    # 8. Pages with zero model tokens
    # -----------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("SECTION 8 - PAGES WITH ZERO MODEL TOKENS")
    print(SEPARATOR)

    zero_token = df[df["token_count_model_no_lemma"] == 0]
    print(f"Pages where token_count_model_no_lemma == 0: {len(zero_token):,}")

    if not zero_token.empty:
        sample_zero = zero_token.sample(min(10, len(zero_token)), random_state=42)
        for _, row in sample_zero.iterrows():
            clean_preview = preview_text(row.get("content_clean_lines", ""), 400)
            model_preview = preview_text(row.get("content_model_no_lemma", ""), 400)
            print(f"\n  file_id        : {row['file_id']}")
            print(f"  page_number    : {row['page_number']}")
            print(f"  content_clean_lines:\n    {clean_preview}")
            print(f"  content_model_no_lemma:\n    {model_preview}")

    # -----------------------------------------------------------------------
    # 9. Random page comparisons
    # -----------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("SECTION 9 - 10 RANDOM PAGE COMPARISONS")
    print(SEPARATOR)

    sample_size = min(10, len(df))
    if sample_size == 0:
        print("No rows available for random page comparisons.")
    else:
        sample_pages = df.sample(sample_size, random_state=42)
        for index, (_, row) in enumerate(sample_pages.iterrows(), 1):
            clean_preview = preview_text(row.get("content_clean_lines", ""), 500)
            model_preview = preview_text(row.get("content_model_no_lemma", ""), 500)
            print(f"\n--- Page {index} | file_id={row['file_id']} | page={row['page_number']} ---")
            print(f"  [CLEAN]  {clean_preview}")
            print(f"  [MODEL]  {model_preview}")

    # -----------------------------------------------------------------------
    # 10. Diagnostic summary
    # -----------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("SECTION 10 - DIAGNOSTIC SUMMARY")
    print(SEPARATOR)

    median_ratio = df["token_ratio"].median()
    low_ratio_pages = int((df["token_ratio"] < 0.3).sum())
    pct_low = (low_ratio_pages / len(df) * 100) if len(df) > 0 else np.nan
    n_zero = len(zero_token)

    entities_present = all(count > 0 for count in entity_counts_model.values())

    def fmt_ratio(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:.3f}"

    def fmt_pct(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:.1f}%"

    print(f"\nToken retention (median ratio)  : {fmt_ratio(median_ratio)}")
    print(
        f"Pages with ratio < 0.3         : {fmt_pct(pct_low)} "
        f"({low_ratio_pages:,} pages)"
    )
    print(f"Pages with zero model tokens   : {n_zero:,}")
    if pd.isna(reduction):
        print("Vocabulary reduction           : n/a")
    else:
        print(f"Vocabulary reduction           : {reduction:.1f}%")
    print(f"All key entities present       : {entities_present}")

    print("\nConclusion:")
    if (
        pd.notna(median_ratio)
        and pd.notna(pct_low)
        and pd.notna(reduction)
        and median_ratio >= 0.3
        and pct_low < 10
        and entities_present
        and reduction < 70
    ):
        print("  Phase 4 output appears VALID and ready for topic modeling.")
        print("  Token reduction is reasonable; important entities are preserved.")
    else:
        issues = []
        if pd.isna(median_ratio) or median_ratio < 0.3:
            issues.append(f"median token ratio is low ({fmt_ratio(median_ratio)})")
        if pd.isna(pct_low) or pct_low >= 10:
            issues.append(f"{fmt_pct(pct_low)} of pages have ratio < 0.3")
        if not entities_present:
            missing = [entity for entity, count in entity_counts_model.items() if count == 0]
            issues.append(f"missing entities: {missing}")
        if pd.isna(reduction) or reduction >= 70:
            issues.append(
                "vocabulary reduction is high "
                f"({('n/a' if pd.isna(reduction) else f'{reduction:.1f}%')})"
            )
        print("  WARNING - potential issues detected:")
        for issue in issues:
            print(f"    - {issue}")


if __name__ == "__main__":
    main()
