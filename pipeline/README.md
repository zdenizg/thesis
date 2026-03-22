# JFK Records Preprocessing Pipeline

This folder contains a five-phase preprocessing pipeline that takes raw OCR text from the JFK Records Collection and produces a clean, filtered, topic-modelable corpus. Starting from 83,621 digitized document pages, the pipeline merges a supplementary batch of 55 recovered records, removes structural noise and archival boilerplate, filters metadata lines, normalises tokens for topic modelling, and applies a quality gate to exclude pages that carry no substantive content.

## Pipeline Structure

| Phase | Folder | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| 1 | `phase 1/` | `JFK Pages Rows.csv`, `jfk_categorization_55missing.csv` | `data/pages_phase1_structural.csv` | Merges the main OCR dataset with 55 manually recovered pages, removes empty-content rows, and adds structural metadata flags (character counts, uppercase ratio, cover-page and low-content indicators). |
| 2 | `phase 2/` | `pages_phase1_structural.csv` | `data/pages_phase2_cleaned.csv` | Strips archive boilerplate lines (FOIA stamps, classification marks, distribution headers) and applies OCR normalisation (lowercasing, symbol removal, whitespace collapse). |
| 3 | `phase 3/` | `pages_phase2_cleaned.csv` | `data/pages_phase3_linefiltered.csv` | Removes residual metadata lines — routing codes, office identifiers, filing actions, and isolated low-content lines — using five conservative rule-based filters. |
| 4 | `phase 4/` | `pages_phase3_linefiltered.csv` | `data/pages_phase4_modeltext.csv` | Removes NLTK English stopwords plus 55 archive-specific stopwords, then lemmatises tokens with WordNetLemmatizer to produce the model-ready text columns. |
| 5 | `phase5/` | `pages_phase4_modeltext.csv` | `data/pages_for_modeling.csv`, `data/pages_excluded.csv`, `data/phase5_summary.json` | Applies a four-criterion quality gate (sparse, low-content flag, cover-page flag, non-English) to split the corpus into retained and excluded sets. |

## Run Order

Run the scripts in the following sequence to reproduce the pipeline from scratch. Validation scripts are listed as optional checks after each phase; they print to console and do not modify data.

```
# Phase 1 — Data integration
python "phase 1/merge_missing.py"

# Phase 2 — Structural analysis
python "phase 1/scripts/phase1_complete_analysis.py"
# Validation (optional):
python "phase 1/scripts/validate_phase1.py"
python "phase 1/scripts/inspect_complete_phase1.py"

# Phase 3 — Boilerplate removal and OCR normalisation
python "phase 2/scripts/phase2_cleaning.py"
# Validation (optional):
python "phase 2/scripts/validate_phase2.py"
python "phase 2/scripts/phase2_inspection.py"
python "phase 2/scripts/phase2_boilerplate_discovery.py"

# Phase 4 — Line-level metadata filtering
python "phase 3/scripts/phase3_line_filtering.py"
# Validation (optional):
python "phase 3/scripts/validate_phase3.py"
python "phase 3/scripts/phase3_validation.py"
python "phase 3/scripts/phase3_line_discovery.py"

# Phase 4 — Stopword removal and token normalisation
python "phase 4/scripts/phase4_modeltext.py"
# Validation (optional):
python "phase 4/scripts/baselinevalidate_phase4.py"
python "phase 4/scripts/validate_phase4.py"
python "phase 4/scripts/phase4_token_discovery.py"

# Phase 5 — Corpus filtering
python "phase5/filter_corpus.py"
# Validation (optional):
python "phase5/validate_phase5.py"
python "phase5/scripts/validate_phase5_baseline.py"
```

## Output Files

| File | Description |
|------|-------------|
| `phase5/data/pages_for_modeling.csv` | The retained corpus (77,982 pages) ready for topic modelling, containing all structural and model-text columns. |
| `phase5/data/pages_excluded.csv` | The excluded pages (5,586) with a pipe-delimited `exclusion_reason` field recording which quality criteria each page failed. |
| `phase5/data/phase5_summary.json` | Aggregate counts for total input, retained, excluded, per-criterion exclusion totals, and multi-flag pages. |

## Key Corpus Statistics

- Input pages (raw, before any processing): 83,621
- After empty-row removal (Phase 2 output): 83,568
- Retained for modelling: 77,982 (93.3%)
- Excluded: 5,586 (6.7%)

Exclusion breakdown: sparse pages 3,742; low-content flag 4,134; cover pages 2,617; non-English 741. Of the excluded pages, 3,832 triggered more than one criterion simultaneously.

## Known Limitations

- OCR-corrupted variants of classification stamps (e.g. partially recognised strings such as `sficret`, `c0nfidential`) are not caught by the Phase 3 boilerplate patterns, which match on clean ASCII forms only; a small number of stamp-heavy pages therefore pass through with residual boilerplate text.
- Pages where both ocr_difficulty and document_quality indicated poor scan quality were evaluated as a candidate exclusion criterion but rejected after spot-checking showed the combination also catching genuine intelligence content. A small share of low-quality pages consequently remain in the retained corpus.
