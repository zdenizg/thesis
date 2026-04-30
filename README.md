# JFK Assassination Archive — NLP Preprocessing Pipeline

NLP preprocessing pipeline for the MA thesis "Topic Modeling and
Thematic Analysis on JFK Assassination Files" by Zeynep Deniz Güvenol
and Furkan Demir. All code in this repository was written by Zeynep
Deniz Güvenol. The pipeline is a six-phase chain that transforms the
OCR text of the John F. Kennedy assassination records collection into
a model-ready corpus for topic modeling. Each phase reads the previous
phase's output from disk, adds new columns or filters rows, and writes
its own output to a `data/` subdirectory.

## Repository Structure

```
thesis/
├── README.md                           (this file)
├── requirements.txt                    pinned Python dependencies
├── Thesis_Proposal.pdf                 thesis proposal
├── OCR/                                one-shot OCR extraction scripts (out of scope for the pipeline)
├── manual/                             manually curated reference samples
└── pipeline/
    ├── run_all.py                      runs phases 1 → 6B in order
    ├── setup_nltk.py                   one-time NLTK data download
    ├── phase1/                         structural analysis
    │   ├── scripts/
    │   │   ├── phase1_complete_analysis.py
    │   │   ├── inspect_complete_phase1.py
    │   │   └── validate_phase1.py
    │   ├── data/
    │   │   └── pages_phase1_structural.csv
    │   ├── JFK_Pages_Merged.csv        raw input (not committed)
    │   ├── jfk_categorization_55missing.csv
    │   └── merge_missing.py            one-shot migration helper (see Notes)
    ├── phase2/                         boilerplate removal, dehyphenation, OCR normalisation
    │   ├── scripts/
    │   │   ├── phase2_cleaning.py
    │   │   ├── phase2_boilerplate_discovery.py
    │   │   ├── phase2_inspection.py
    │   │   └── validate_phase2.py
    │   └── data/pages_phase2_cleaned.csv
    ├── phase3/                         line-level metadata filtering
    │   ├── scripts/
    │   │   ├── phase3_line_filtering.py
    │   │   ├── phase3_line_discovery.py
    │   │   ├── phase3_validation.py
    │   │   └── validate_phase3.py
    │   └── data/pages_phase3_linefiltered.csv
    ├── phase4/                         tokenisation, stopword removal, lemmatisation
    │   ├── scripts/
    │   │   ├── phase4_modeltext.py
    │   │   ├── phase4_token_discovery.py
    │   │   ├── baselinevalidate_phase4.py
    │   │   └── validate_phase4.py
    │   └── data/pages_phase4_modeltext.csv
    ├── phase5/                         page-level corpus filtering
    │   ├── scripts/
    │   │   ├── filter_corpus.py
    │   │   ├── validate_phase5.py
    │   │   └── validate_phase5_baseline.py
    │   └── data/
    │       ├── pages_for_modeling.csv
    │       ├── pages_excluded.csv
    │       └── phase5_summary.json
    └── phase6/                         document aggregation and modeling preparation
        ├── scripts/
        │   ├── phase6_aggregation.py        (Phase 6A)
        │   └── phase6b_modeling_prep.py     (Phase 6B)
        └── data/
            ├── documents_for_modeling.csv   (6A output)
            └── documents_final.csv          (6B output, model input)
```

CSV files under `data/` are gitignored and reproduced by running the
pipeline.

## Pipeline Overview

Phases 1 through 4 enrich the page-level dataset without dropping rows,
so the page count stays at 83,568 throughout. Phase 5 is the first
filter that removes pages. Phase 6 changes the unit of analysis from
page to document.

### Phase 1 — Structural Analysis

- **Input**: `phase1/JFK_Pages_Merged.csv` (83,621 rows)
- **Output**: `phase1/data/pages_phase1_structural.csv` (**83,568** rows)

Loads the merged page-level dataset, drops rows with empty or null
`content`, and validates that every page from the 55-file supplementary
CSV is present in the merged input. Computes per-page structural metrics
(character/word/line counts, uppercase/numeric/short-token/unique-word
ratios, code-like line ratio) and sets three heuristic flags
(`is_low_content_page`, `is_likely_cover_page`,
`is_likely_distribution_page`).

### Phase 2 — Boilerplate Removal, Dehyphenation, OCR Normalisation

- **Input**: `phase1/data/pages_phase1_structural.csv` (83,568 rows)
- **Output**: `phase2/data/pages_phase2_cleaned.csv` (**83,568** rows)

Removes whole lines that match a fixed list of archive-noise patterns
(FOIA notices, archive ID stamps such as `14-00000`, classification
headers, distribution headings). Joins line-end hyphens that look like
OCR-broken words (e.g. `MEX-\nICO` → `MEXICO`) while leaving deliberate
compounds and archival codes alone. Applies a conservative OCR
normalisation pass: lowercase, whitespace collapse, and removal of
symbol-only and purely-numeric tokens.

### Phase 3 — Line-Level Metadata Filtering

- **Input**: `phase2/data/pages_phase2_cleaned.csv` (83,568 rows)
- **Output**: `phase3/data/pages_phase3_linefiltered.csv` (**83,568** rows)

Removes individual lines that are routing metadata, office codes, or
distribution headings while preserving narrative content. Each line is
classified by five rules in order of confidence: pure routing/office
code (e.g. `wh/3/b`), known metadata phrase, single standalone admin
keyword, known CIA filing/action prefix, or very low natural-language
content (short lines dominated by codes/digits/punctuation). When a
rule is uncertain the line is kept.

### Phase 4 — Tokenisation, Stopword Removal, Lemmatisation

- **Input**: `phase3/data/pages_phase3_linefiltered.csv` (83,568 rows)
- **Output**: `phase4/data/pages_phase4_modeltext.csv` (**83,568** rows)

Tokenises each page with NLTK's Penn Treebank tokenizer, lowercases,
and keeps only tokens of length ≥ 2 with no digits and no pure
punctuation. Removes stopwords using NLTK's English list (198 terms)
plus an archive-specific list (39 terms covering routing words,
reporting verbs, modal verbs, document templates, and the Penn
Treebank possessive marker `'s`). Produces two model-text columns:
`content_model_no_lemma` (filtered tokens only) and
`content_model_lemma` (filtered + WordNet-lemmatised, with a second
stopword pass). Eight Cold War anchor terms — `cuba`, `soviet`,
`mexico`, `embassy`, `oswald`, `cia`, `fbi`, `surveillance` — are
explicitly verified absent from every stopword list.

### Phase 5 — Page-Level Corpus Filtering

- **Input**: `phase4/data/pages_phase4_modeltext.csv` (83,568 rows)
- **Output**: `phase5/data/pages_for_modeling.csv` (**78,022** retained rows)
- **Side outputs**: `pages_excluded.csv` (5,546 rows), `phase5_summary.json`

Splits pages into a retained set and an excluded set without modifying
any text. Four exclusion criteria are evaluated and recorded in a
pipe-delimited `exclusion_reason` column on the excluded set:
`sparse` (fewer than 15 model tokens), `low_content` (Phase 1 heuristic
flag), `cover` (Phase 1 heuristic flag), and `non_english` (more than 5%
non-ASCII characters in the original `content`).

### Phase 6A — Document Aggregation

- **Input**: `phase5/data/pages_for_modeling.csv` (78,022 rows)
- **Output**: `phase6/data/documents_for_modeling.csv` (**2,560** documents)

Concatenates page-level model text into one row per document
(`file_id`), preserving page order. Computes document-level token
counts for both model-text columns and a `retention_ratio` of pages
kept versus the original page count. Picks the most common
`document_type` and `ocr_difficulty` per document and flags whether
any constituent page included handwriting.

### Phase 6B — Modeling Preparation

- **Input**: `phase6/data/documents_for_modeling.csv` (2,560 documents)
- **Output**: `phase6/data/documents_final.csv` (**4,049** documents)

Three-step post-processing on the aggregated documents. First, strips
26 residual archive tokens (e.g. `umbra`, `noforn`, `docid`, `nw`) and
15 metadata phrases (e.g. `jfk assassination system identification`,
`record number`, `agency file number`). Second, chunks any document
longer than 5,000 tokens into numbered sub-documents
(`{file_id}_chunk_001`, `_chunk_002`, …); the row count rises in this
step because long documents become multiple rows. Third, drops any
document with fewer than 50 raw tokens, then drops any document with
fewer than 50 lemma tokens. The final 4,049 documents are the input to
the topic model.

## How to Reproduce

```bash
# 1. Clone the repository
git clone <repo-url> thesis
cd thesis

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pre-fetch NLTK corpora used by Phase 4
python pipeline/setup_nltk.py

# 5. Place the raw input file
#    The pipeline expects pipeline/phase1/JFK_Pages_Merged.csv to exist.
#    This file is gitignored due to size and is not part of the repo.

# 6. Run the full pipeline (phases 1 → 6B)
python pipeline/run_all.py
```

The runner accepts two flags for partial runs:

```bash
python pipeline/run_all.py --phase 3      # run only phase 3
python pipeline/run_all.py --phase 6b     # run only phase 6B
python pipeline/run_all.py --from 4       # run phases 4 → end
```

Each phase is invoked as a subprocess; if any phase exits with a
non-zero status the runner stops and prints which phase failed.

## Requirements

Developed and tested on **Python 3.11.9** (macOS). Pinned dependency
versions are in [`requirements.txt`](requirements.txt). The
preprocessing pipeline itself depends on `pandas`, `numpy`, `tqdm`, and
`nltk`. The OCR scripts under `OCR/` have their own dependencies
(grouped under a separate section in `requirements.txt`) but are not
needed to run the pipeline — their output has already been merged into
`pipeline/phase1/JFK_Pages_Merged.csv`.

## Notes

`pipeline/phase1/merge_missing.py` is a one-shot migration script that
produced `JFK_Pages_Merged.csv` by merging the 55-file supplementary
CSV into the main merged dataset. It contains hardcoded paths from a
previous repository layout (`/Users/.../thesis/cleaning/phase 1 /...`)
and is **not** invoked by `run_all.py`. It is retained in the
repository for provenance only; its output has already been generated
and is now the canonical input to Phase 1.

## Author

Code author: Zeynep Deniz Güvenol.
Thesis co-author: Furkan Demir (not a contributor to this repository).

