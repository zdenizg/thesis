# OCR Re-extraction (55 Missing Pages)

This folder contains scripts for re-extracting OCR text for the 55 document pages that were missing from the main National Archives dataset due to Google Cloud Vision API quota limits during the initial extraction run.

## Scripts

- `ocr_missing_google.py` — Re-extracts OCR text for the missing pages using the Google Cloud Vision API.
- `categorize_55_missing.py` — Manual categorisation of the 55 recovered pages (document type, content quality, etc.).
- `check_missing_ids.py` — Verifies that all 55 expected record IDs have been recovered and match the expected format.

## Notes

Output files (extracted text, categorisation results, intermediate data) are excluded from version control. The recovered pages are merged into the main corpus in Phase 1 of the preprocessing pipeline (`pipeline/phase 1/`).
