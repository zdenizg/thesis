# JFK Records Collection — NLP & Topic Modelling

This repository contains the code and scripts for a computational analysis of the John F. Kennedy Records Collection held by the National Archives. Using natural language processing and topic modelling techniques, the project explores thematic structure, document classification, and the linguistic patterns present across tens of thousands of declassified government records. The thesis is a joint project by Deniz Güvenoğlu and Furkan Demir.

For project background and research questions, see [Thesis_Proposal.pdf](Thesis_Proposal.pdf).

## Repository Structure

| Folder | Contents |
|--------|----------|
| [`pipeline/`](pipeline/) | Five-phase preprocessing pipeline that takes raw OCR text and produces a clean, topic-modelable corpus (83,621 pages → 77,982 retained). Covers data integration, boilerplate removal, OCR normalisation, stopword/lemmatisation, and quality filtering. |
| [`OCR/`](OCR/) | Scripts for re-extracting the 55 document pages missing from the main dataset due to API quota limits during initial extraction, plus manual categorisation and verification. |
| [`manual/`](manual/) | Manually curated ground-truth index and sample PDFs used for OCR accuracy evaluation. |

## Data Availability

Large data files (CSV exports, PDFs, Excel sheets, Parquet files) are excluded from version control via `.gitignore`. The full dataset is available on request.
