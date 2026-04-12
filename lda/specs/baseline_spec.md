# Baseline vs. Full-Pipeline Comparison Specification

## Purpose

This document defines the minimal-preprocessing baseline against which the
full cleaning pipeline is evaluated.  Both conditions use identical LDA
hyperparameters (see `lda_params.md`) and identical dictionary-filtering
thresholds so that any difference in topic quality is attributable to the
preprocessing steps alone.

## Baseline preprocessing (minimal)

The baseline applies only standard NLP normalisation:

1. **Lowercasing** — all text converted to lowercase.
2. **Tokenisation** — NLTK Penn Treebank tokeniser (`nltk.tokenize.word_tokenize`).
3. **Stop-word removal** — NLTK English stop-word list (`nltk.corpus.stopwords`).
4. **Lemmatisation** — WordNet lemmatiser (`nltk.stem.WordNetLemmatizer`),
   using default POS tag (noun).
5. **Document aggregation** — tokens grouped by `file_id` (one document per
   declassified file).
6. **Chunking** — documents exceeding 5 000 tokens are split into consecutive
   chunks of at most 5 000 tokens.
7. **Short-document filter** — documents (or chunks) with fewer than 50 tokens
   after the above steps are dropped.

## Preprocessing excluded from the baseline

The following pipeline stages are deliberately omitted so their effect can be
measured:

- **Phase 2** — archive-header regex removal.
- **Phase 3** — boilerplate line filtering (rule-based and classifier-based).
- **Phase 5** — page-level quality filtering.
- **Phase 6B** — term blacklist removal.

## Shared settings

| Parameter | Value |
|---|---|
| Dictionary `no_below` | 5 |
| Dictionary `no_above` | 0.5 |
| LDA hyperparameters | as specified in `lda_params.md` |
| Candidate *k* values | identical grid for both conditions |
| Evaluation metrics | identical metrics applied to both conditions |

## Evaluation protocol

Both the baseline and the full-pipeline corpus are evaluated at the same
candidate *k* values using the same coherence and held-out metrics.  Topic
interpretability is scored with the rubric defined in
`interpretability_rubric.md` at the selected *k* for each condition.
