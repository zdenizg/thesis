# Chunk-Size Sensitivity — Summary

Three independent LDA runs at k = 25 with the frozen hyperparameters (lda/specs/lda_params.md), differing only in the chunk-size threshold applied during an isolated Phase 6B rerun on `pipeline/phase6/data/documents_for_modeling.csv`.

## Comparison

| chunk_size | num_docs | vocab_size | c_v | docs chunked | mean tokens/doc | median tokens/doc |
|-----------:|---------:|-----------:|----:|-------------:|----------------:|------------------:|
| 3,000 | 5,203 | 54,927 | 0.5908 | 443 | 1884.4 | 3000.0 |
| 5,000 | 4,023 | 54,027 | 0.5978 | 369 | 2437.1 | 1470.0 |
| 10,000 | 3,156 | 52,844 | 0.5426 | 272 | 3106.7 | 591.5 |

Highest coherence: chunk_size = 5,000 (c_v = 0.5978).

## Fragmentation

The Phase 6A input contains 2,560 unchunked documents. Each setting splits any document whose raw-token count exceeds the chunk-size threshold into numbered sub-documents.

- chunk_size = 3,000: 443 input documents fragmented; 5,203 sub-documents in the final corpus; mean 1884.4 / median 3000.0 tokens per (sub-)document.
- chunk_size = 5,000: 369 input documents fragmented; 4,023 sub-documents in the final corpus; mean 2437.1 / median 1470.0 tokens per (sub-)document.
- chunk_size = 10,000: 272 input documents fragmented; 3,156 sub-documents in the final corpus; mean 3106.7 / median 591.5 tokens per (sub-)document.

## Recommendation

**Recommended chunk size: 5,000.**

Coherence spread is 0.0553 c_v, exceeding the 0.02 tolerance band. The setting with the highest c_v (chunk_size = 5000, c_v = 0.5978) is preferred.

## Interpretation

Coherence varies by 0.0553 c_v across the three settings. Smaller chunk sizes split more documents into more sub-documents, which can sharpen topic boundaries but also dilutes long-document context, while larger chunk sizes preserve context at the cost of uneven document-length distribution. Per the decision principle in `lda_plan.md`, when coherence differences across the three settings fall within ~0.02 c_v the choice is governed by fragmentation and interpretability rather than by coherence alone; this empirical result is what the methodology's Section [X.X] placeholder is filled with.
