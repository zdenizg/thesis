# Baseline vs. Full Pipeline — Summary

Stage 4 (RQ2): does archive-specific cleaning measurably improve topic quality? Both models are trained at k = 25 with identical frozen LDA hyperparameters (lda/specs/lda_params.md). The only thing that varies is the preprocessing — Phases 2, 3, 4, 5, and 6B are applied to the full-pipeline corpus and skipped for the baseline corpus.

Convention: `delta = full_pipeline − baseline`. Positive delta on coherence means the full pipeline scores higher; negative delta on contamination / noise-topic counts means the full pipeline is cleaner.

## Metrics

| metric | full pipeline | baseline | delta |
|:-------|--------------:|---------:|------:|
| c_v coherence (k=25) | 0.6128 | 0.5639 | +0.0489 |
| metadata blacklist hits in top-20 | 0 | 13 | -13 |
| noise topics (≥3/top-10 from noise list) | 0 | 0 | +0 |

## Metadata contamination breakdown

Counts the times each of the 26 Phase 6B blacklist tokens appears anywhere in any topic's top-20 list. The baseline never strips these tokens, so a non-trivial gap here is direct evidence that Phase 6B is doing the work it was designed for.

| token | full pipeline | baseline |
|:------|--------------:|---------:|
| iden | 0 | 1 |
| nw | 0 | 12 |

## Noise topics

Topics with ≥ 3 of the top-10 words drawn from the blacklist + archive-boilerplate noise list (see script 09 source for the exact list).

- Full pipeline noise topics (n = 0): —
- Baseline noise topics      (n = 0): —

## Interpretation

At identical hyperparameters and identical k, the comparison is mixed and warrants a closer look. Coherence shifts by +0.0489 c_v from baseline to full pipeline. Phase 6B blacklist tokens appear in the top-20 lists 0 times in the full-pipeline model versus 13 times in the baseline (delta -13). Topics flagged as administrative noise (≥3 of top-10 from the noise list) drop from 0 in the baseline to 0 in the full pipeline (delta +0). Because the only thing that varies between the two conditions is the archive-specific preprocessing, these differences are attributable to that pipeline. The contamination and noise-topic counts speak more directly than coherence to RQ2 — c_v is a relative quality measure, but a topic whose top words are `noforn`, `docid`, and `decl` is unambiguously less useful to a historian than one whose top words name agencies and operations.
