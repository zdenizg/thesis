# Multi-seed Robustness — Full Pipeline vs. Baseline (k = 25)

Both models are trained at k = 25 with identical frozen LDA hyperparameters (lda/specs/lda_params.md); only `random_state` varies. The question this script answers: is the coherence gap reported in script 09 (single-seed) robust across seeds, or an artefact of the seed used there?

Seeds tested: [42, 123, 456, 789, 2025]

## Per-seed results

| seed | full pipeline c_v | baseline c_v | delta (full − baseline) |
|-----:|------------------:|-------------:|------------------------:|
| 42 | 0.6128 | 0.5639 | +0.0489 |
| 123 | 0.5871 | 0.5818 | +0.0053 |
| 456 | 0.6200 | 0.5851 | +0.0349 |
| 789 | 0.5695 | 0.5466 | +0.0229 |
| 2025 | 0.6015 | 0.6150 | -0.0135 |

## Aggregate (n = 5 seeds)

| quantity | mean | SD |
|:---------|-----:|---:|
| full pipeline c_v | 0.5982 | 0.0203 |
| baseline c_v | 0.5785 | 0.0256 |
| delta (full − baseline) | +0.0197 | 0.0245 |

## Consistency

Full pipeline beat baseline at **4 of 5** seeds. Verdict: **Gap is inconsistent**.

## Interpretation

The full pipeline does NOT consistently beat the baseline across seeds. Of 5 seeds, the full pipeline won at 4. Mean delta is +0.0197 c_v (SD 0.0245); per-seed deltas range from -0.0135 to +0.0489. The single-seed gap reported in script 09 should not be read as evidence that the full pipeline strictly improves coherence — c_v differences at this scale appear to be within seed-driven noise. The contamination and noise-topic metrics from script 09 remain the more reliable signals for RQ2.
