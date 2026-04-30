# Baseline k Sweep — Fairness Check

k = 25 was chosen on the full-pipeline coherence curve. To check whether that choice unfairly disadvantages the baseline, this script sweeps the baseline corpus over a broad grid of k with all other LDA hyperparameters frozen (lda/specs/lda_params.md). The reported best c_v is the baseline's own best-case score.

k grid: [10, 15, 20, 25, 30, 35, 40]

## Per-k results

| k | c_v | log_perplexity |
|--:|----:|---------------:|
| 10 | 0.6380  ★ | -8.2637 |
| 15 | 0.6053 | -8.2003 |
| 20 | 0.5897 | -8.1761 |
| 25 | 0.5639 | -8.1495 |
| 30 | 0.5748 | -8.1349 |
| 35 | 0.5925 | -8.1250 |
| 40 | 0.5735 | -8.1214 |

## Best k vs. full pipeline

- Baseline best k         : **10**  (c_v = 0.6380)
- Full pipeline at k = 25  : c_v = 0.6128
- Δ (full − baseline_best): -0.0252

## Interpretation

At its own best k (10), the baseline scores c_v = 0.6380, which is +0.0252 above the full pipeline's k = 25 score of 0.6128. This means the headline coherence number from script 09 was sensitive to the choice of k: anchoring the baseline at k = 25 understated its achievable c_v. The contamination and noise-topic counts from script 09 remain the more direct evidence for RQ2 — c_v alone does not distinguish the two preprocessing regimes once k is allowed to vary.
