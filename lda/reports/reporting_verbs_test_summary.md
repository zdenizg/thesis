# Reporting Verbs Test — Summary

**Question.** Does removing `said`, `stated`, `advised` from `ARCHIVE_STOPWORDS` improve LDA topic quality?

**Method.** Re-ran Phase 4 → 6B → LDA(k=25) with these three verbs RETAINED. All other settings (rest of stoplist, Phase 5 filters, Phase 6 blacklist + chunking, dictionary `filter_extremes(no_below=5, no_above=0.5)`, and frozen LDA hyperparameters) are identical to the main pipeline.

## Results

| metric | main pipeline | verbs retained | Δ |
|---|---:|---:|---:|
| c_v (k=25) | 0.6128 | 0.6146 | +0.0018 |
| documents | — | 4,015 | — |
| vocabulary | — | 54,068 | — |

## Reporting verbs in top-20 topic words

- `advised` appears in topics: 2, 11, 15, 16
- `said` appears in topics: 0, 2, 9, 14, 21
- `stated` appears in topics: 2, 16

## Verdict

**Keep current stopword list.** |Δc_v| = 0.0018 < 0.01; the change is within noise. Retaining the verbs adds no measurable benefit, and the original justification (zero topical signal) holds.

_Decision threshold: |Δc_v| ≥ 0.01 to recommend a change to the stopword list._

