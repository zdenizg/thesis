# Fine Coherence Sweep — Summary

Range swept: k ∈ [18, 40] at unit resolution.

## Top 3 k values by c_v coherence

| Rank | k | c_v | log_perplexity |
|---|---|---|---|
| 1 | 29 | 0.6188 | -8.1176 |
| 2 | 25 | 0.6128 | -8.1467 |
| 3 | 28 | 0.6099 | -8.1287 |

## Final-k recommendation

Primary signal: c_v coherence.  Top c_v = 0.6188 at k = 29.  Tie-breaking rule: candidates within 0.02 c_v of the maximum are adjudicated by the interpretability rubric applied in script 04.

Candidates within the 0.02 c_v band: k ∈ {29, 25, 28, 18, 27, 19, 31, 30, 26, 20, 21, 32}.

Inspect in script 04: top-20 words per topic, three representative documents per topic, and distinctiveness between candidates before confirming the final k.
