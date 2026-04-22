# Fine Coherence Sweep — Summary

Range swept: k ∈ [18, 40] at unit resolution.

## Top 3 k values by c_v coherence

| Rank | k | c_v | log_perplexity |
|---|---|---|---|
| 1 | 23 | 0.6115 | -8.1288 |
| 2 | 21 | 0.6111 | -8.1518 |
| 3 | 24 | 0.6110 | -8.1293 |

## Final-k recommendation

Primary signal: c_v coherence.  Top c_v = 0.6115 at k = 23.  Tie-breaking rule: candidates within 0.02 c_v of the maximum are adjudicated by the interpretability rubric applied in script 04.

Candidates within the 0.02 c_v band: k ∈ {23, 21, 24, 22, 33, 38, 39, 19, 35}.

Inspect in script 04: top-20 words per topic, three representative documents per topic, and distinctiveness between candidates before confirming the final k.
