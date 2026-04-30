# Cold War Relevance — k = 25

First-pass heuristic scores from `lda/scripts/05_cold_war_relevance.py`. Scores are fractions of the pre-registered Cold War vocabulary (`lda/specs/cold_war_vocabulary.yml`) matched in each topic's top-20 words. **Not an objective classifier** — use together with the manual interpretability rubric.

- Cold-War-core       : overall ≥ 0.15
- Cold-War-adjacent   : 0.05 ≤ overall < 0.15
- Low-overall         : overall < 0.05 (analyst assigns an 'Administrative' label only after confirming top-word content)

## Ranked topics (highest overall first)

| rank | topic_id | overall | tier | geo | agency | trade | diplo | actor | matched |
|-----:|---------:|--------:|:-----|----:|-------:|------:|------:|------:|:--------|
| 1 | 24 | 0.164 | Cold-War-core | 0.29 | 0.00 | 0.14 | 0.14 | 0.25 | soviet (geopolitical); mexico (geopolitical); surveillance (tradecraft); embassy (diplomatic); oswald (named_actor) |
| 2 | 13 | 0.147 | Cold-War-adjacent | 0.14 | 0.20 | 0.00 | 0.14 | 0.25 | soviet (geopolitical); kgb (agency); embassy (diplomatic); oswald (named_actor) |
| 3 | 21 | 0.136 | Cold-War-adjacent | 0.29 | 0.00 | 0.00 | 0.14 | 0.25 | cuba (geopolitical); soviet (geopolitical); communist (diplomatic); castro (named_actor) |
| 4 | 16 | 0.126 | Cold-War-adjacent | 0.14 | 0.20 | 0.00 | 0.29 | 0.00 | soviet (geopolitical); fbi (agency); communist (diplomatic); party (diplomatic) |
| 5 | 5 | 0.119 | Cold-War-adjacent | 0.14 | 0.20 | 0.00 | 0.00 | 0.25 | cuba (geopolitical); fbi (agency); oswald (named_actor) |
| 6 | 7 | 0.086 | Cold-War-adjacent | 0.29 | 0.00 | 0.00 | 0.14 | 0.00 | soviet (geopolitical); mexico (geopolitical); embassy (diplomatic) |
| 7 | 12 | 0.086 | Cold-War-adjacent | 0.14 | 0.00 | 0.00 | 0.29 | 0.00 | vietnam (geopolitical); communist (diplomatic); party (diplomatic) |
| 8 | 20 | 0.069 | Cold-War-adjacent | 0.00 | 0.20 | 0.00 | 0.14 | 0.00 | fbi (agency); party (diplomatic) |
| 9 | 1 | 0.057 | Cold-War-adjacent | 0.00 | 0.00 | 0.00 | 0.29 | 0.00 | communist (diplomatic); party (diplomatic) |
| 10 | 4 | 0.057 | Cold-War-adjacent | 0.29 | 0.00 | 0.00 | 0.00 | 0.00 | cuba (geopolitical); mexico (geopolitical) |
| 11 | 11 | 0.057 | Cold-War-adjacent | 0.29 | 0.00 | 0.00 | 0.00 | 0.00 | cuba (geopolitical); mexico (geopolitical) |
| 12 | 18 | 0.057 | Cold-War-adjacent | 0.00 | 0.00 | 0.00 | 0.29 | 0.00 | communist (diplomatic); party (diplomatic) |
| 13 | 0 | 0.040 | Low-overall | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | fbi (agency) |
| 14 | 3 | 0.040 | Low-overall | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | fbi (agency) |
| 15 | 10 | 0.040 | Low-overall | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | fbi (agency) |
| 16 | 6 | 0.029 | Low-overall | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | cuba (geopolitical) |
| 17 | 22 | 0.029 | Low-overall | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | mexico (geopolitical) |
| 18 | 2 | 0.000 | Low-overall | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| 19 | 8 | 0.000 | Low-overall | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| 20 | 9 | 0.000 | Low-overall | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| 21 | 14 | 0.000 | Low-overall | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| 22 | 15 | 0.000 | Low-overall | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| 23 | 17 | 0.000 | Low-overall | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| 24 | 19 | 0.000 | Low-overall | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| 25 | 23 | 0.000 | Low-overall | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |

## Tier counts

- Cold-War-core: 1
- Cold-War-adjacent: 11
- Low-overall: 13

## Category-level breakdown

Top-3 topic IDs per category (non-zero scores only, ties broken by topic_id):

- **geopolitical** : 4, 7, 11
- **agency** : 0, 3, 5
- **tradecraft** : 24
- **diplomatic** : 1, 12, 16
- **named_actor** : 5, 13, 21

## Methodological caveat

The tradecraft term `source` is also a generic English word (deferred item #10). Inspect `matched_terms` in the CSV to see how much of a topic's tradecraft score is driven by `source` alone before drawing interpretive conclusions.
