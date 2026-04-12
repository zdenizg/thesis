# Topic Interpretability Rubric

## Purpose

This rubric provides a structured, reproducible scoring system for evaluating
the human interpretability of individual LDA topics.  Scores are assigned
during **pass 1** of the labelling protocol (see `topic_labelling_protocol.md`)
— that is, after inspecting the top-20 words for each topic but **before**
reading representative documents — and are **not revised** in pass 2.

## Dimensions

Each topic is scored on three dimensions using a 1–3 integer scale.

### 1. Semantic Coherence

Do the top words form a recognisable, unified concept?

| Score | Criterion |
|---|---|
| 3 | Top words clearly and immediately suggest a single coherent theme. |
| 2 | A theme is discernible but some words are off-topic or ambiguous. |
| 1 | No coherent theme is apparent; words appear unrelated. |

### 2. Distinctiveness

Is this topic clearly different from every other topic in the model?

| Score | Criterion |
|---|---|
| 3 | Topic is clearly distinct; no significant overlap with other topics. |
| 2 | Partial overlap with one or two other topics but still identifiable. |
| 1 | Topic is a near-duplicate or subset of another topic. |

### 3. Substantive Density

Does the topic carry meaningful content (as opposed to residual noise or
boilerplate)?

| Score | Criterion |
|---|---|
| 3 | Topic captures substantive, domain-relevant content. |
| 2 | Mix of substantive terms and generic or boilerplate language. |
| 1 | Topic is dominated by generic, procedural, or noise terms. |

## Recording

Scores are recorded in a structured table alongside the provisional label and
relevance class assigned in pass 1.  The combined interpretability score for a
topic is the sum of its three dimension scores (range 3–9).  Aggregate
statistics (mean, median, distribution) are reported at the model level.
