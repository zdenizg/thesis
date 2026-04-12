# Topic Labelling Protocol

## Purpose

This document specifies the two-pass procedure used to assign human labels
and relevance classes to each LDA topic.  The protocol is designed to
separate word-level impressions (pass 1) from document-level confirmation
(pass 2) so that each source of evidence is recorded independently.

## Relevance classes

Every topic is assigned to exactly one of the following classes:

| Class | Definition |
|---|---|
| **Cold-War-core** | Topic directly concerns Cold War intelligence, espionage, or geopolitical conflict as reflected in the JFK files. |
| **Cold-War-adjacent** | Topic relates to Cold War context (e.g., domestic politics, media coverage, legal proceedings) but is not primarily about intelligence or geopolitics. |
| **Administrative** | Topic captures bureaucratic, procedural, or organisational content (e.g., filing metadata, routing slips, form language). |
| **Other** | Topic does not fit the above categories (e.g., OCR noise, residual boilerplate). |

## Pass 1 — Word-level labelling

**Input:** Top-20 words for each topic (by weight), plus the interpretability
rubric (`interpretability_rubric.md`).

**Procedure:**

1. For each topic, inspect the ranked top-20 words.
2. Assign an **interpretability score** on each of the three rubric dimensions
   (semantic coherence, distinctiveness, substantive density) using the 1–3
   scale defined in `interpretability_rubric.md`.
3. Assign a **provisional label** — a short descriptive phrase (e.g.,
   "Cuba / DGI operations", "Filing metadata").
4. Assign a **relevance class** (Cold-War-core, Cold-War-adjacent,
   administrative, or other).
5. Record all scores, labels, and classes before proceeding to pass 2.

**Constraint:** Interpretability scores assigned in pass 1 are final and are
not revised in pass 2.

## Pass 2 — Document-level confirmation

**Input:** Three representative documents per topic (highest topic-weight
documents), plus the pass-1 labels and classes.

**Procedure:**

1. For each topic, read the three documents with the highest posterior weight
   for that topic.
2. **Confirm or revise** the provisional label from pass 1.  If revised,
   record both the original and revised labels with a brief justification.
3. **Confirm or revise** the relevance class from pass 1.  If revised,
   record both the original and revised classes with a brief justification.

If pass 2 revises the relevance class for a topic, this is recorded as
evidence that top-20 words alone were insufficient for labelling that topic;
the overall frequency of such revisions is reported as a methodological
finding in the Results chapter.

## Output

The final labelling table contains, for each topic:

- Topic number (*k* index).
- Pass-1 interpretability scores (three dimensions + sum).
- Pass-1 provisional label and relevance class.
- Pass-2 confirmed/revised label and relevance class.
- Revision flag and justification (if any change occurred between passes).
