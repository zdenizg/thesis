# LDA Hyperparameter Specification

## Purpose

This document freezes the hyperparameters used for all LDA runs (both
baseline and full-pipeline conditions).  No parameter listed here may be
changed after modelling begins without updating this specification and
re-running all affected experiments.

## Frozen hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `alpha` | `'auto'` | Asymmetric, learned from data by Gensim |
| `eta` | `'auto'` | Symmetric, learned from data by Gensim |
| `passes` | `10` | Full passes over the corpus |
| `iterations` | `400` | Maximum E-step iterations per chunk |
| `chunksize` | `2000` | Documents per training chunk |
| `minimum_probability` | `0.01` | Suppress topic weights below this threshold |
| `eval_every` | `None` | Disable in-training perplexity (speed) |
| `random_state` | `42` | Fixed seed for reproducibility |
| `workers` | `3` | Parallel workers for `LdaMulticore` |

## Dictionary filtering

| Parameter | Value |
|---|---|
| `no_below` | 5 (term must appear in at least 5 documents) |
| `no_above` | 0.5 (term must appear in no more than 50 % of documents) |

## Note on stability

A fixed random seed (`random_state=42`) ensures exact reproducibility of any
single run.  During model selection the candidate-*k* solutions near the
coherence optimum are inspected qualitatively for gross topic instability
(e.g., near-duplicate or degenerate topics).  Formal multi-seed stability
analysis (training the same *k* with multiple seeds and measuring topic
alignment across runs) is out of scope for this thesis.
