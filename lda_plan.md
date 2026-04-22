# LDA Implementation Plan

## Scope and research-question mapping

The main research question asks to what extent thematic patterns in the 2025 JFK declassified files reflect Cold War intelligence concerns, as identified through topic modelling. Subquestion 1 asks which latent themes emerge and how interpretable they are. Subquestion 2 asks to what extent these themes correspond to Cold War concerns. The modelling work must therefore produce: (a) a defensible final topic model, (b) evidence that the preprocessing pipeline meaningfully improved topic quality over a minimal baseline, (c) empirical justification for the 5,000-token chunking threshold, and (d) a transparent, replicable first-pass indicator of Cold War relevance that structures, but does not replace, manual topic interpretation.

## Relationship between `pipeline/` and `lda/`

The pipeline is frozen in its current state and produces the canonical 4,049-document corpus that the methodology describes. Modelling experiments live under `lda/` and read frozen pipeline outputs as input. Where sensitivity analyses require rerunning parts of the pipeline under alternate settings (different chunk sizes, different stopword list), those reruns happen inside `lda/scripts/` using isolated output directories; they do not modify files under `pipeline/`. The pipeline is only edited if a sensitivity result demonstrates that a preprocessing choice was wrong, in which case the change is committed separately with an updated methodology. This separation keeps the canonical corpus reproducible and keeps experiments traceable.

## Directory layout

```
thesis/
  pipeline/          (existing preprocessing pipeline, frozen)
  lda/
    scripts/         (all modelling scripts)
    outputs/         (models, corpora, large artefacts — gitignored)
    reports/         (evaluation tables, topic inspections — committed)
    specs/           (pre-registered specifications — committed)
```

## Pre-registered specifications (written before any modelling runs)

### `lda/specs/baseline_spec.md`
Exact preprocessing applied by the minimal baseline: lowercase, NLTK Penn Treebank tokenisation, NLTK English stopwords, WordNet lemmatisation, document aggregation by `file_id`, chunking at 5,000 tokens, drop documents under 50 tokens. Exact preprocessing excluded: Phase 2 archive regex boilerplate removal, Phase 3 line filtering, Phase 5 page-quality filtering, Phase 6B blacklist. Both baseline and full pipeline are evaluated at the same k and with the same metrics. Committed before any script runs.

### `lda/specs/lda_params.md`
Fixed LDA hyperparameters used throughout: `alpha='auto'`, `eta='auto'`, `passes=10`, `iterations=400`, `chunksize=2000`, `update_every=3`, `minimum_probability=0.01`, `eval_every=None`, `random_state=42`. Training uses `gensim.models.LdaModel` (single-process) because `alpha='auto'` is not supported by `LdaMulticore`; see `lda_params_decision_record.md`. The final reported model is trained with this fixed seed for reproducibility. Candidate solutions near the coherence optimum are inspected qualitatively for gross instability in topic composition before a final k is selected; a formal multi-seed stability experiment is out of scope.

### `lda/specs/cold_war_vocabulary.yml` (machine-readable source) and `lda/specs/cold_war_vocabulary.md` (human-readable appendix)
Reference vocabulary for the relevance heuristic, grouped into five categories: geopolitical actors and places (cuba, soviet, russia, mexico, vietnam, germany, berlin), intelligence agencies (cia, fbi, kgb, nsa, dgi), tradecraft terms (surveillance, agent, source, operation, covert, defector, handler), diplomatic and political vocabulary (embassy, consulate, ambassador, defection, asylum, communist, party), and named actors (oswald, ruby, castro, khrushchev). Each term carries a one-line source annotation. The YAML file is the single source of truth consumed by `05_cold_war_relevance.py`; the markdown file is regenerated from the YAML for the thesis appendix. The score is computed per category as well as overall, so that topics scoring high on broad bureaucratic vocabulary alone can be distinguished from topics with genuine geopolitical or actor-level content.

### `lda/specs/interpretability_rubric.md`
Three-dimensional rubric for manual topic interpretation, scored on a 1–3 scale per topic: semantic coherence of top-20 words (do they describe a coherent theme), distinctiveness from other topics (does this topic say something other topics do not), and substantive density (proportion of top words that carry substantive meaning versus administrative or filler vocabulary). Rubric scores are recorded during pass 1 of the labelling protocol (after inspecting the top-20 words, before reading representative documents) and are not revised in pass 2, so that document reading influences topic labels but not rubric scores. Scores are recorded in the topic inspection reports and combined with Cold War relevance scores in final analysis.

### `lda/specs/topic_labelling_protocol.md`
Labels are assigned in two passes. First pass: after inspecting only the top-20 words, assign a provisional label and one of four relevance classes (Cold-War-core, Cold-War-adjacent, administrative, other). Second pass: after reading three representative documents per topic, confirm or revise. Both rounds are recorded so that any revisions are auditable.

## Stage 1 — Corpus construction and k-selection

### `scripts/01_build_dictionary.py`
Input: `pipeline/phase6/data/documents_final.csv` (4,049 documents, lemmatised column). Output: `lda/outputs/dictionary.gensim`, `lda/outputs/corpus.mm`. Tokenises by whitespace (text is pre-cleaned), builds a Gensim `Dictionary`, and filters extremes with `no_below=5` and `no_above=0.5`. The `no_below=5` threshold excludes tokens appearing in fewer than five documents, removing OCR-induced hapax and near-hapax noise without eliminating rare but substantive terms. The `no_above=0.5` threshold is a conservative high-frequency filter for residual widely-distributed vocabulary; it is not claimed to be self-evidently correct, but is standard practice for LDA dictionary construction and was fixed before any modelling results were seen. Both values are frozen in this script and are not revisited after seeing coherence results.

### `scripts/02_coherence_sweep_broad.py`
Input: dictionary and corpus from step 01. Output: `lda/reports/coherence_broad.csv`, `lda/reports/coherence_broad.png`. Trains LDA across `k ∈ {5, 10, 15, 20, 25, 30, 35, 40, 50, 60}`, computes c_v coherence and log-perplexity, saves each model to `lda/outputs/models/`. Prints the top three candidate k values by coherence.

### `scripts/03_coherence_sweep_fine.py`
Input: the best region identified in step 02. Output: `lda/reports/coherence_fine.csv`, `lda/reports/coherence_fine.png`. Trains LDA at every integer k in the range [18, 40], selected based on the broad sweep results (plateau from k=20 to k=40 with two local peaks at k=20 and k=35; buffer below k=20 to resolve the rising region from k=15). This two-pass design resolves both candidate regions at unit resolution.

**Decision rule:** final k is chosen by c_v coherence as the primary signal, with manual interpretability from Stage 2 used to adjudicate between candidates within 0.02 of the coherence maximum. Log-perplexity is reported descriptively but does not drive the choice; if perplexity and coherence disagree sharply, the disagreement is discussed rather than used to break ties.

## Stage 2 — Main model, inspection, and relevance

### `scripts/04_inspect_topics.py`
Input: a trained model and candidate k. Outputs to `lda/reports/`: `topics_k{K}_top_words.csv`, `topics_k{K}_representative_docs.csv` (top five documents per topic by topic weight; these may be chunked sub-documents produced by Phase 6B rather than whole original files, which is noted in the inspection output), `topics_k{K}_pyldavis.html`. Also applies the interpretability rubric and the two-pass labelling protocol, writing results to `topics_k{K}_interpretation.csv` (topic_id, label_pass_1, label_pass_2, rubric_coherence, rubric_distinctiveness, rubric_substantive, notes).

### `scripts/05_cold_war_relevance.py`
Input: `topics_k{K}_top_words.csv` and `lda/specs/cold_war_vocabulary.yml`. Output: `lda/reports/cold_war_relevance_k{K}.csv` with columns for category-level scores (geopolitical, agency, tradecraft, diplomatic, named_actor) and an overall score. Relevance is a continuous score in [0, 1] per category, not a binary label. The category breakdown exists precisely so that a topic scoring high only on `tradecraft` but not on `geopolitical` or `named_actor` is visibly distinguishable from a topic scoring high across multiple categories. The output of this script is presented in the thesis as a structured first-pass indicator, combined with the manual labelling from step 04; it is not framed as an objective classifier of Cold War content.

## Stage 3 — Chunk-size sensitivity

### `scripts/06_chunk_sensitivity.py`
Input: `pipeline/phase6/data/documents_for_modeling.csv` (Phase 6A output, pre-chunking). Output: `lda/reports/chunk_sensitivity.csv`, `lda/reports/chunk_sensitivity_topics.md`, `lda/reports/chunk_sensitivity_summary.md`. For each chunk size in {3000, 5000, 10000}, reruns Phase 6B chunking and filtering logic in an isolated output directory, builds dictionary and corpus, trains LDA at the final k from Stage 2 with the frozen parameters, computes c_v coherence, and produces top-20 words per topic.

**Decision principle (stated before running):** the preferred chunk size balances coherence and interpretability while avoiding unnecessary fragmentation of long documents. If coherence differences across the three settings are minor — a pragmatic tolerance band of roughly 0.02 c_v, below which coherence differences are not treated as decisive — interpretability rubric scores and fragmentation counts become the deciding criteria; otherwise the setting with clearly higher coherence is preferred. Results write into the methodology's Section [X.X] as empirical justification for the chosen threshold.

## Stage 4 — Minimal baseline comparison

### `scripts/07_build_baseline_corpus.py`
Input: `pipeline/phase1/data/pages_phase1_structural.csv` (post-empty-page-removal, pre-archival-cleaning). Output: `lda/outputs/baseline_dictionary.gensim`, `lda/outputs/baseline_corpus.mm`, `lda/outputs/baseline_documents.csv`. Implements exactly the preprocessing in `baseline_spec.md`; no archive-specific steps are applied. The baseline builds its own independent dictionary from the baseline document set using the same filtering thresholds as the full pipeline (`no_below=5`, `no_above=0.5`), so that differences observed in the comparison are attributable to preprocessing and not to unequal vocabulary filtering. Aggregation precedes lemmatisation is irrelevant here — the baseline applies lemmatisation at the token level during tokenisation, before aggregation, matching the full pipeline's ordering.

### `scripts/08_train_baseline_lda.py`
Input: baseline dictionary and corpus. Output: `lda/outputs/models/baseline_lda_k{K}.gensim`, `lda/reports/baseline_coherence.csv`. Trains LDA at the final k from Stage 2, with the identical frozen LDA hyperparameters specified in `lda_params.md` (same alpha, eta, passes, iterations, chunksize, seed).

### `scripts/09_compare_baseline.py`
Input: baseline model and full-pipeline model at the chosen k. Output: `lda/reports/baseline_vs_full.md`, `lda/reports/baseline_vs_full_table.csv`. Three comparison dimensions: c_v coherence, interpretability rubric applied to both sets of topics, and metadata contamination (count of tokens from the Phase 6B blacklist appearing in the top-20 words of each model's topics). Side-by-side top-20-word listings support qualitative reading.

## Stage 5 — Reporting-verbs sensitivity (optional)

Run only if Stages 1–4 are complete and time remains.

### `scripts/10_reporting_verbs_test.py`
Input: `pipeline/phase3/data/pages_phase3_linefiltered.csv`. Output: `lda/reports/reporting_verbs_test.md`. Reruns Phase 4 onward with `said`, `stated`, and `advised` **excluded from the archive stopword list**, so that these verbs are retained in the modelling text (the current pipeline removes them). Trains LDA at the chosen k, compares coherence and top-20 words to the full-pipeline model. If topics are meaningfully better with the verbs retained, the recommendation is to edit `pipeline/phase4/scripts/phase4_modeltext.py` accordingly in a follow-up commit; if not, the test confirms the current choice and goes into a methodology footnote or appendix.

## Stage 6 — Methodology and Results finalisation

- Replace the `[X.X]` placeholder in the methodology with the Stage 3 chunk-size results.
- Update the methodology's opening paragraph with an explicit reference to the Stage 4 baseline findings (e.g. "baseline modelling with generic preprocessing produced topics dominated by X; full-pipeline preprocessing produced topics dominated by Y; see Results Section [Y.Y]").
- Apply the prose edits already agreed (remove `drops tokens`, soften non-English language, neutralise anchor-term framing, etc.).
- Draft Results chapter: k-selection (Stages 1–2), final topic interpretation and Cold War relevance (Stage 2), chunk-size sensitivity (Stage 3) if placed in Results rather than Methods, baseline comparison findings (Stage 4), reporting-verbs sensitivity (Stage 5) if run.

## Dependencies to add

`gensim`, `pyLDAvis`, `matplotlib`. Pin current installed versions in `requirements.txt`.

## Thesis placement

| Artefact | Chapter |
|---|---|
| Preprocessing methodology | Methods |
| Baseline specification | Methods or Appendix |
| Cold War reference vocabulary | Methods or Appendix |
| Interpretability rubric | Methods or Appendix |
| Topic-labelling protocol | Methods or Appendix |
| k-selection coherence sweep | Results |
| Final topic inspection and labels | Results |
| Cold War relevance analysis | Results |
| Chunk-size sensitivity results | Methods Section [X.X] |
| Baseline comparison design | Methods |
| Baseline comparison findings | Results |
| Reporting-verbs sensitivity | Methods footnote or Appendix |
