# Thesis — Deferred / Flagged Items

A running list of methodological points, code improvements, and writing
tasks flagged during planning and implementation but deliberately postponed.
Each entry says what, why it was deferred, and when it should be revisited.

Maintained by Deniz. Update or tick off as items are resolved.

---

## Methodology — preprocessing pipeline

### 1. Phase 4 "coherence claim" originally cut from the methodology
- **Status:** resolved in the methodology draft; no action needed.
- **Context:** Earlier Phase 4 wording claimed lemmatisation "tightens topic
  coherence for the narrative reports that dominate this corpus." Removed
  because unsupported without empirical backing.
- **Revisit:** only if a lemma-vs-non-lemma comparison is added as an LDA
  experiment. Not currently planned.

### 2. Reporting-verbs sensitivity test (`said`, `stated`, `advised`)
- **Status:** planned as optional Stage 5 in the LDA plan.
- **Why:** these three verbs are the most contestable items in
  `ARCHIVE_STOPWORDS` (Phase 4). Testing whether topics improve when they
  are retained gives a defensible answer to an obvious viva question.
- **Revisit:** after Stages 1–4 complete and time remains. If the test
  shows topics are better with them retained, edit
  `pipeline/phase4/scripts/phase4_modeltext.py` and rerun the pipeline.

### 3. Methodology `[X.X]` placeholders
- **Status:** two placeholders remain in `methodology_preprocessing.md`.
- **Location 1:** Phase 6B paragraph — references the chunk-size
  sensitivity analysis section. Fill after Stage 3 completes.
- **Location 2:** the opening paragraph needs to be updated with an
  explicit reference to the Stage 4 baseline findings ("baseline
  modelling with generic preprocessing produced topics dominated by X;
  full-pipeline preprocessing produced topics dominated by Y").
- **Revisit:** during Stage 6 (methodology finalisation).

### 4. Prose edits to methodology after experiments settle
- **Status:** not yet applied. Safer to wait until experiments are done
  so we don't polish around decisions that might change.
- **Specific items:**
  - "drops tokens" → "removes standalone numeric and punctuation-only
    strings" (Phase 2 paragraph).
  - "anchor terms" wording — reframe more neutrally (Phase 4).
  - "topic assignment" → "topic mixture estimation" (Phase 6B retention rule).
  - Phase 1: "without modifying the text" → "without altering retained
    page text" (small precision fix).
  - Reproducibility section: mention what the validation scripts actually
    check (row counts, file-ID continuity, schema) rather than just
    "baseline validation script".
- **Revisit:** during Stage 6, after all experiments are done.

### 5. Phase 1 `merge_missing.py` one-shot script
- **Status:** committed with corrected paths but still a one-shot.
- **Why noted:** does not run via `run_all.py`; documented in README
  Notes section. No action unless someone clones the repo and wants to
  regenerate `JFK_Pages_Merged.csv` from scratch.
- **Revisit:** only if OCR output changes.

---

## Methodology — LDA stage

### 6. Table B vocabulary audit tradeoff
- **Status:** observed during `00_vocab_audit.py` — borderline
  `no_below=5` losses include rare cryptonyms (`amlasii`, `jmdevil`,
  `aeboor`) and Russian names (`moskalev`, `fedoseev`).
- **Why noted:** losing rare proper names and cryptonyms is the
  acknowledged tradeoff at `no_below=5`. Worth one sentence in the
  methodology's Stage 1 writeup under Results.
- **Revisit:** during Stage 6 writeup.

### 7. `update_every=3` rationale
- **Status:** added to `lda_params.md` and decision record.
- **Why noted:** this is a subtle implementation detail (matches the
  M-step schedule that `LdaMulticore(workers=3, chunksize=2000)` would
  have used). Easy viva question. Make sure the rationale appears
  somewhere in the written methodology, not just in the spec files.
- **Revisit:** during Stage 6.

### 8. Strict reproducibility environment variables (Codex suggestion)
- **Status:** not implemented.
- **Why noted:** Codex recommended setting
  `PYTHONHASHSEED=0, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
  MKL_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, NUMEXPR_NUM_THREADS=1`
  before importing NumPy/Gensim for bitwise reproducibility.
- **Revisit:** before the final thesis-reported models are trained
  (probably Stage 2 final run). Include in an appendix or reproducibility
  note.

### 9. Record fitted `alpha` and `eta` per model
- **Status:** requested as part of script 02's logging.
- **Why noted:** evidence that `alpha='auto'` actually fitted a non-trivial
  asymmetric prior, not a degenerate solution.
- **Revisit:** during Stage 6 — check coherence CSVs and include a
  summary table in the methodology or appendix.

### 10. Cold War vocabulary known limitations
- **Status:** accepted into the pre-registered vocabulary.
- **Why noted:** the term `source` is also a generic English word and
  will inflate the `tradecraft` category score in many non-intelligence
  contexts. Acknowledge as a known limitation when writing up Stage 2
  results.
- **Revisit:** during Stage 2 write-up.

### 11. Borderline blacklist terms in Phase 6B
- **Status:** ✓ RESOLVED. At k=25: `originator`, `docid`, `decl`, `cfr` confirmed absent from all topic top-words — blacklisting correct. Topic 22 is the metadata-residue topic from `ref`, `cite`, `per` removal. Decision made in item #29: accept and report.
- **Why noted:** ChatGPT flagged `originator`, `docid`, `decl`, `cfr` as
  borderline. If they were wrongly blacklisted, related
  document-routing themes may look weird in the full-pipeline topics.
  Check during normal topic inspection in Stage 2.
- **Revisit:** resolved.

### 12. `log_perplexity` interpretation
- **Status:** computed by `02_coherence_sweep_broad.py` but reported
  descriptively only.
- **Why noted:** log-perplexity is computed on the training corpus,
  so it's overfitted. Per the plan, it is not decisive for k-selection.
  If perplexity and coherence disagree sharply, discuss the disagreement
  rather than using it to break ties.
- **Revisit:** during Stage 1 results interpretation.

---

## Repository / tooling

### 13. Phase 5 `pages_phase5_summary.json` transient shift
- **Status:** reverted on the working tree; committed version stands.
- **Why noted:** a Phase 5 re-run at some point produced a one-page shift
  (78,022 → 78,023). Root cause not fully diagnosed beyond "an upstream
  regeneration probably changed inputs". If a future pipeline rerun
  produces the same shift reproducibly, that's a methodology numbers
  update.
- **Revisit:** if the number shifts again after any deliberate pipeline
  rerun.

### 14. `requirements.txt` dependency versions
- **Status:** pinned.
- **Why noted:** NLTK data (WordNet, stopwords, punkt, punkt_tab) is
  not a pip dependency and must be fetched separately via
  `pipeline/setup_nltk.py`. README covers this.
- **Revisit:** only if moving to a different Python or NLTK version.

---

## Writing / defensibility

### 15. APA 7 citations
- **Status:** three in the methodology: Bird, Klein, & Loper (2009) for
  NLTK; Marcus, Santorini, & Marcinkiewicz (1993) for Penn Treebank;
  Fellbaum (1998) for WordNet.
- **Why noted:** no pandas/NumPy citations (would look odd in this field).
  Add Gensim citation during Stage 6: Rehurek & Sojka (2010), "Software
  Framework for Topic Modelling with Large Corpora".
- **Revisit:** Stage 6.

### 16. Thesis title vs repo description
- **Status:** README now mentions the thesis title: "Topic Modeling and
  Thematic Analysis on JFK Assassination Files".
- **Why noted:** keep wording consistent between README, methodology,
  and thesis chapter titles.
- **Revisit:** during Stage 6.

### 17. Furkan Demir attribution
- **Status:** README states code authored by Zeynep Deniz Güvenol;
  thesis co-authored with Furkan Demir. Methodology does not mention
  him (implicitly first-person).
- **Why noted:** confirm with supervisor whether methodology wording
  should reflect joint authorship or stay first-person.
- **Revisit:** before thesis submission.

---

## Process / session continuity

### 18. New conversation onboarding checklist
If starting a new Claude/ChatGPT/Codex session, provide:
- This `deferred_items.md` file.
- `methodology_preprocessing.md` (current draft).
- `lda_plan.md` (current plan).
- Latest commit hash from `zdenizg/thesis` repo.
- Current stage of work (which LDA script is in flight, which is done).

Without these, the assistant will not remember prior decisions and may
suggest things that contradict already-committed choices.

### 19. Verify learned alpha is non-trivial in each k
- **Status:** logged per k via alpha_min / alpha_max columns in the sweep CSVs.
- **Why noted:** if any k shows degenerate equal alphas (alpha_min ≈ alpha_max), LDA failed to fit a meaningful asymmetric prior and that k should not drive k-selection.
- **Revisit:** during Stage 1 k-selection — check alpha columns before treating any k as a candidate.

### 20. Delete stale model files before re-running a sweep after spec changes
- **Status:** process guideline; not automated.
- **Why noted:** mixing models trained under different hyperparameters in a single sweep is a methodological error that is easy to make by accident.
- **Revisit:** whenever a spec change in `lda_params.md` or `baseline_spec.md` requires a rerun.

### 21. Source-of-truth when terminal output looks stale
- **Status:** process note.
- **Why noted:** session confusion around stale terminal output — when in doubt, cat the file on disk as the source of truth, not scrollback.
- **Revisit:** as needed during multi-step work.

### 22. Fine-sweep range decision rationale
- **Status:** range locked at k=[18, 40] steps of 1.
- **Why noted:** trimming below 18 excludes a region where coherence is still rising; going above 40 wastes runs on known-lower values. Second ChatGPT review with corrected broad-sweep numbers confirmed this over alternatives [15, 40] and [20, 40].
- **Revisit:** Stage 6 writeup — include one-sentence justification in the Results chapter k-selection subsection.

### 23. Stage 1 methodology sentence for the vocabulary audit
- **Status:** audit complete (vocab_audit.md), but no sentence yet in methodology.
- **Why noted:** one sentence acknowledging the no_below=5 tradeoff (rare cryptonyms and Russian names lost) strengthens the preprocessing defence.
- **Revisit:** Stage 6. Merges with item #6.

### 24. Fine-sweep results and k-selection candidates
- **Status:** ✓ RESOLVED. After Phase 4 stopword fix, fine sweep rerun produced new top 3: k=29 (0.6188), k=25 (0.6128), k=28 (0.6099). Broad sweep showed k=15 (0.6261) as highest overall. Script 04 inspection of k=15, k=25, k=29 followed by independent ChatGPT consultation: k=15 rejected (underfits — merges tax forms with assassination discussion), k=29 rejected (overfits — fragments Black Panthers and metadata topics), k=25 selected as final model. Coherence difference between k=25 and k=29 is only 0.006; k=25 has cleaner topic separation.
- **Why noted:** these three are within 0.0005 c_v — effectively tied. Script 04 will inspect k=23, k=21, and k=33 (secondary peak). Final k chosen by interpretability rubric adjudication.
- **Revisit:** resolved. Final k = 25.

### 25. Script 04: check topic top-20 words for OCR noise
- **Status:** ✓ RESOLVED. After Phase 4 stopword fix and noise checker improvement (6-criterion allowlist with known_entities.yml), k=25 inspection shows 1.60% flagged noise rate (8 words out of 500), of which real OCR noise is only 2 tokens (`lifeat`, `gerende`). Pipeline confirmed clean.
- **Why noted:** the pipeline's functional cleanliness is only confirmed if no OCR garbage appears in topic top-words. If garbled tokens show up, add a spell-correction or dictionary-filtering step between Phase 3 and Phase 4, then rerun everything.
- **Revisit:** resolved by k=23 inspection; no further action needed.

### 26. dict_hit_rate drop across pipeline (0.776 → 0.742)
- **Status:** observed in cleanliness_report.py output. Explained by stopword removal reducing the proportion of common English words.
- **Why noted:** an examiner could ask why dictionary hit rate went down after cleaning. The answer (stopwords are real English words; domain terms like oswald, cia, kgb are not in /usr/share/dict/words) should appear in the methodology when the evaluation results are written up.
- **Revisit:** Stage 6 writeup, alongside the Zimmermann et al. citation.

### 27. Add `n't` to ARCHIVE_STOPWORDS in Phase 4
- **Status:** ✓ RESOLVED. Added `n't`, `dr.`, `jr.`, `mrs.`, `sr.` to
  ARCHIVE_STOPWORDS. Also added a regex filter rejecting single-letter-
  period tokens (^[a-z]\.$) to catch name initials (j., w., p., etc.).
  Pipeline rerun from Phase 4 completed. New corpus: 4,010 docs, 54,025
  vocab. New coherence winner shifted from k=23 to k=25.
- **Why noted:** not OCR noise, but a tokenisation artefact with zero topical value. Adding it to ARCHIVE_STOPWORDS would require rerunning Phase 4 onward + rebuilding dictionary + retraining all LDA models.
- **Revisit:** resolved; follow-on numbers work tracked in item #30.

### 28. Spanish-language topic (Topic 2 at k=23)
- **Status:** confirmed at k=25 as Topic 4. Entirely Spanish function words. Stable across all candidate k values tested (15, 25, 29). Will be reported in Results as a corpus feature.
- **Why noted:** the non-English Phase 5 filter (5% non-ASCII) misses Spanish written in ASCII. This topic is a real feature of the corpus, not a pipeline failure. Report it honestly in Results as evidence of the filter's acknowledged limitation.
- **Revisit:** Stage 6 writeup only.

### 29. Topic 13 cable-routing metadata residue
- **Status:** confirmed at k=25 as Topic 22 (info, message, ref, city, unit, reproduction, per, issuing, letter, press). Metadata-residue topic appears at every k tested. Decision: option (b) — keep terms off blacklist, accept one metadata topic, report it. One admin topic out of 25 is expected and defensible.
- **Why noted:** `ref`, `cite`, `per` were deliberately removed from the Phase 6B blacklist because they carry topical content in some contexts. Topic 13 shows they also carry routing-metadata signal. Two options: (a) re-add them to the blacklist and accept losing some topical uses, or (b) keep them off the blacklist and accept one metadata-residue topic. Option (b) is defensible — one administrative topic out of 23 is expected.
- **Revisit:** resolved. Report in Results chapter.

### 30. Post-stopword-fix corpus numbers differ from methodology text
- **Status:** pending. Methodology currently cites 4,049 docs, 54,058
  vocab, 78,022 retained pages, median retention 55.4%, k=23. After the
  Phase 4 stopword fix, the correct numbers are: 4,010 docs, 54,025
  vocab, 78,012 retained pages, median retention 54.4%, k=25.
- **Why noted:** all numbers in the thesis LaTeX, the evaluation tables,
  and the coherence sweep reports need updating once experiments settle.
- **Revisit:** Stage 6 (methodology finalisation), after Stages 3-4
  complete.

### 31. Noise checker improved with 6-criterion allowlist
- **Status:** ✓ RESOLVED. Script 04 noise checker updated to use 6-criterion conjunction: dictionary check, Cold War vocabulary, known_entities.yml (66 entries: 25 persons, 15 organizations, 9 places, 17 abbreviations), abbreviation regex, Spanish stopwords (31 words), and high document frequency (>=100). Previous checker over-flagged proper names and abbreviations at ~17%; new checker flags ~1.6%.
- **Why noted:** the improvement is methodologically significant — it distinguishes real OCR noise from domain vocabulary the dictionary doesn't contain.
- **Revisit:** resolved.

### 32. Two-pass topic labelling completed for k=25
- **Status:** ✓ RESOLVED. All 25 topics labelled. Pass 1 (top-words only) and Pass 2 (representative documents) completed. Results recorded in topic_labelling_k25.xlsx. Final tier counts after Pass 2: 6 Core, 11 Adjacent, 6 Admin, 2 Other. Key finding: 17/25 topics (68%) show Cold War relevance. Notable Pass 2 corrections: Topic 19 upgraded Other→Core (CIA assassination-planning testimony hidden behind generic conversational top-words), Topic 0 upgraded Other→Adjacent (Hampton raid forensics), Topic 6 upgraded Adjacent→Core (JMWAVE), Topic 16 downgraded Adjacent→Other (Riha witness interviews), Topic 10 downgraded Other→Admin (FBI routing templates).
- **Why noted:** the two-pass protocol proved its value — 8 of 25 topics changed tier between Pass 1 and Pass 2.
- **Revisit:** resolved. Results feed into thesis Chapter 5.

### 33. Script 05 Cold War relevance scores for k=25
- **Status:** ✓ RESOLVED. Cold War relevance computed using 30-term vocabulary across 5 categories. Results: 1 topic scored as Cold-War-core by the automated heuristic (Topic 24, overall=0.164), 11 as adjacent (overall 0.05–0.15), 13 as low-overall (<0.05). Manual labelling (item #32) substantially reclassified these: 6 Core and 11 Adjacent after human inspection. The gap between automated scores and manual labels confirms the vocabulary is a conservative first-pass indicator, not a classifier.
- **Why noted:** the discrepancy between automated and manual Cold War classification is itself a finding for the Results chapter.
- **Revisit:** resolved. Report both automated and manual scores.

---

## How to use this file

- Add new items as they come up. Format: what, why deferred, when to revisit.
- Tick items off (`[x]`) or move to a "Done" section at the bottom when
  resolved.
- Store this in the repo under `docs/` or keep a local copy. It is not
  thesis-facing; it is a working document.
