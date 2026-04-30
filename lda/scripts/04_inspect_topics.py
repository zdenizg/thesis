"""
04 — Topic Inspection
=======================
Input:  lda/outputs/dictionary.gensim
        lda/outputs/corpus.mm
        lda/outputs/models/lda_k{K}.gensim
        pipeline/phase6/data/documents_final.csv
        lda/specs/cold_war_vocabulary.yml
        lda/specs/known_entities.yml
Output: lda/reports/topics_k{K}_top_words.csv
        lda/reports/topics_k{K}_representative_docs.csv
        lda/reports/topics_k{K}_pyldavis.html            (best-effort)
        lda/reports/topics_k{K}_noise_check.csv

For a given candidate k, produces the raw material for final k
selection and for the Results chapter:

  * top-20 words per topic (ranked by weight),
  * five representative documents per topic (highest posterior
    topic weight — these may be chunked sub-documents produced by
    Phase 6B; their `file_id` may carry a `_chunk_NNN` suffix),
  * an interactive pyLDAvis HTML (optional — degrades gracefully
    if the dependency import fails),
  * a noise-check CSV that flags a top-20 word only when it fails
    every allowlist: not in /usr/share/dict/words, not in the Cold
    War reference vocabulary, not in the known-entities allowlist,
    not matching an abbreviation pattern (dr., u.s., …), not a
    Spanish function word, and appearing in fewer than 100
    documents. Short / non-alpha flags are recorded as separate
    informational columns.

This script does NOT apply the interpretability rubric or the
two-pass labelling protocol.  Those are manual steps performed by
the analyst on the output of this script.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LDA_DIR = SCRIPT_DIR.parent
REPO_ROOT = LDA_DIR.parent

DICT_PATH = LDA_DIR / "outputs" / "dictionary.gensim"
CORPUS_PATH = LDA_DIR / "outputs" / "corpus.mm"
MODEL_DIR = LDA_DIR / "outputs" / "models"
INPUT_CSV = REPO_ROOT / "pipeline" / "phase6" / "data" / "documents_final.csv"
VOCAB_YAML = LDA_DIR / "specs" / "cold_war_vocabulary.yml"
ENTITIES_YAML = LDA_DIR / "specs" / "known_entities.yml"
REPORTS_DIR = LDA_DIR / "reports"
SYSTEM_DICT = Path("/usr/share/dict/words")

TEXT_COLUMN = "document_text_lemma"
ID_COLUMN = "file_id"

TOP_N_WORDS = 20
N_REP_DOCS = 5
PREVIEW_CHARS = 200

# Noise-check thresholds and patterns
MIN_DOC_FREQ = 100

# Honorifics and short period-terminated abbreviations (dr., jr., sr., mrs.)
ABBREV_HONORIFIC_RE = re.compile(r"^[a-z]{1,4}\.$")
# Period-separated acronyms (u.s., u.s.s.r., a.k.a.)
ABBREV_MULTIDOT_RE = re.compile(r"^([a-z]+\.){2,}$")

# Top-30 Spanish function words — intercepted Cuban/Mexican cables
# occasionally leak these into the English corpus.
SPANISH_STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
    "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo",
    "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "e",
    "si",
}


def model_path(k: int) -> Path:
    return MODEL_DIR / f"lda_k{k}.gensim"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_system_dictionary() -> set[str]:
    if not SYSTEM_DICT.exists():
        print(f"WARNING: {SYSTEM_DICT} not found — dict-hit check will flag "
              "everything as out-of-dictionary.", file=sys.stderr)
        return set()
    with open(SYSTEM_DICT, "r", encoding="utf-8", errors="ignore") as f:
        return {line.strip().lower() for line in f if line.strip()}


def load_cold_war_vocab() -> set[str]:
    with open(VOCAB_YAML, "r") as f:
        data = yaml.safe_load(f)
    terms: set[str] = set()
    for category, entries in data.items():
        for entry in entries:
            terms.add(entry["term"].lower())
    return terms


def load_known_entities() -> set[str]:
    """Flatten all categories in known_entities.yml into a single lowercase set."""
    if not ENTITIES_YAML.exists():
        print(f"WARNING: {ENTITIES_YAML} not found — entity allowlist empty.",
              file=sys.stderr)
        return set()
    with open(ENTITIES_YAML, "r") as f:
        data = yaml.safe_load(f) or {}
    entities: set[str] = set()
    for entries in data.values():
        for entry in entries:
            entities.add(str(entry).lower())
    return entities


def is_abbreviation(word: str) -> bool:
    """True if the token looks like an abbreviation (dr., u.s., etc.)."""
    return bool(ABBREV_HONORIFIC_RE.match(word) or ABBREV_MULTIDOT_RE.match(word))


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------
def build_top_words_csv(lda: LdaModel, k: int) -> pd.DataFrame:
    rows = []
    for topic_id in range(k):
        pairs = lda.show_topic(topic_id, topn=TOP_N_WORDS)
        row: dict = {"topic_id": topic_id}
        for i, (word, weight) in enumerate(pairs, 1):
            row[f"word_{i}"] = word
            row[f"weight_{i}"] = round(float(weight), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def build_representative_docs(
    lda: LdaModel,
    corpus: list,
    df_docs: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """For each topic, rank documents by posterior topic weight and
    return the top N_REP_DOCS."""
    # topic_weights[topic_id] -> list of (doc_idx, weight)
    topic_weights: list[list[tuple[int, float]]] = [[] for _ in range(k)]

    for doc_idx, bow in enumerate(tqdm(corpus, desc="doc-topic posterior")):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        for topic_id, weight in dist:
            topic_weights[topic_id].append((doc_idx, float(weight)))

    rows = []
    for topic_id in range(k):
        ranked = sorted(topic_weights[topic_id], key=lambda x: x[1], reverse=True)
        for doc_idx, weight in ranked[:N_REP_DOCS]:
            row = df_docs.iloc[doc_idx]
            text = row.get(TEXT_COLUMN, "")
            if not isinstance(text, str):
                text = ""
            rows.append({
                "topic_id": topic_id,
                "doc_index": int(doc_idx),
                "file_id": row.get(ID_COLUMN, ""),
                "topic_weight": round(weight, 6),
                "text_preview": text[:PREVIEW_CHARS],
            })
    return pd.DataFrame(rows)


def build_noise_check(
    lda: LdaModel,
    k: int,
    dict_words: set[str],
    cw_vocab: set[str],
    entities: set[str],
    doc_freqs: dict[str, int],
) -> pd.DataFrame:
    """A top-word is flagged as noise only if it fails every safety net:
    not in the system dictionary, not in the Cold War vocab, not a known
    entity, not an abbreviation pattern, not a Spanish function word, and
    not appearing in at least MIN_DOC_FREQ documents."""
    rows = []
    for topic_id in range(k):
        for word, weight in lda.show_topic(topic_id, topn=TOP_N_WORDS):
            wl = word.lower()
            too_short = len(word) < 3
            non_alpha = not word.isalpha()
            in_dict = wl in dict_words
            in_cw = wl in cw_vocab
            in_entities = wl in entities
            is_abbrev = is_abbreviation(wl)
            is_spanish = wl in SPANISH_STOPWORDS
            df_count = int(doc_freqs.get(wl, 0))
            high_df = df_count >= MIN_DOC_FREQ
            flagged = not (
                in_dict or in_cw or in_entities
                or is_abbrev or is_spanish or high_df
            )
            rows.append({
                "topic_id": topic_id,
                "word": word,
                "weight": round(float(weight), 6),
                "in_dictionary": in_dict,
                "in_cold_war_vocab": in_cw,
                "in_known_entities": in_entities,
                "is_abbreviation": is_abbrev,
                "is_spanish_stopword": is_spanish,
                "doc_frequency": df_count,
                "high_doc_frequency": high_df,
                "too_short": too_short,
                "non_alpha": non_alpha,
                "flagged_as_noise": flagged,
            })
    return pd.DataFrame(rows)


def try_pyldavis(lda: LdaModel, corpus: list, dictionary: Dictionary,
                 html_path: Path) -> bool:
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensim_models
        vis = gensim_models.prepare(lda, corpus, dictionary)
        pyLDAvis.save_html(vis, str(html_path))
        return True
    except Exception as exc:
        print(f"\nWARNING: pyLDAvis output skipped ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an LDA model for a given k")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of topics (model must already exist under lda/outputs/models/)")
    args = parser.parse_args()
    k = args.k

    separator = "=" * 60

    print(separator)
    print(f"04 — Topic Inspection  (k = {k})")
    print(separator)

    # ------------------------------------------------------------------
    # 1. Load everything
    # ------------------------------------------------------------------
    mp = model_path(k)
    if not mp.exists():
        print(f"ERROR: model file not found: {mp}", file=sys.stderr)
        sys.exit(1)

    print("\nLoading dictionary ...")
    dictionary = Dictionary.load(str(DICT_PATH))
    print(f"  Vocabulary: {len(dictionary):,}")

    print("Loading corpus ...")
    mm = MmCorpus(str(CORPUS_PATH))
    corpus = list(mm)
    print(f"  Documents: {len(corpus):,}")

    print(f"Loading model {mp.name} ...")
    lda = LdaModel.load(str(mp))

    print("Loading document metadata ...")
    df_docs = pd.read_csv(INPUT_CSV, usecols=[ID_COLUMN, TEXT_COLUMN],
                          low_memory=False)
    df_docs[TEXT_COLUMN] = df_docs[TEXT_COLUMN].fillna("")
    if len(df_docs) != len(corpus):
        print(f"WARNING: document count mismatch (csv={len(df_docs)}, "
              f"corpus={len(corpus)})", file=sys.stderr)

    print("Loading system word dictionary ...")
    dict_words = load_system_dictionary()
    print(f"  Entries: {len(dict_words):,}")

    print("Loading Cold War vocabulary ...")
    cw_vocab = load_cold_war_vocab()
    print(f"  Terms: {len(cw_vocab)}")

    print("Loading known entities ...")
    entities = load_known_entities()
    print(f"  Entries: {len(entities)}")

    # Document frequencies from the gensim dictionary — used to spare
    # high-DF tokens from the noise flag (real OCR junk is rare, not common).
    doc_freqs = {
        dictionary[token_id]: freq
        for token_id, freq in dictionary.dfs.items()
    }

    # ------------------------------------------------------------------
    # 2. Top-words CSV
    # ------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBuilding top-20 words per topic ...")
    df_words = build_top_words_csv(lda, k)
    top_words_path = REPORTS_DIR / f"topics_k{k}_top_words.csv"
    df_words.to_csv(top_words_path, index=False)
    print(f"  Saved → {top_words_path.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 3. Representative documents
    # ------------------------------------------------------------------
    print("\nBuilding representative documents per topic ...")
    df_reps = build_representative_docs(lda, corpus, df_docs, k)
    reps_path = REPORTS_DIR / f"topics_k{k}_representative_docs.csv"
    df_reps.to_csv(reps_path, index=False)
    print(f"  Saved → {reps_path.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 4. Noise check
    # ------------------------------------------------------------------
    print("\nBuilding noise check ...")
    df_noise = build_noise_check(lda, k, dict_words, cw_vocab, entities, doc_freqs)
    noise_path = REPORTS_DIR / f"topics_k{k}_noise_check.csv"
    df_noise.to_csv(noise_path, index=False)
    print(f"  Saved → {noise_path.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 5. pyLDAvis (best effort)
    # ------------------------------------------------------------------
    vis_path = REPORTS_DIR / f"topics_k{k}_pyldavis.html"
    print("\nBuilding pyLDAvis visualisation ...")
    if try_pyldavis(lda, corpus, dictionary, vis_path):
        print(f"  Saved → {vis_path.relative_to(LDA_DIR)}")

    # ------------------------------------------------------------------
    # 6. Per-topic stdout printout
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print(f"PER-TOPIC INSPECTION  (k = {k})")
    print(separator)

    flagged_by_topic = (
        df_noise[df_noise["flagged_as_noise"]]
        .groupby("topic_id")["word"].apply(list).to_dict()
    )

    for topic_id in range(k):
        top10 = [w for w in df_words.loc[topic_id, [f"word_{i}" for i in range(1, 11)]]]
        top10_str = ", ".join(top10)
        flagged = flagged_by_topic.get(topic_id, [])
        flag_str = f"  [noise: {', '.join(flagged)}]" if flagged else ""
        print(f"  topic {topic_id:>3}: {top10_str}{flag_str}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    total_topwords = k * TOP_N_WORDS
    total_flagged = int(df_noise["flagged_as_noise"].sum())
    clean_topics = [t for t in range(k) if t not in flagged_by_topic]
    noisy_topics = sorted(flagged_by_topic.keys())
    noise_rate = total_flagged / total_topwords if total_topwords else 0.0

    print(f"\n{separator}")
    print("SUMMARY")
    print(separator)
    print(f"  k                              : {k}")
    print(f"  Total topics                   : {k}")
    print(f"  Topics with zero noise flags   : {len(clean_topics)}")
    print(f"  Topics with 1+ noise flags     : {len(noisy_topics)}")
    if noisy_topics:
        print(f"    → topic_ids: {noisy_topics}")
    print(f"  Top-words inspected            : {total_topwords}")
    print(f"  Top-words flagged as noise     : {total_flagged}")
    print(f"  Overall noise rate             : {noise_rate:.2%}")
    print(separator)


if __name__ == "__main__":
    main()
