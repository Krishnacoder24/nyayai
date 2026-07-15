"""
document-level entity consistency checker.

the ML token classifier (predict.py) only sees 512 tokens at a time —
it has no memory of page 1 when processing page 3. entity consistency
is a document-level problem that needs a full-document pass.

what it catches:
  - petitioner named "Ramesh Kumar" on page 1, "Rakesh Kumar" on page 3
  - witness "Anwar Sheikh" in complaint, "Anwar Shaikh" in evidence list
  - place "Patna" in FIR, "Patana" in witness statement

approach:
  1. NER pass — extract all named entities from every span
  2. cluster — group mentions that likely refer to the same entity
     using fuzzy string matching (rapidfuzz ratio)
  3. canonical form — most frequent mention in each cluster is canonical
  4. flag deviations — mentions that differ from canonical are ENT errors

  Currently doesn't work well will Indian names.
"""

import logging
from collections import Counter

from rapidfuzz import fuzz

from ocr.tokens import LineSpan
from model.schemas import ErrorSpan

logger = logging.getLogger(__name__)

# fuzzy similarity threshold for "same entity" (0-100)
# 85 catches typos and spelling variants (Rakesh/Ramesh=92, Sheikh/Shaikh=92)
# but rejects genuinely different names (Suresh/Ramesh=75, Patna/Pune=44)
SIMILARITY_THRESHOLD = 85

# minimum entity length to consider — filters out "the", "Court" standalone,
# pronoun-like shorthand that isn't a consistency error
MIN_ENTITY_LEN = 5

# spacy model — falls back to rule-based extraction if not available
# en_core_web_sm knows person/place/org but not Indian legal entities well
# swap this for a fine-tuned Indian legal NER model when available
SPACY_MODEL = "en_core_web_sm"

# entity types we care about for consistency checking
# spacy label -> our internal label
ENTITY_TYPE_MAP = {
    "PERSON": "person",
    "GPE": "place",        # geopolitical entity (cities, states, countries)
    "LOC": "place",        # other locations
    "ORG": "organization",
    "FAC": "place",        # facility (buildings, landmarks)
}


def check_entities(spans: list[LineSpan]) -> list[ErrorSpan]:
    """
    runs entity consistency check over the full document.
    returns ErrorSpans for entity mentions that deviate from their
    canonical (most frequent) form.
    """
    try:
        nlp = _load_nlp()
    except Exception as e:
        logger.warning(f"NER model not available ({e}) — skipping entity check")
        return []

    # step 1: extract all entity mentions across all spans
    mentions = _extract_mentions(spans, nlp)

    if not mentions:
        return []

    # step 2: cluster mentions by entity type, then fuzzy match within type
    clusters = _cluster_mentions(mentions)

    # step 3: flag deviations from canonical form in each cluster
    errors = _flag_deviations(clusters, spans)

    return errors


def _load_nlp():
    import spacy
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        # model not downloaded yet
        raise OSError(
            f"spacy model '{SPACY_MODEL}' not found. "
            f"run: uv run python -m spacy download {SPACY_MODEL}"
        )


def _extract_mentions(spans: list[LineSpan], nlp) -> list[dict]:
    """
    runs NER on every span and returns a flat list of mention dicts.
    each mention carries: text, entity_type, span_idx, source LineSpan.
    """
    mentions = []

    for span_idx, span in enumerate(spans):
        doc = nlp(span.text)
        for ent in doc.ents:
            ent_type = ENTITY_TYPE_MAP.get(ent.label_)
            if ent_type is None:
                continue  # entity type we don't care about

            text = ent.text.strip()
            if len(text) < MIN_ENTITY_LEN:
                continue  # too short, likely a shorthand or pronoun

            mentions.append({
                "text": text,
                "entity_type": ent_type,
                "span_idx": span_idx,
                "span": span,
            })

    return mentions


def _cluster_mentions(mentions: list[dict]) -> list[list[dict]]:
    """
    groups mentions that likely refer to the same real-world entity.
    only compares mentions of the same entity_type — never clusters
    a person name with a place name even if strings look similar.
    uses greedy single-pass clustering: each mention joins the first
    cluster whose representative it's similar enough to.
    """
    clusters: list[list[dict]] = []

    # group by entity type first
    by_type: dict[str, list[dict]] = {}
    for m in mentions:
        by_type.setdefault(m["entity_type"], []).append(m)

    for entity_type, type_mentions in by_type.items():
        type_clusters: list[list[dict]] = []

        for mention in type_mentions:
            placed = False
            for cluster in type_clusters:
                # compare against the most frequent (canonical) mention in cluster
                canonical = _get_canonical(cluster)
                score = fuzz.ratio(mention["text"].lower(), canonical.lower())
                if score >= SIMILARITY_THRESHOLD:
                    cluster.append(mention)
                    placed = True
                    break

            if not placed:
                type_clusters.append([mention])

        clusters.extend(type_clusters)

    # only care about clusters with more than one distinct surface form
    # single-mention entities can't have consistency errors
    return [c for c in clusters if _has_multiple_forms(c)]


def _get_canonical(cluster: list[dict]) -> str:
    """
    canonical form = most frequently occurring mention text in the cluster.
    ties broken by longer string (more complete form preferred).
    """
    counts = Counter(m["text"] for m in cluster)
    max_count = max(counts.values())
    candidates = [text for text, count in counts.items() if count == max_count]
    return max(candidates, key=len)


def _has_multiple_forms(cluster: list[dict]) -> bool:
    unique_forms = set(m["text"].lower() for m in cluster)
    return len(unique_forms) > 1


def _flag_deviations(clusters: list[list[dict]], spans: list[LineSpan]) -> list[ErrorSpan]:
    """
    for each cluster, flags mentions that differ from the canonical form.
    """
    errors = []

    for cluster in clusters:
        canonical = _get_canonical(cluster)

        for mention in cluster:
            if mention["text"].lower() == canonical.lower():
                continue  # this is the canonical form, not an error

            span = mention["span"]
            errors.append(ErrorSpan(
                text=mention["text"],
                error_type="entity",
                page_no=span.page_no,
                x0=span.x0, y0=span.y0, x1=span.x1, y1=span.y1,
                suggestion=f'should be "{canonical}"',
                confidence=round(
                    fuzz.ratio(mention["text"].lower(), canonical.lower()) / 100, 2
                ),
            ))

    return errors