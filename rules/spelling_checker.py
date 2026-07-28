"""
rule-based spelling checker, complementing the ML token classifier.

per docs/roadmap.md's known limitations: using a full ML token classifier
for simple spelling typos is overkill, and until issue #12's fine-tuned
model exists there's no spelling detection running at all. this gives
useful spelling detection independent of fine-tuning - a generic English
dictionary (pyspellchecker) plus a custom word list of common Indian-legal
vocabulary layered on top, so genuine typos still get flagged without
every "cognizable", "vakalatnama", "panchnama" etc. becoming a false
positive.

grammar detection is intentionally NOT attempted here - issue #41 scopes
this file to spelling only. a rule-based grammar engine is a separate,
bigger effort (see docs/roadmap.md's "more rule checkers" section).

IMPORTANT - British vs. American spelling: pyspellchecker's bundled "en"
dictionary is American-frequency-based. Indian legal English is written
in British spelling throughout (the Constitution, IPC/BNS, every court
judgment) - "offence", "defence", "authorised", "organisation" are not
typos here, they're the correct and near-universal spelling. Flagging
every one of those would swamp real typos in noise and make the checker
useless in practice. _is_british_spelling() below recognises the regular
British/American suffix pairs (-our/-or, -ise/-ize, -re/-er, -nce/-nse,
-mme/-m, -ogue/-og) by deriving the American form and checking it against
the dictionary; BRITISH_VOCAB covers the irregular pairs that don't fit
a suffix rule (judgement, cheque, tyre, ...).

NOTE on scope: this only checks words the dictionary has genuinely never
seen and can still suggest a fix for. two categories are deliberately
skipped, not because they can't be wrong, but because a cheap rule can't
tell wrong from right for them without doing another module's job over
again:
  - ALL-CAPS words ("IPC", "FIR", "PS") - acronyms, not English words.
  - Capitalized-mid-word tokens ("Ramesh", "Patna") - almost always
    proper nouns. entity_checker.py already owns cross-document name/place
    consistency; a spelling rule re-flagging "Rakesh" as "did you mean
    Rakesh" would just be noisy duplication of that check, and a plain
    dictionary has no way to tell a typo'd name from a real uncommon one
    anyway. trade-off: a typo that happens to be the very first word of a
    sentence (and therefore capitalized) is missed too. acceptable, since
    the alternative is flagging most proper nouns in every FIR.
"""

import re
import logging

from spellchecker import SpellChecker

from ocr.tokens import LineSpan
from model.schemas import ErrorSpan

logger = logging.getLogger(__name__)

# pyspellchecker's default max edit distance (2) is fine for short-word
# typos in filings - no reason to widen it and slow every document down
EDIT_DISTANCE = 2

# words shorter than this are basically never a genuine, actionable
# spelling error - "u/s" fragments, single initials, "a", "of"
MIN_WORD_LEN = 3

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# common Indian-legal vocabulary that a generic English dictionary either
# doesn't have at all (vernacular loanwords like "panchnama", "khasra") or
# would otherwise flag as unusual. confirmed missing from pyspellchecker's
# default word list by direct check before writing this list.
# grouped by rough category purely for human readability; the checker
# only ever sees the flat set below.
LEGAL_VOCAB = {
    # Latin legal maxims and their component words
    "malafide", "mala", "bonafide", "bona", "fide",
    "prima", "facie", "suo", "moto", "motu", "propio",
    "habeas", "corpus", "mens", "rea", "actus", "reus",
    "vires", "ultra", "decidendi", "ratio", "obiter", "dicta",
    "sine", "qua", "non", "ipso", "facto", "inter", "alia",
    "vice", "versa", "amicus", "curiae",

    # Indian criminal-procedure / FIR vocabulary
    "vakalatnama", "wakalatnama", "panchnama", "mahazar", "zimni",
    "chargesheet", "chargesheets", "chargesheeted",
    "absconding", "absconded", "abscond",
    "undertrial", "undertrials", "cognizance", "cognizable",
    "noncognizable", "bailable", "nonbailable",
    "remand", "remanded", "custodial", "custody",
    "deponent", "deponents", "complainant", "complainants",
    "accused", "acquittal", "acquitted", "conviction", "convicted",
    "abetment", "abet", "abetted", "abettor",
    "dacoity", "dacoits", "extortion", "trespass", "trespasser",
    "nuisance", "defamatory", "slander", "libel", "perjury",
    "forgery", "counterfeit", "embezzlement", "misappropriation",

    # courts, revenue and land-record officialdom
    "panchayat", "panchayats", "sarpanch", "patwari", "tehsil",
    "tehsildar", "kotwali", "chowki", "munsif", "sessions",
    "magisterial", "khasra", "girdawari", "jamabandi", "patta",
    "nazul", "benami", "karta", "coparcener", "coparcenary",
    "mutation", "easementary", "mesne",

    # pleadings / civil-procedure vocabulary
    "plaintiff", "plaintiffs", "defendant", "defendants",
    "petitioner", "petitioners", "respondent", "respondents",
    "appellant", "appellants", "appellee", "litigant", "litigants",
    "adjudication", "adjudicate", "adjudicated", "tribunal", "tribunals",
    "jurisprudence", "decree", "decreed", "injunction", "injunctions",
    "subpoena", "subpoenaed", "summons", "summoned", "indictment",
    "recognizance", "surety", "sureties", "indemnify", "indemnity",
    "tortious", "culpable", "culpability", "exculpatory", "inculpatory",
    "corroborate", "corroborative", "corroboration", "hearsay",
    "admissible", "inadmissible", "adducing", "adduce", "adduced",
    "alibi", "vakil", "arzi", "hundi", "wasiyatnama", "ikrarnama",
}

# suffix pairs that turn a British spelling into the American spelling
# pyspellchecker's dictionary actually knows. applied in order; first
# rule whose transformed word is a known dictionary word wins.
# each pattern is anchored to the end of the word, with common
# inflections (plural/past-tense/adverb endings) folded in as an
# optional group so "favourable"/"labours"/"organisations" etc. match
# too, not just the bare root.
_BRITISH_SUFFIX_RULES = [
    (re.compile(r"our(s|ed|ing|able|ably)?$"), r"or\1"),   # colour -> color
    (re.compile(r"isation(s)?$"), r"ization\1"),            # organisation -> organization
    (re.compile(r"ising$"), "izing"),                        # organising -> organizing
    (re.compile(r"ised$"), "ized"),                           # authorised -> authorized
    (re.compile(r"ises$"), "izes"),                           # realises -> realizes
    (re.compile(r"ise(d|s|r|rs)?$"), r"ize\1"),              # realise -> realize
    (re.compile(r"ysing$"), "yzing"),                         # analysing -> analyzing
    (re.compile(r"yse(d|s)?$"), r"yze\1"),                   # analyse -> analyze
    (re.compile(r"([bcdfgklmnprstvz])re(s)?$"), r"\1er\2"),  # centre -> center
    (re.compile(r"nce(s)?$"), r"nse\1"),                     # offence -> offense
    (re.compile(r"mme(s)?$"), r"m\1"),                       # programme -> program
    (re.compile(r"ogue(s)?$"), r"og\1"),                      # catalogue -> catalog
]

# irregular British spellings that don't fit a suffix rule above - each
# is a genuine, correctly-spelled word, just not the American form
# pyspellchecker's dictionary expects.
BRITISH_VOCAB = {
    "judgement", "judgements", "cheque", "cheques", "tyre", "tyres",
    "kerb", "kerbs", "mould", "moulded", "moulding", "manoeuvre",
    "manoeuvres", "manoeuvring", "connexion", "storey", "storeys",
    "enrolment", "enrolments", "fulfil", "fulfilment", "fulfilled",
    "skilful", "skilfully", "wilful", "wilfully", "instalment",
    "instalments", "aeroplane", "aeroplanes", "artefact", "artefacts",
    "draught", "plough", "ploughed", "grey", "gaol", "gaoled",
}

_checker: SpellChecker | None = None


def check_spelling(spans: list[LineSpan]) -> list[ErrorSpan]:
    """
    runs a plain-English dictionary spell check (extended with
    LEGAL_VOCAB and British-spelling awareness) over every span, returns
    ErrorSpans for words the dictionary doesn't know and can still offer
    a correction for.

    pure Python, no model weights, no network calls - runs on every
    document regardless of fine-tuning status or Qdrant availability.
    """
    checker = _get_checker()

    errors = []
    for span in spans:
        errors.extend(_check_span(span, checker))
    return errors


def _get_checker() -> SpellChecker:
    """
    lazy singleton, same reasoning as model/predict.py's model cache
    (issue #35) - building the base word-frequency dictionary is the
    expensive part, no reason to redo it per document.
    """
    global _checker
    if _checker is None:
        _checker = SpellChecker(distance=EDIT_DISTANCE)
        _checker.word_frequency.load_words(w.lower() for w in LEGAL_VOCAB)
        _checker.word_frequency.load_words(w.lower() for w in BRITISH_VOCAB)
    return _checker


def _check_span(span: LineSpan, checker: SpellChecker) -> list[ErrorSpan]:
    errors = []

    for match in _WORD_RE.finditer(span.text):
        word = match.group(0)

        if len(word) < MIN_WORD_LEN or _is_skippable(word):
            continue

        lower = word.lower()
        if lower in checker or _is_british_spelling(lower, checker):
            continue

        correction = checker.correction(lower)
        if correction is None or correction == lower:
            # dictionary doesn't recognise it but also can't suggest a fix
            # (edit distance too large, or it's not a word at all) -
            # a flag with no actionable suggestion isn't worth surfacing
            continue

        errors.append(ErrorSpan(
            text=word,
            error_type="spelling",
            page_no=span.page_no,
            x0=span.x0, y0=span.y0, x1=span.x1, y1=span.y1,
            suggestion=f'did you mean "{correction}"?',
            confidence=0.7,
            source="spelling_rule",
        ))

    return errors


def _is_skippable(word: str) -> bool:
    """see this module's docstring for why these two categories exist."""
    if word.isupper():
        return True  # acronym, e.g. "IPC", "FIR", "PS"
    if word[0].isupper():
        return True  # capitalized mid-word - almost always a proper noun
    return False


def _is_british_spelling(lower_word: str, checker: SpellChecker) -> bool:
    """
    True if lower_word is a regular British spelling of a word the
    dictionary already knows under its American spelling. see this
    module's docstring for why this exists.
    """
    for pattern, replacement in _BRITISH_SUFFIX_RULES:
        candidate = pattern.sub(replacement, lower_word)
        if candidate != lower_word and candidate in checker:
            return True
    return False