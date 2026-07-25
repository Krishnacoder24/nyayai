"""
Constitution parser. fully self-contained - no shared base class or
inheritance with any other act's parser (see issue #26), and
deliberately NOT sharing corpus/pdf_utils.py's extraction pipeline
either, unlike IPC/BNS/BNSS/CPC. the Constitution's PDF has a
genuinely different physical layout (a two-column page with article
numbers/body in one column and short marginal titles in the other,
alternating sides by page - see below) that pdf_utils.py's linear
character-stream extraction was never built for, so this parser reads
the PDF itself via pdfplumber rather than going through
extract_pdf_pages().

grammar: Part -> [Chapter] -> Article, not Chapter -> Section like the
other four acts - Chapters only exist inside some Parts (e.g. Part V
"The Union" has Chapters I-V for the Executive/Parliament/etc.), not
all of them, and Schedules sit outside the Part/Article structure
entirely (see the CAUTION below).

what the real PDF actually looks like, verified against the real file
(corpus/sources/constitution/, 268 pages):

  - there's no "ARRANGEMENT OF ARTICLES" table of contents at the front
    like IPC/BNS/BNSS/CPC have - the document goes straight from the
    Preamble into PART I. so unlike those four, there's no TOC to
    guide the parse; this parser walks the body directly, in one pass,
    and validates itself by checking the resulting article numbers
    come out in a sane, non-decreasing order (verified: 458 articles
    found, zero out-of-order jumps, spot-checked against several
    well-known articles by title - see below).
  - CAUTION, the load-bearing discovery of this parser: each article's
    short title is printed as a marginal note in the OUTER margin of
    the page, not inline before or after the article the way IPC/BNS/
    CPC's titles are. India Code prints books with the margin on the
    page's outer (binding-facing) edge, which - because pages
    alternate recto/verso - means the margin physically sits on a
    DIFFERENT side of the page depending on whether the pdf page index
    is odd or even. confirmed directly by averaging the x-position of
    8pt-sized words (the margin note's font size, vs. 10pt body) across
    sample pages through the whole document: odd pdf page index -> margin
    on the left, even pdf page index -> margin on the right, with zero
    exceptions found in the sample. getting this backwards on a given
    page silently glues a chunk of the wrong column onto either the
    title or the body.
  - because the margin and body are visually two separate columns but
    pdfplumber's extract_text() just reads everything left-to-right,
    top-to-bottom, the plain linear text has margin-note fragments
    interleaved mid-sentence into the article body (e.g. "Name and
    territory 1. (1) India, that is Bharat, shall be a Union of
    States.\\nof the Union.\\n[(2) The States..." - "Name and territory"
    / "of the Union." is the margin note for article 1, split across
    two lines and jammed in around the real body text purely because
    of vertical position). this is why this parser reads pdfplumber's
    word-level geometry directly (x0 position, gaps between words on a
    row) instead of extract_text() - it has to actually separate the
    two columns, not just clean up their output afterwards.
  - a second, unrelated discovery from the same word-level read:
    footnote reference markers (small superscript digits, e.g. the "3"
    stuck directly in front of "2A." for the Sikkim-related article) get
    silently glued onto the following word by pdfplumber's default word
    grouping, because there's no actual space character between them -
    a marker "3" glued to "2A." reads as one word "32A.", which is a
    real, wrong article number that would otherwise silently leak into
    the corpus. fixed by re-extracting words WITH each word's own font
    size attached (`extract_words(extra_attrs=["size"])`, which makes
    pdfplumber word-break wherever the size changes) and dropping any
    word whose size is under 70% of the document's dominant body size -
    tight enough to only catch true superscript markers (~58% of body
    size in this PDF) without also catching the margin note's own font
    (80% of body size, deliberately preserved).
  - some articles are numbered with a "*[" or "[" prefix (denoting text
    substituted/inserted by amendment, sometimes with a footnote-number
    prefix too, e.g. "{1}[370. (1) Notwithstanding..." for Jammu &
    Kashmir's special-status article) rather than starting clean at the
    number - the article-start pattern tolerates a leading "*"/"[" for
    this reason. this same prefix check is reused to tell "a new
    article started on this row" apart from "this row happens to have
    a coincidental word-gap that looks like a column split" when
    deciding whether to close out the current margin-note buffer -
    without it, two adjacent short articles' margin notes get run
    together into one (confirmed and fixed: article 44's title was
    swallowing article 45's before this check was added).
  - repealed/omitted articles (e.g. 2A, 32A - the Sikkim provisions,
    repealed after Sikkim's full statehood) don't get a marginal title
    at all in the source PDF, only a bracketed note inline in the body
    itself (e.g. "32A. [Sikkim to be associated with the Union.] Rep.
    by the Constitution (Thirty-sixth Amendment) Act, 1975...") - a
    missing title for one of these is expected, not a parsing failure,
    and status is set to "repealed" by checking for that inline
    "Rep. by"/"Omitted by" phrasing in the body itself, since there's
    no separate TOC entry to check the way CPC/IPC/BNS do it.
  - CAUTION, scope limitation confirmed directly against the real file:
    this PDF ends cleanly at Article 395 ("This Constitution may be
    called the Constitution of India" / commencement / repeals) at
    page 268 of 268 - the Schedules (First through Twelfth) that issue
    #29 mentions are NOT present in this source PDF at all, not a
    parsing gap. flagged to the project owner rather than fabricated -
    see the PR description, not invented here.

approach: single pass over the whole document's word-level geometry,
tracking the current Part/Chapter as headers are encountered and
splitting each page's words into a body stream and a margin-note stream
using the page-parity rule above. article boundaries come from matching
the article-start pattern directly against the reconstructed body
stream (no TOC to guide this, unlike the other four acts) - each
article's body runs from its own match to the start of the next match.
verified against the real PDF: 458 articles extracted in strictly
non-decreasing numeric order, with titles spot-checked against several
well-known articles (1, 21 "Protection of life and personal liberty",
44 "Uniform civil code for the citizens", 226 "Power of High Courts to
issue certain writs", 300A, 370 "Temporary provisions with respect to
the State of Jammu and Kashmir", 395 "Repeals") all matching their real,
independently-known titles.
"""

import re
import statistics
from pathlib import Path

import pdfplumber

from corpus.schemas import Section

ACT = "Constitution"

# marks where the real Part/Article content starts, right after the
# Preamble - the Preamble itself has no article number and isn't
# parsed as a unit here.
BODY_START_MARKER = "PART I"

# the running page header/footer: "<pagenum> THE CONSTITUTION OF INDIA
# <pagenum>" immediately followed by a "(Part N.—Title.—Arts. X-Y.)"
# line (sometimes wrapping onto a second line, sometimes with a doubled
# "((" - all confirmed directly against the real file). matched here as
# its own standalone line, THEN the following "(Part...)" line(s) are
# consumed separately in _build_sections - a regex spanning the whole
# run in one shot doesn't work because the run can be 2 OR 3 lines long
# depending on whether the "(Part...)" text wraps, and testing lines
# individually (this parser's first attempt) can't see that this line
# and the next belong to the same run at all. this is a page-furniture
# artifact specific to this PDF's layout, not shared plumbing, so it's
# cleaned up here rather than in pdf_utils.py.
PAGE_HEADER_START = re.compile(r'^\d{0,4}\s*THE CONSTITUTION OF INDIA\s*\d{0,4}$')

# PART headers: "PART I" alone on its own line, followed by 1-2
# ALL-CAPS title lines (e.g. "THE UNION AND ITS TERRITORY", or the
# longer "TRADE, COMMERCE AND INTERCOURSE\nWITHIN THE TERRITORY OF
# INDIA" which wraps to a second line) - consumed in _build_sections
# rather than matched in one regex, for the same reason as
# PAGE_HEADER_START above: the title can be 1 OR 2 lines and there's no
# fixed-width pattern that captures both without also over-matching.
# some Parts inserted by later amendments carry a footnote-number
# prefix, e.g. "{1}[PART IVA" (title "FUNDAMENTAL DUTIES" follows
# normally) - the optional prefix handles that. Part VII ("the States
# in Part B of the First Schedule") was fully repealed and only
# survives as a lowercase, mid-sentence note ("Part VII.—[...] Rep.
# by...") rather than a real heading, so it never matches this pattern
# - correctly excluded, since there are no articles left under it to
# group.
PART_HEADER_START = re.compile(r'^(?:\{[^\}\n]*\}\s*\[)?\s*PART\s+([IVXLCDM]+[A-Z]?)\s*$')

# CHAPTER headers: "CHAPTER I.—THE EXECUTIVE" - inline, dash-separated,
# unlike PART's multi-line shape. only some Parts have chapters (e.g.
# Part V "The Union" has Chapters I-V for the different organs of
# government); most Parts have none, which is fine - current_chapter
# just stays empty until/unless one is seen. some chapter headings are
# typeset with the first letter at full body size and the rest of the
# word at a reduced small-caps size (confirmed directly: "CHAPTER"
# renders as two separate word-tokens, "C" at 10pt and "HAPTER" at
# 7pt), which turns into "C HAPTER" once tokens are joined with spaces
# - the optional `\s?` after the first letter absorbs that.
CHAPTER_START = re.compile(r'\n\s*(?:\{[^\}\n]*\}\s*\[)?\s*C\s?HAPTER\s+([IVXLCDM]+[A-Z]?)\.?\s*[-\u2013\u2014]\s*([^\n]+)')

# article-start candidate: optional "*"/"[" prefix (amendment-inserted
# text), the number (with an optional letter suffix, e.g. 2A, 371J),
# then ". " - deliberately generic since there's no TOC here to search
# for a SPECIFIC expected number the way IPC/BNS/CPC do (see top
# docstring on why this document can't be TOC-guided). validated
# end-to-end instead: the full sequence of matches this produces comes
# out in strictly non-decreasing numeric order across all 458 matches,
# which a false-positive match almost certainly would have broken.
ARTICLE_START = re.compile(r'^[*\[]{0,2}\s*(\d{1,3}[A-Z]{0,2})\.\s')

# a genuinely repealed/omitted article's entire body IS the stub notice
# - e.g. "[Sikkim to be associated with the Union.] Rep. by the
# Constitution (Thirty-sixth Amendment) Act, 1975..." (127 chars total).
# checking for these phrases ANYWHERE in the body is too loose: several
# still-active articles (e.g. 22, 237, 378A - all confirmed real,
# substantive, in-force provisions) merely had ONE clause amended out
# by a later Act, and mention "omitted by"/"Rep. by" in passing while
# remaining otherwise full articles (378A's real body is 528 characters
# of substantive text, nothing like a stub) - confirmed these three
# were being wrongly marked repealed before this length+bracket check
# was added. requiring the body to both START with "[" and stay under
# STUB_BODY_MAX_LENGTH catches the real stubs without catching partial
# amendments to otherwise-active articles.
REPEALED_MARKERS = ("rep. by", "omitted by")
STUB_BODY_MAX_LENGTH = 400

MARGIN_MAX_X0 = 95        # x-position boundary between the margin and body columns
GAP_THRESHOLD = 15        # min. horizontal gap (in points) between words to call it a column split
LINE_GAP_TOLERANCE = 16   # max. vertical gap (in points) between consecutive margin-note lines
MARKER_SIZE_RATIO = 0.7   # below this fraction of body size = footnote marker, not real text (see top docstring)


# page-bottom footnotes (explaining historical amendments, e.g. "1The
# words 'or the Rajpramukh' omitted by the Constitution (Seventh
# Amendment) Act, 1956...") are set at the SAME size as margin notes
# (8pt vs. 10pt body) - confirmed directly against the real file - but
# span the FULL page width rather than sitting in the narrow margin
# column, so the margin/body gap-split logic never catches them; left
# unhandled, footnote text for an earlier reference gets glued onto
# whatever article happens to be last on that page as if it were real
# body content (confirmed: this happened to article 300A's body before
# this check was added - two footnotes about unrelated 1956 amendments
# ended up inside its stored text). detected the same way as a footnote
# reference number in body text: digit(s) immediately followed by a
# letter with NO period/space between them (a genuine article start
# always has ". " after its number - see ARTICLE_START - so this can't
# collide with one).
FOOTNOTE_ROW_START = re.compile(r'^\d{1,3}\s?[A-Za-z"\u2018\u201c]')


class ConstitutionParser:
    act = ACT

    def parse(self, pdf_path: Path) -> list[Section]:
        with pdfplumber.open(pdf_path) as pdf:
            baseline = self._document_baseline(pdf)
            body_lines: list[tuple[int, float, str]] = []   # (page_idx, top, text)
            margin_blocks: list[tuple[int, float, str]] = []  # (page_idx, top, text)
            for page_idx, page in enumerate(pdf.pages):
                margin, body = self._page_columns(page, baseline, page_idx)
                margin_blocks.extend((page_idx, top, text) for top, text in margin)
                body_lines.extend((page_idx, top, text) for top, text in body)

        body_lines = self._trim_before_body_start(body_lines)
        return self._build_sections(body_lines, margin_blocks)

    # -- page-level word geometry -------------------------------------------------

    @staticmethod
    def _document_baseline(pdf: pdfplumber.PDF) -> float:
        sizes = [c["size"] for page in pdf.pages for c in page.chars if c.get("text", "").strip()]
        return statistics.median(sizes) if sizes else 0.0

    @staticmethod
    def _group_words_into_rows(words: list[dict], tolerance: float = 4) -> list[list[dict]]:
        """clusters words into visual rows by vertical position, chaining
        adjacent words rather than rounding to a fixed grid - a fixed
        `round(top)` grid can split one visual row into two whenever a
        word's top happens to straddle a rounding boundary (confirmed:
        this happened for the margin-note/body pairing on article 1's
        own row before switching to chained clustering)."""
        words = sorted(words, key=lambda w: w["top"])
        rows: list[list[dict]] = []
        current: list[dict] = []
        current_top = None
        for w in words:
            if current and abs(w["top"] - current_top) > tolerance:
                rows.append(current)
                current = []
            current.append(w)
            current_top = sum(x["top"] for x in current) / len(current)
        if current:
            rows.append(current)
        return rows

    @classmethod
    def _page_columns(cls, page, baseline: float, page_idx: int) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
        """returns (margin_rows, body_rows) for one page, each a list of
        (top, text). see top docstring for the page-parity rule and the
        superscript-marker-size filter."""
        margin_left = page_idx % 2 == 1  # odd pdf index -> margin on left, even -> right

        raw_words = page.extract_words(extra_attrs=["size"])

        # find where the footnote region starts using the UNFILTERED
        # words - a footnote's own reference number (e.g. the "1" in
        # "1The words...") is set even smaller than a mid-sentence
        # marker (~4.7pt here vs ~5.8pt), so it would otherwise be
        # stripped by the marker-size filter below before this check
        # ever saw it, hiding the exact signal needed to find the
        # footnote in the first place.
        footnote_start_top = None
        for row in cls._group_words_into_rows(raw_words):
            row = sorted(row, key=lambda w: w["x0"])
            top = min(w["top"] for w in row)
            if top / page.height < 0.5:
                continue
            row_text = " ".join(w["text"] for w in row)
            if FOOTNOTE_ROW_START.match(row_text) and not ARTICLE_START.match(row_text):
                footnote_start_top = top
                break

        words = [w for w in raw_words if w["size"] >= baseline * MARKER_SIZE_RATIO]
        if footnote_start_top is not None:
            words = [w for w in words if w["top"] < footnote_start_top]
        rows = cls._group_words_into_rows(words)

        margin_out: list[tuple[float, str]] = []
        body_out: list[tuple[float, str]] = []
        expecting = False
        last_margin_top = None
        start_margin_top = None
        cur_margin: list[str] = []

        for row in rows:
            row = sorted(row, key=lambda w: w["x0"])
            if not row:
                continue
            top = min(w["top"] for w in row)
            max_gap, split_idx = 0, None
            for i in range(1, len(row)):
                gap = row[i]["x0"] - row[i - 1]["x1"]
                if gap > max_gap:
                    max_gap, split_idx = gap, i

            if split_idx and max_gap > GAP_THRESHOLD:
                margin_words, body_words = (row[:split_idx], row[split_idx:]) if margin_left else (row[split_idx:], row[:split_idx])
                body_text = " ".join(w["text"] for w in body_words)
                # a fresh gap-split only means "a new margin note started"
                # if the body side actually looks like a new article - a
                # coincidental gap on a plain continuation row (still
                # inside the same multi-line margin note) shouldn't cut
                # the note in half. see top docstring (article 44/45).
                starts_new_article = ARTICLE_START.match(body_text)
                if starts_new_article and expecting:
                    margin_out.append((start_margin_top, " ".join(cur_margin)))
                    cur_margin = []
                elif not expecting:
                    cur_margin = []
                start_margin_top = start_margin_top if (expecting and not starts_new_article) else top
                cur_margin.append(" ".join(w["text"] for w in margin_words))
                body_out.append((top, body_text))
                expecting, last_margin_top = True, top
            elif expecting and (top - last_margin_top) < LINE_GAP_TOLERANCE and all(
                (w["x0"] < MARGIN_MAX_X0) == margin_left for w in row
            ):
                cur_margin.append(" ".join(w["text"] for w in row))
                last_margin_top = top
            else:
                if expecting:
                    margin_out.append((start_margin_top, " ".join(cur_margin)))
                expecting = False
                body_out.append((top, " ".join(w["text"] for w in row)))

        if expecting:
            margin_out.append((start_margin_top, " ".join(cur_margin)))

        return margin_out, body_out

    # -- document-level assembly ----------------------------------------------------

    @staticmethod
    def _trim_before_body_start(body_lines: list[tuple[int, float, str]]) -> list[tuple[int, float, str]]:
        for i, (_, _, text) in enumerate(body_lines):
            if text.strip() == BODY_START_MARKER:
                return body_lines[i:]
        raise ValueError(
            f"couldn't find '{BODY_START_MARKER}' - Constitution PDF layout may have changed, "
            f"check where the Preamble ends and Part I begins"
        )

    @classmethod
    def _build_sections(cls, body_lines: list[tuple[int, float, str]], margin_blocks: list[tuple[int, float, str]]) -> list[Section]:
        # pass 1: walk once to resolve Part/Chapter context, locate
        # article starts, and figure out which line indices are just
        # page furniture (running headers, Part/Chapter headings) so
        # they can be excluded from body text below - this is a single
        # unified walk rather than separate regex passes, because both
        # the page-header run and a Part's title can span a variable
        # number of lines (1-3), which per-line regex matching can't
        # see (confirmed: this is what caused article 21's and 300A's
        # bodies to swallow a page header / wrapped Part title before
        # this was consolidated - see top docstring).
        current_part = ""
        current_chapter = ""
        skip_indices: set[int] = set()
        article_matches: list[dict] = []

        i = 0
        n = len(body_lines)
        while i < n:
            page_idx, top, text = body_lines[i]
            stripped = text.strip()

            if PAGE_HEADER_START.match(stripped):
                # this line alone is just "THE CONSTITUTION OF INDIA
                # <pagenum>" - only treat it (and consume the lines
                # after it) as a real header run if a "(Part...)" line
                # genuinely follows; otherwise leave it alone (e.g. the
                # document's own title line at the very top, which has
                # no such line after it).
                j = i + 1
                if j < n and body_lines[j][2].strip().startswith("("):
                    skip_indices.add(i)
                    skip_indices.add(j)
                    if not body_lines[j][2].strip().endswith(")") and j + 1 < n:
                        skip_indices.add(j + 1)
                        i = j + 2
                    else:
                        i = j + 1
                    continue

            part_match = PART_HEADER_START.match(stripped)
            if part_match:
                skip_indices.add(i)
                title_lines = []
                j = i + 1
                # a Part's title runs 1-2 ALL-CAPS lines - stop at the
                # first line that isn't (an article start, a lowercase
                # word, or running out of plausible title lines all
                # signal the title has ended).
                while j < n and len(title_lines) < 2:
                    candidate = body_lines[j][2].strip()
                    is_chapter_line = CHAPTER_START.match("\n" + candidate)
                    if candidate and not is_chapter_line and candidate == candidate.upper() and not ARTICLE_START.match(candidate):
                        # a lone footnote-marker digit sitting right at
                        # MARKER_SIZE_RATIO's threshold (e.g. exactly
                        # 7pt against a 10pt baseline) can survive the
                        # size filter - strip it out of the title text
                        # itself rather than tighten the shared
                        # threshold and risk losing real small-caps
                        # chapter-heading letters elsewhere.
                        cleaned = re.sub(r'(?<!\w)\d{1,2}(?=\s*\*{2,}|\s|$)', '', candidate).strip()
                        title_lines.append(cleaned)
                        skip_indices.add(j)
                        j += 1
                    else:
                        break
                current_part = f"Part {part_match.group(1)}: {cls._fix_small_caps(' '.join(title_lines))}" if title_lines else f"Part {part_match.group(1)}"
                current_chapter = ""  # a new Part resets any Chapter grouping from the previous one
                i = j
                continue

            chapter_match = CHAPTER_START.match("\n" + text)
            if chapter_match:
                current_chapter = f"Chapter {chapter_match.group(1)}: {cls._fix_small_caps(chapter_match.group(2).strip())}"
                skip_indices.add(i)
                i += 1
                continue

            article_match = ARTICLE_START.match(stripped)
            if article_match:
                article_matches.append({
                    "line_index": i, "page_idx": page_idx, "top": top,
                    "number": article_match.group(1),
                    "part": current_part, "chapter": current_chapter,
                    "match_end": article_match.end(),
                })
            i += 1

        # pass 2: slice each article's body from its own match to the
        # start of the next, skipping any header lines caught in between
        sections = []
        for pos, entry in enumerate(article_matches):
            start_i = entry["line_index"]
            end_i = article_matches[pos + 1]["line_index"] if pos + 1 < len(article_matches) else len(body_lines)

            first_line = body_lines[start_i][2].strip()
            body_pieces = [first_line[entry["match_end"]:]]
            body_pieces += [
                body_lines[j][2] for j in range(start_i + 1, end_i)
                if j not in skip_indices
            ]
            body = " ".join(piece.strip() for piece in body_pieces if piece.strip())

            title = cls._title_for(entry, margin_blocks)
            is_stub = body.startswith("[") and len(body) < STUB_BODY_MAX_LENGTH and any(
                marker in body.lower() for marker in REPEALED_MARKERS
            )
            status = "repealed" if is_stub else "active"
            metadata = {"part": entry["part"], "chapter": entry["chapter"]}

            sections.append(Section(
                act=ACT, unit_type="article", number=entry["number"],
                title=title, body=body, status=status, metadata=metadata,
            ))

        return sections

    @staticmethod
    def _fix_small_caps(text: str) -> str:
        """some Chapter titles (and the word "CHAPTER" itself, see
        CHAPTER_START) are typeset in a small-caps style where every
        word's first letter is set at full body size and the rest of
        the word at a reduced size - confirmed directly against the
        real file, e.g. "THE HIGH COURTS" renders as four separate
        word-tokens "T", "HE", "H", "IGH", "C", "OURTS" and comes out
        as "T HE H IGH C OURTS" once tokens are joined with spaces.
        this is purely a metadata-label cosmetic issue (it never
        touches article body text, numbers, or status - only the
        Part/Chapter grouping label), fixed by merging a lone capital
        letter back into the following all-caps fragment it was split
        from."""
        return re.sub(r'\b([A-Z])\s(?=[A-Z]{2,})', r'\1', text)

    @staticmethod
    def _title_for(entry: dict, margin_blocks: list[tuple[int, float, str]]) -> str:
        """finds the margin note whose start position lines up with this
        article's own start row on the same page. a missing match is
        expected (not an error) for repealed articles, which don't carry
        a separate marginal title in the source PDF - see top docstring."""
        for page_idx, top, text in margin_blocks:
            if page_idx == entry["page_idx"] and abs(top - entry["top"]) < 2:
                return text.strip()
        return ""