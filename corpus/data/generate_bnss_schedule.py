"""
Extracts the "First Schedule / Classification of Offences" table from the
CrPC and BNSS PDFs into the same {section_number: [{"cognizable", "bailable",
"details"}]} shape corpus/search.py already expects (see SCHEDULE_FILES).

Why this needs custom code (issue #71): the table has no ruling lines in the
PDF, so pdfplumber's extract_tables() (which looks for ruled boundaries)
returns nothing. Instead we bucket every word into one of 6 columns by
x-position, using boundaries recovered empirically from where body text
actually starts (the printed "1 2 3 4 5 6" header's own x-positions turned
out NOT to line up with real column starts -- see get_column_edges' docstring
for how these were derived).

Two more layout quirks had to be handled specifically:
  1. CrPC prints the section-number label on the SAME physical line as the
     first row of content ("117 Abetting the commission...").
  2. BNSS prints it on its OWN, otherwise-blank physical line, positioned
     one line AFTER the row's true first content line. A 1-line lookahead
     buffer is used to detect this and re-attach the held-back line to the
     label that follows it.

Sub-row segmentation (the "if committed" / "if not committed" style splits
under many sections) is detected via a column-3 (punishment) gap: a genuine
new sub-row restates the punishment from scratch, which shows up as column 3
going empty for at least one line and then becoming non-empty again. All
sub-rows under one section are merged into a single "details" string
(matching the one-entry-per-section convention already used by
crpc_schedule.json). cognizable/bailable are resolved per sub-row from
column 4/5 text, "Ditto" is chained against the previous section's already
resolved value, and disagreement across sub-rows becomes
"Multiple (see details)".

This is heuristic, OCR-adjacent table extraction from a PDF with no ruling
lines -- not a verified government source. Treat it the way the existing
mapping tables are treated: useful, spot-checked, but not something to cite
as authoritative without cross-checking the primary text for anything
user-facing. (For calibration: run this file directly to see it checked
against the existing crpc_schedule.json -- ~75% of sections match exactly,
and nearly all of the remainder are one classification word off on
already-ambiguous "Ditto" chains, or -- for a handful of keys like "1", "10"
-- cases where the *existing* file itself mis-parsed a footnote marker as a
section number, which this version deliberately refuses to do.)
"""
import re
import json
from collections import defaultdict

import pdfplumber

HEADER_NUMS = {'1', '2', '3', '4', '5', '6'}
LABEL_RE = re.compile(r'^\d+[A-Z]{0,2}(\(\d+\))?(\([a-z]\))?$')
LABEL_CONT_RE = re.compile(r'^\(\d+\)$|^\([a-z]\)$')

# Column boundaries derived empirically by clustering real body-word x0
# positions across the schedule pages (see module docstring) -- this,
# not the printed numeral header, is the actual fix for issue #71.
CRPC_EDGES = [70, 243, 345, 425, 470]
BNSS_EDGES = [98, 210, 290, 345, 421]


def bucket(x0, edges):
    for i, e in enumerate(edges):
        if x0 < e:
            return i
    return 5


def page_lines(page, edges):
    words = page.extract_words()
    lines = defaultdict(list)
    for w in words:
        lines[round(w['top'], 1)].append(w)
    out = []
    for top in sorted(lines):
        ws = sorted(lines[top], key=lambda w: w['x0'])
        texts = [w['text'] for w in ws]
        if set(texts) <= HEADER_NUMS and len(texts) <= 6:
            continue
        cols = defaultdict(list)
        for w in ws:
            cols[bucket(w['x0'], edges)].append(w['text'])
        out.append(tuple(' '.join(cols.get(i, [])).strip() for i in range(6)))
    return out


def classify(raw, kind):
    if not raw:
        return None
    low = raw.lower().strip()
    collapsed = re.sub(r'\s+', '', low)
    if 'according as' in low:
        return 'Conditional'
    if low.strip('. ') == 'ditto':
        return 'Ditto'
    if kind == 'cognizable':
        if 'non-cognizable' in low or 'non- cognizable' in low or 'non cognizable' in low or 'non-cognizable' in collapsed:
            return 'Non-cognizable'
        if 'cognizable' in low or 'cognizable' in collapsed:
            return 'Cognizable'
    else:
        if 'non-bailable' in low or 'non- bailable' in low or 'non bailable' in low or 'non-bailable' in collapsed:
            return 'Non-bailable'
        if 'bailable' in low or 'bailable' in collapsed:
            return 'Bailable'
    return None


def _leading_int(label):
    m = re.match(r'\d+', label)
    return int(m.group()) if m else None


def flatten_rows(pdf_path, start_page, end_page, edges, stop_at_part_ii=True):
    """One continuous, page-boundary-crossing sequence of rows for the whole
    table, each tagged 'label' (isolated section-number line), 'chapter'
    (heading to skip), or 'content' (a normal table line, possibly also
    carrying a label in c1 CrPC-style)."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno in range(start_page, end_page + 1):
            page = pdf.pages[pno]
            page_text_upper = (page.extract_text() or '').upper()
            for c1, c2, c3, c4, c5, c6 in page_lines(page, edges):
                if not any((c1, c2, c3, c4, c5, c6)):
                    continue
                if 'CHAPTER' in c1.upper() or (not c1 and 'CHAPTER' in c2.upper() and not any((c3, c4, c5, c6))):
                    continue
                if 'CLASSIFICATION OF OFFENCES AGAINST' in (c1 + ' ' + c2).upper():
                    if stop_at_part_ii:
                        return rows
                    continue
                token = c1.strip()
                is_label_only = bool(token) and LABEL_RE.match(token) and not any((c2, c3, c4, c5, c6))
                is_label_cont = bool(token) and LABEL_CONT_RE.match(token) and not any((c2, c3, c4, c5, c6))
                rows.append({
                    'label_only': token if is_label_only else None,
                    'label_cont': token if is_label_cont else None,
                    'label_inline': token if (token and LABEL_RE.match(token) and any((c2, c3, c4, c5, c6))) else None,
                    'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5, 'c6': c6,
                })
    return rows


def parse_schedule(pdf_path, start_page, end_page, edges, stop_at_part_ii=True):
    rows = flatten_rows(pdf_path, start_page, end_page, edges, stop_at_part_ii)

    result = {}
    last_resolved = {'cognizable': None, 'bailable': None}

    section = None
    subrows = []
    cur = None
    pending_label_token = None  # for compound labels split across lines, e.g. "61(2)" then "(a)"

    def blank_cur():
        return {'c2': '', 'c3': '', 'c4': '', 'c5': '', 'c6': '',
                'c4_final': '', 'c5_final': '', 'c3_ever_filled': False, 'c3_active': False}

    def feed(cur_dict, c2, c3, c4, c5, c6):
        new_subrow = bool(c3) and cur_dict['c3_ever_filled'] and not cur_dict['c3_active']
        if new_subrow:
            return None, True  # signal caller to close+reopen
        cur_dict['c2'] = (cur_dict['c2'] + ' ' + c2).strip()
        cur_dict['c3'] = (cur_dict['c3'] + ' ' + c3).strip()
        cur_dict['c4'] = (cur_dict['c4'] + ' ' + c4).strip()
        cur_dict['c5'] = (cur_dict['c5'] + ' ' + c5).strip()
        cur_dict['c6'] = (cur_dict['c6'] + ' ' + c6).strip()
        if c4:
            cur_dict['c4_final'] = cur_dict['c4']
        if c5:
            cur_dict['c5_final'] = cur_dict['c5']
        if c3:
            cur_dict['c3_ever_filled'] = True
            cur_dict['c3_active'] = True
        else:
            cur_dict['c3_active'] = False
        return cur_dict, False

    def new_cur(c2, c3, c4, c5, c6):
        return {'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5, 'c6': c6,
                'c4_final': c4, 'c5_final': c5,
                'c3_ever_filled': bool(c3), 'c3_active': bool(c3)}

    def close_subrow():
        nonlocal cur
        if cur is None:
            return
        details = ' '.join(p for p in (cur['c2'], cur['c3'], cur['c4'], cur['c5'], cur['c6']) if p)
        details = re.sub(r'\s+', ' ', details).strip()
        subrows.append({
            'cognizable': classify(cur['c4_final'], 'cognizable'),
            'bailable': classify(cur['c5_final'], 'bailable'),
            'details': details,
        })
        cur = None

    def finish_section():
        nonlocal section, subrows
        if section is None:
            return
        close_subrow()
        resolved_cogs, resolved_bails = [], []
        for s in subrows:
            v = s['cognizable']
            resolved_cogs.append(last_resolved['cognizable'] if v == 'Ditto' and last_resolved['cognizable'] else v)
        for s in subrows:
            v = s['bailable']
            resolved_bails.append(last_resolved['bailable'] if v == 'Ditto' and last_resolved['bailable'] else v)
        cogs = {v for v in resolved_cogs if v}
        bails = {v for v in resolved_bails if v}
        cog_final = cogs.pop() if len(cogs) == 1 else ('Multiple (see details)' if len(cogs) > 1 else 'Unknown')
        bail_final = bails.pop() if len(bails) == 1 else ('Multiple (see details)' if len(bails) > 1 else 'Unknown')
        details = re.sub(r'\s+', ' ', ' '.join(s['details'] for s in subrows)).strip()
        result[section] = [{'cognizable': cog_final, 'bailable': bail_final, 'details': details}]
        if cog_final not in ('Unknown', 'Multiple (see details)'):
            last_resolved['cognizable'] = cog_final
        if bail_final not in ('Unknown', 'Multiple (see details)'):
            last_resolved['bailable'] = bail_final
        section = None
        subrows = []

    def start_section(label, c2='', c3='', c4='', c5='', c6=''):
        nonlocal section, subrows, cur, last_int
        if section is not None:
            finish_section()
        section = label
        subrows = []
        cur = new_cur(c2, c3, c4, c5, c6)
        li = _leading_int(label)
        if li is not None:
            last_int = li

    last_int = None
    held = None  # a content-only row dict held back pending a lookahead check for an isolated label

    def sane(label):
        li = _leading_int(label)
        return last_int is None or li is None or li >= last_int - 1

    i = 0
    n = len(rows)
    while i < n:
        row = rows[i]

        if row['label_cont'] and pending_label_token:
            row = dict(row)
            row['label_only'] = pending_label_token + row['label_cont'] if not any(
                (row['c2'], row['c3'], row['c4'], row['c5'], row['c6'])) else None
            row['label_inline'] = pending_label_token + row['label_cont'] if any(
                (row['c2'], row['c3'], row['c4'], row['c5'], row['c6'])) else None

        if row['label_only'] and sane(row['label_only']):
            label = row['label_only']
            pending_label_token = label
            if held is not None:
                # the held content line is this label's true first line
                start_section(label, held['c2'], held['c3'], held['c4'], held['c5'], held['c6'])
                held = None
            else:
                start_section(label)
            i += 1
            continue

        if row['label_inline'] and sane(row['label_inline']):
            label = row['label_inline']
            pending_label_token = label
            if held is not None:
                # held content belonged to the section that's ending now
                if section is not None:
                    result_before = cur
                    _, restart = feed(cur, held['c2'], held['c3'], held['c4'], held['c5'], held['c6']) if cur else (None, False)
                    if restart:
                        close_subrow()
                        cur = new_cur(held['c2'], held['c3'], held['c4'], held['c5'], held['c6'])
                held = None
            start_section(label, row['c2'], row['c3'], row['c4'], row['c5'], row['c6'])
            i += 1
            continue

        # plain content line (no label on it) -- hold it one step so we can
        # tell whether the *next* row is an isolated label claiming it.
        if held is not None:
            if section is None:
                held = {'c2': row['c2'], 'c3': row['c3'], 'c4': row['c4'], 'c5': row['c5'], 'c6': row['c6']}
                i += 1
                continue
            if cur is None:
                cur = new_cur(held['c2'], held['c3'], held['c4'], held['c5'], held['c6'])
            else:
                _, restart = feed(cur, held['c2'], held['c3'], held['c4'], held['c5'], held['c6'])
                if restart:
                    close_subrow()
                    cur = new_cur(held['c2'], held['c3'], held['c4'], held['c5'], held['c6'])
        held = {'c2': row['c2'], 'c3': row['c3'], 'c4': row['c4'], 'c5': row['c5'], 'c6': row['c6']}
        i += 1

    if held is not None and section is not None:
        if cur is None:
            cur = new_cur(held['c2'], held['c3'], held['c4'], held['c5'], held['c6'])
        else:
            _, restart = feed(cur, held['c2'], held['c3'], held['c4'], held['c5'], held['c6'])
            if restart:
                close_subrow()
                cur = new_cur(held['c2'], held['c3'], held['c4'], held['c5'], held['c6'])
    finish_section()
    return result


if __name__ == '__main__':
    import sys
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    BNSS_PDF = BASE_DIR / 'corpus' / 'sources' / 'bnss' / 'bnss.pdf'
    OUT_PATH = BASE_DIR / 'corpus' / 'data' / 'bnss_schedule.json'

    if '--calibrate' in sys.argv:
        # Sanity check: run the same extractor against the CrPC schedule
        # (which already has a hand/heuristic-produced reference file) and
        # report how often cognizable/bailable agree. See module docstring
        # for why ~100% agreement isn't expected or even desirable (the
        # existing file has some known-bad entries from misread footnote
        # markers, which this extractor deliberately rejects).
        CRPC_PDF = BASE_DIR / 'corpus' / 'sources' / 'crpc' / 'crpc.pdf'
        TRUTH_PATH = BASE_DIR / 'corpus' / 'data' / 'crpc_schedule.json'
        data = parse_schedule(str(CRPC_PDF), 195, 222, CRPC_EDGES)
        truth = json.load(open(TRUTH_PATH))
        mismatches, checked = 0, 0
        for k, v in data.items():
            t = truth.get(k)
            if t is None:
                continue
            checked += 1
            got, exp = v[0], t[0]
            if got['cognizable'] != exp['cognizable'] or got['bailable'] != exp['bailable']:
                mismatches += 1
        print(f"CRPC calibration: checked={checked} agree={checked - mismatches} "
              f"mismatches={mismatches} extracted_sections={len(data)} truth_sections={len(truth)}")
        sys.exit(0)

    bdata = parse_schedule(str(BNSS_PDF), 172, 218, BNSS_EDGES)
    unknown_cog = sum(1 for v in bdata.values() if v[0]['cognizable'] == 'Unknown')
    unknown_bail = sum(1 for v in bdata.values() if v[0]['bailable'] == 'Unknown')
    print(f"Extracted {len(bdata)} BNSS First Schedule sections "
          f"({unknown_cog} with unresolved cognizable, {unknown_bail} with unresolved bailable).")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(bdata, f, indent=2, ensure_ascii=False, sort_keys=False)
    print(f"Wrote {OUT_PATH}")