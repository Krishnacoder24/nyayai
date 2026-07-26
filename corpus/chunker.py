"""
chunks a Section by legal structure, not token count: the main operative
text becomes one Passage, and each Explanation/Illustration/Exception
becomes its own Passage. this keeps every chunk semantically whole (an
Explanation is a complete thought) instead of an arbitrary word-count
window that might cut a sentence in half.
"""

import re
from corpus.schemas import Section, Passage

# Updated to catch markers at the start of a string, after a newline, or after a period
STRUCTURAL_MARKER = re.compile(
    r'(?:^|\n|\.\s+)\s*(Explanation|Illustration|Exception)\s*([0-9a-zA-Z\(\)]*)\s*\.\s*[-—–]?\s*',
    re.IGNORECASE,
)

# Matches structural sub-sections like "(1)", "(2)", "(a)", "(i)" at the start of a line
SUB_SECTION_MARKER = re.compile(
    r'(?:^|\n)\s*(\([0-9a-zA-Z]+\))\s*'
)


def _split_operative_text(section: Section, text: str) -> list[Passage]:
    """Splits the main operative text into smaller structural sub-sections to prevent massive chunks."""
    matches = list(SUB_SECTION_MARKER.finditer(text))
    
    if not matches:
        return [_make_passage(section, text, "body")]
        
    passages = []
    
    # Extract any introductory text before the first sub-section "(1)"
    intro_text = text[:matches[0].start()].strip()
    if intro_text:
        passages.append(_make_passage(section, intro_text, "body_intro"))
        
    for i, match in enumerate(matches):
        clause_id = match.group(1) # e.g., "(1)" or "(a)"
        part_label = f"clause_{clause_id}"
        
        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        sub_text = text[text_start:text_end].strip()
        
        # Prepend the marker back into the text so context is not lost (e.g., "(1) Whoever...")
        full_sub_text = f"{clause_id} {sub_text}"
        
        if full_sub_text.strip():
            passages.append(_make_passage(section, full_sub_text.strip(), part_label))
            
    return passages


def chunk_section(section: Section) -> list[Passage]:
    matches = list(STRUCTURAL_MARKER.finditer(section.body))

    if not matches:
        return _split_operative_text(section, section.body)

    passages = []

    # Everything before the first marker is the main operative text
    # We pass it to our sub-section splitter to handle massive clauses
    main_text = section.body[:matches[0].start()].strip()
    if main_text:
        passages.extend(_split_operative_text(section, main_text))

    for i, match in enumerate(matches):
        label = match.group(1).lower()
        number = match.group(2)
        part = f"{label}_{number}" if number else label

        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(section.body)
        text = section.body[text_start:text_end].strip()

        if text:
            passages.append(_make_passage(section, text, part))

    return passages


def _make_passage(section: Section, text: str, part: str) -> Passage:
    metadata = dict(section.metadata)
    metadata["part"] = part

    return Passage(
        act=section.act,
        unit_type=section.unit_type,
        number=section.number,
        title=section.title,
        status=section.status,
        text=text,
        metadata=metadata,
    )


def chunk_sections(sections: list[Section]) -> list[Passage]:
    passages = []
    for section in sections:
        passages.extend(chunk_section(section))
    return passages