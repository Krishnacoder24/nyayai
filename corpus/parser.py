"""
router: dispatches to the correct act-specific parser. no parsing logic
lives here - each act's document grammar differs enough (IPC's
century-old amendment noise vs. the Constitution's Part -> Chapter ->
Article structure) that a single implementation would just become a
pile of if/else branches. per issue #26, parsers are deliberately
independent - no shared base class or inheritance - so add new acts by
registering a parser class in _PARSERS, not by editing this dispatch
logic or by inheriting from another act's parser.
"""

from pathlib import Path

from corpus.schemas import Section
from corpus.parsers.ipc import IPCParser
# from corpus.parsers.bns import BNSParser
# from corpus.parsers.bnss import BNSSParser
# from corpus.parsers.cpc import CPCParser
# from corpus.parsers.constitution import ConstitutionParser

_PARSERS = {
    "IPC": IPCParser(),
    # "BNS": BNSParser(),
    # "BNSS": BNSSParser(),
    # "CPC": CPCParser(),
    # "CONSTITUTION": ConstitutionParser(),
}


def parse_act(pdf_path: Path, act: str) -> list[Section]:
    parser = _PARSERS.get(act.upper())
    if parser is None:
        raise ValueError(f"no parser registered for act '{act}' - add one to corpus/parsers/")
    return parser.parse(pdf_path)