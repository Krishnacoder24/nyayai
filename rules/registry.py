"""
central registry of all rule-based error checkers.

to add a new checker:
  1. import its public function here
  2. append it to RULES

that's it — engine.py never needs to change.
"""

from rules.citation_checker import check_citations
from rules.entity_checker import check_entities
from rules.cross_reference_checker import check_cross_references
from rules.spelling_checker import check_spelling

RULES = [
    check_citations,
    check_entities,
    check_cross_references,
    check_spelling,
]