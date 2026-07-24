from pathlib import Path

from corpus.parser import parse_act

pdf = Path("corpus/sources/bns/bns.pdf")

sections = parse_act(pdf, "BNS")

print(f"Parsed {len(sections)} sections\n")

for section in sections[:2]:
    print("=" * 80)
    print(f"{section.unit_type.title()} {section.number}")
    print(section.title)
    print(section.status)
    print(section.metadata)
    print()
    print(section.body[:])
    print()