from pathlib import Path

from corpus.parser import parse_act

pdf = Path("corpus/sources/bnss/bnss.pdf")

sections = parse_act(pdf, "BNSS")

print(f"Parsed {len(sections)} sections\n")

for section in sections[-5:-1]:
    print("=" * 80)
    print(f"{section.unit_type.title()} {section.number}")
    print(section.title)
    print(section.status)
    print(section.metadata)
    print()
    print(section.body[:])
    print()