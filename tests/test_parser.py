from pathlib import Path

from corpus.parser import parse_act

pdf = Path("corpus/sources/ipc/ipc.pdf")

sections = parse_act(pdf, "IPC")

print(f"Parsed {len(sections)} sections\n")

for section in sections[:10]:
    print("=" * 80)
    print(f"{section.unit_type.title()} {section.number}")
    print(section.title)
    print(section.status)
    print(section.metadata)
    print()
    print(section.body[:])
    print()