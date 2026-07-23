import sys
from pathlib import Path
from corpus.parser import _PARSERS

def main():
    if "BNSS" not in _PARSERS:
        print("Error: BNSS parser not registered in corpus/parser.py")
        sys.exit(1)

    parser = _PARSERS["BNSS"]
    pdf_path = Path("corpus/sources/bnss/bnss.pdf")
    
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)
        
    print(f"Parsing {pdf_path} using {parser.__class__.__name__}...")
    sections = parser.parse(pdf_path)
    
    print(f"\nSuccessfully parsed {len(sections)} sections!")
    print("\n--- SAMPLE SECTIONS (Testing Superscript Brackets) ---\n")
    
    # We want to specifically show a section with a footnote replacement, 
    # like Section 106 which we know has a footnote from earlier tests
    for sec in sections:
        if sec.number in ("1", "106", "531"):
            print(f"[{sec.number}] {sec.title}")
            print(f"Chapter: {sec.metadata.get('chapter')}")
            # Print first 200 chars of body
            body_preview = sec.body[:250].replace('\n', ' ')
            print(f"Body Preview: {body_preview}...")
            print("-" * 50)
            
if __name__ == "__main__":
    main()
