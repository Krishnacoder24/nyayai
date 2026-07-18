"""
generates synthetic training data for NyayAI's token classifier by deliberately
corrupting real, verified legal text pulled from real PDFs.

Supports multiple PDFs across legal domains (IPC, BNS, BNSS, Constitution, CPC, etc.)
Groups OCR lines into paragraphs for context. Applies realistic legal errors.

Corruption order matters and is NOT arbitrary:
  1. GRAM first - grammar corruptions (dropping/duplicating a word) change
     the word count. every later step labels words by index, so if this
     ran last it would silently invalidate every index-based label applied
     before it.
  2. CITE second - citation corruption finds "Section N ACT" patterns and
     swaps the number. in-place (word count unchanged), so it's safe to run
     after GRAM and before SPELL.
  3. SPELL last - character-level typos on whatever words are still
     available (not already corrupted by an earlier step).

Labeling convention:
  - GRAM errors on words (wrong preposition, wrong modal): label the corrupted word
  - GRAM errors on missing words (dropped article): label the word after the gap
  - CITE errors: label the entire citation span (B-CITE, I-CITE)
  - SPELL errors: label the corrupted word (B-SPELL)

Usage:
    uv run python scripts/generate_data.py --corpus corpus/sources/ --out data/training
"""

import argparse
import json
import random
import re
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ocr.pipeline import extract
from model.schemas import LABELS
from rules.citation_checker import CITATION_PATTERNS


@dataclass(frozen=True)
class GeneratorConfig:
    """Immutable configuration for the dataset generator."""
    min_paragraph_words: int = 20
    max_paragraph_words: int = 150
    window_size: int = 128
    window_stride: int = 64
    corruption_rate: float = 0.7
    min_examples_per_type: int = 500
    corruption_weights: Dict[str, float] = field(default_factory=lambda: {
        "spell": 0.45,
        "gram": 0.30,
        "cite": 0.20,
        "mixed": 0.05,
    })
    
    # QWERTY keyboard neighbors for realistic typos
    qwerty_neighbors: Dict[str, str] = field(default_factory=lambda: {
        "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
        "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
        "a": "qsz", "s": "awdz", "d": "serfx", "f": "drtgc", "g": "ftyhv",
        "h": "gyujb", "j": "huikn", "k": "jiolm", "l": "kop",
        "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
        "n": "bhjm", "m": "njk",
    })
    
    wrong_prepositions: Dict[str, List[str]] = field(default_factory=lambda: {
        "in": ["on", "at"], "on": ["in", "at"], "at": ["in", "on"],
        "of": ["for", "to"], "for": ["of", "to"], "to": ["for", "of"],
        "by": ["with", "from"], "with": ["by", "from"], "from": ["by", "with"],
        "under": ["over", "in"],
    })
    
    articles: Set[str] = field(default_factory=lambda: {"the", "a", "an"})
    
    legal_modals: Dict[str, List[str]] = field(default_factory=lambda: {
        "shall": ["may", "will", "should"],
        "may": ["shall", "can", "might"],
        "must": ["may", "shall"],
        "will": ["shall", "may"],
    })
    
    legal_connectives: Dict[str, List[str]] = field(default_factory=lambda: {
        "provided": ["subject", "notwithstanding"],
        "notwithstanding": ["provided", "despite"],
        "subject": ["provided", "notwithstanding"],
    })
    
    metadata_patterns: List[str] = field(default_factory=lambda: [
        r"^Page \d+",
        r"^www\.",
        r"\.(com|org|in|gov|pdf)$",
        r"^Government of",
        r"^[A-Z]{3,}$",
        r"^\d+$",
        r"^\d+ of \d+$",
    ])


# Global config instance
CONFIG = GeneratorConfig()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True, 
                       help="path to corpus directory containing IPC/, BNS/, etc. or specific PDF")
    parser.add_argument("--out", type=Path, default=Path("data/training"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--min-examples", type=int, default=500,
                       help="minimum examples needed before rebalancing")
    parser.add_argument("--no-validate", action="store_true", default=False,
                       help="skip BIO label validation (faster but riskier)")
    parser.add_argument("--no-manifest", action="store_true", default=False,
                       help="skip manifest file generation")
    parser.add_argument("--corruption-weights", type=str, default=None,
                       help="comma-separated corruption weights: spell=0.45,gram=0.30,cite=0.20,mixed=0.05")
    args = parser.parse_args()

    random.seed(args.seed)

    # Parse and validate custom corruption weights if provided
    global CONFIG
    if args.corruption_weights:
        weights = {}
        for item in args.corruption_weights.split(","):
            key, val = item.split("=")
            weights[key.strip()] = float(val.strip())
        
        # Validate weights sum to approximately 1.0
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            print(f"Warning: Corruption weights sum to {total:.2f}, normalizing...")
            weights = {k: v/total for k, v in weights.items()}
        
        CONFIG = GeneratorConfig(**{**CONFIG.__dict__, "corruption_weights": weights})

    # Load all PDFs from corpus
    pdf_paths = []
    if args.corpus.is_file():
        pdf_paths = [args.corpus]
    else:
        for pdf in args.corpus.rglob("*.pdf"):
            pdf_paths.append(pdf)

    print(f"Found {len(pdf_paths)} PDF files:")
    for pdf in pdf_paths[:5]:
        print(f"  - {pdf}")
    if len(pdf_paths) > 5:
        print(f"  ... and {len(pdf_paths) - 5} more")

    # Extract and build paragraphs
    paragraphs = []
    pdfs_used = []
    for i, pdf_path in enumerate(pdf_paths):
        print(f"Processing {i+1}/{len(pdf_paths)}: {pdf_path.name}")
        try:
            spans = extract(pdf_path)
            new_paragraphs = _build_paragraphs(spans)
            if new_paragraphs:
                paragraphs.extend(new_paragraphs)
                pdfs_used.append(str(pdf_path))
                print(f"  Extracted {len(new_paragraphs)} paragraphs")
        except Exception as e:
            print(f"  ERROR processing {pdf_path.name}: {e}")
            continue

    print(f"\nTotal paragraphs extracted: {len(paragraphs)}")

    # Generate clean windows from all paragraphs (no corruption)
    clean_windows = []
    for para in paragraphs:
        windows = _build_windows(para)
        clean_windows.extend(windows)
    
    print(f"Built {len(clean_windows)} clean windows")

    # Generate examples
    examples = []
    skipped = 0
    for window in clean_windows:
        new_words, new_labels = _generate_example_from_clean_window(window)
        if not args.no_validate:
            if _validate_example(new_words, new_labels):
                examples.append((new_words, new_labels))
            else:
                skipped += 1
        else:
            examples.append((new_words, new_labels))

    if skipped > 0:
        print(f"Skipped {skipped} invalid examples")

    # Ensure minimum examples and balance using regeneration
    if len(examples) < args.min_examples:
        print(f"Warning: Only {len(examples)} examples generated. Expected at least {args.min_examples}.")
        print("Consider adding more PDFs or reducing filtering.")

    # Rebalance using regeneration instead of duplication
    examples = _rebalance_with_regeneration(examples, clean_windows, args.min_examples, args.no_validate)

    # Shuffle and split
    random.shuffle(examples)
    n = len(examples)
    n_train = int(n * args.train_split)
    n_val = int(n * args.val_split)

    splits = {
        "train": examples[:n_train],
        "val": examples[n_train:n_train + n_val],
        "test": examples[n_train + n_val:],
    }

    # Write and report
    args.out.mkdir(parents=True, exist_ok=True)
    for name, split_examples in splits.items():
        path = args.out / f"{name}.jsonl"
        with open(path, "w") as f:
            for words, labels in split_examples:
                f.write(json.dumps({"words": words, "labels": labels}) + "\n")
        print(f"{name}: {len(split_examples)} examples -> {path}")

    # Compute checksums for manifest
    checksums = {}
    for name in splits.keys():
        path = args.out / f"{name}.jsonl"
        if path.exists():
            with open(path, "rb") as f:
                checksums[name] = hashlib.sha256(f.read()).hexdigest()

    # Statistics
    stats = _report_statistics(splits)

    # Generate manifest
    if not args.no_manifest:
        _generate_manifest(args, pdfs_used, stats, splits, checksums)


def _build_paragraphs(spans: List[Any]) -> List[str]:
    """Group OCR spans into coherent paragraphs using heuristics."""
    paragraphs = []
    current_para = []
    
    for span in spans:
        text = span.text.strip()
        if not text:
            continue
            
        # Skip metadata lines
        if _is_metadata_line(text):
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            continue
        
        # Detect paragraph boundaries using multiple heuristics
        is_new_paragraph = False
        
        # 1. Check for indentation (if available in span)
        if hasattr(span, 'is_paragraph_start') and span.is_paragraph_start:
            is_new_paragraph = True
        
        # 2. Check for blank line gaps (if available)
        if hasattr(span, 'vertical_gap') and span.vertical_gap > 10:  # pixels
            is_new_paragraph = True
        
        # 3. Check for section headers (ALL CAPS, short)
        if len(text) < 30 and text.isupper() and len(text.split()) <= 4:
            is_new_paragraph = True
        
        # 4. Check for numbered sections
        if re.match(r"^\d+\.\s+[A-Z]", text):
            is_new_paragraph = True
        
        # Start new paragraph if needed
        if is_new_paragraph and current_para:
            paragraphs.append(" ".join(current_para))
            current_para = []
        
        current_para.append(text)
        
        # End paragraph if long enough
        if len(" ".join(current_para).split()) > CONFIG.max_paragraph_words:
            paragraphs.append(" ".join(current_para))
            current_para = []
    
    if current_para:
        paragraphs.append(" ".join(current_para))
    
    # Filter very short paragraphs
    return [p for p in paragraphs if len(p.split()) >= CONFIG.min_paragraph_words]


def _is_metadata_line(text: str) -> bool:
    """Check if line looks like header/footer/page number/metadata."""
    text = text.strip()
    
    # Check against patterns
    for pattern in CONFIG.metadata_patterns:
        if re.search(pattern, text, re.I):
            return True
    
    # Very short all-caps lines
    if len(text) < 20 and text.isupper():
        return True
    
    # Lines with only numbers and punctuation
    if re.match(r"^[\d\.,\s]+$", text):
        return True
    
    return False


def _build_windows(paragraph: str) -> List[List[str]]:
    """Convert paragraph into clean windows (no corruption)."""
    words = paragraph.split()
    if not words:
        return []
    
    windows = []
    
    # For short paragraphs, use as-is
    if len(words) <= CONFIG.window_size:
        windows.append(words)
        return windows
    
    # For long paragraphs, use sliding window
    for start in range(0, len(words) - CONFIG.window_size + 1, CONFIG.window_stride):
        window = words[start:start + CONFIG.window_size]
        if len(window) >= 20:  # Skip very short windows
            windows.append(window)
    
    # If we didn't cover the end, add a final window
    if len(words) % CONFIG.window_stride != 0:
        final_start = len(words) - CONFIG.window_size
        if final_start > 0 and final_start % CONFIG.window_stride != 0:
            window = words[final_start:]
            if len(window) >= 20:
                windows.append(window)
    
    return windows


def _generate_example_from_clean_window(window: List[str]) -> Tuple[List[str], List[str]]:
    """Generate one training example from a clean window."""
    words = list(window)
    labels = ["O"] * len(words)
    
    # Decide corruption strategy
    if random.random() < CONFIG.corruption_rate:
        # Weighted selection of corruption type
        strategy = random.choices(
            list(CONFIG.corruption_weights.keys()),
            weights=list(CONFIG.corruption_weights.values())
        )[0]
        
        if strategy == "mixed":
            # Apply 2-3 corruptions
            num_corruptions = random.randint(2, 3)
            # Ensure we get a mix of types
            types = ["gram", "cite", "spell"]
            selected = random.sample(types, min(num_corruptions, len(types)))
            # Apply in correct order
            selected = sorted(selected, key={"gram": 0, "cite": 1, "spell": 2}.get)
            for corrupt_type in selected:
                if corrupt_type == "gram":
                    words, labels = _apply_gram_corruption(words, labels)
                elif corrupt_type == "cite":
                    words, labels = _apply_cite_corruption(words, labels)
                elif corrupt_type == "spell":
                    words, labels = _apply_spell_corruption(words, labels)
        elif strategy == "gram":
            words, labels = _apply_gram_corruption(words, labels)
        elif strategy == "cite":
            words, labels = _apply_cite_corruption(words, labels)
        elif strategy == "spell":
            words, labels = _apply_spell_corruption(words, labels)
    
    return words, labels


def _validate_example(words: List[str], labels: List[str]) -> bool:
    """Validate BIO labels for correctness using model.schemas."""
    if not words or not labels:
        return False
    
    if len(words) != len(labels):
        return False
    
    valid_prefixes = {"B", "I", "O"}
    
    # Derive valid tags from LABELS constant
    valid_tags = {label.split("-")[1] for label in LABELS if "-" in label}
    
    for i, label in enumerate(labels):
        if label == "O":
            continue
        
        # Check label format
        if "-" not in label:
            return False
        
        prefix, _, tag = label.partition("-")
        
        if prefix not in valid_prefixes:
            return False
        
        if tag not in valid_tags:
            return False
        
        # Check I- tag has previous label
        if prefix == "I":
            if i == 0:
                return False
            prev_prefix, _, prev_tag = labels[i-1].partition("-")
            if prev_prefix not in ("B", "I") or prev_tag != tag:
                return False
        
        # Check B- tag doesn't continue a span
        if prefix == "B" and i > 0:
            prev_prefix, _, prev_tag = labels[i-1].partition("-")
            if prev_prefix in ("B", "I") and prev_tag == tag:
                return False
    
    return True


def _rebalance_with_regeneration(
    examples: List[Tuple[List[str], List[str]]],
    clean_windows: List[List[str]],
    min_per_type: int,
    skip_validation: bool = False
) -> List[Tuple[List[str], List[str]]]:
    """Rebalance by regenerating fresh corruptions from clean windows."""
    if len(examples) < min_per_type * 3:
        return examples
    
    # Categorize examples
    by_type = defaultdict(list)
    for words, labels in examples:
        error_types = set()
        for label in labels:
            if label.startswith("B-"):
                error_types.add(label.split("-")[1])
        if error_types:
            if len(error_types) > 1:
                by_type["mixed"].append((words, labels))
            else:
                by_type[list(error_types)[0].lower()].append((words, labels))
        else:
            by_type["clean"].append((words, labels))
    
    # Check if we need rebalancing
    needs_rebalance = any(len(ex) < min_per_type for typ, ex in by_type.items() 
                         if typ != "clean" and typ != "mixed")
    
    if not needs_rebalance:
        return examples
    
    # Regenerate for underrepresented types
    print(f"Rebalancing dataset...")
    balanced = []
    
    for typ, ex_list in by_type.items():
        if typ == "clean":
            balanced.extend(ex_list)
            continue
        
        if typ == "mixed":
            balanced.extend(ex_list)
            continue
        
        if len(ex_list) >= min_per_type:
            balanced.extend(ex_list)
        else:
            # Need more examples of this type - regenerate from clean windows
            print(f"  Regenerating {typ} examples: {len(ex_list)} -> {min_per_type}")
            needed = min_per_type - len(ex_list)
            
            # Keep existing examples
            balanced.extend(ex_list)
            
            # Generate new examples from clean windows
            new_examples = []
            attempts = 0
            max_attempts = needed * 20
            
            while len(new_examples) < needed and attempts < max_attempts:
                attempts += 1
                # Pick a random clean window
                window = random.choice(clean_windows)
                words = list(window)
                
                # Apply specific corruption type to the clean window
                if typ == "gram":
                    new_words, new_labels = _apply_gram_corruption(words, ["O"] * len(words))
                elif typ == "cite":
                    new_words, new_labels = _apply_cite_corruption(words, ["O"] * len(words))
                elif typ == "spell":
                    new_words, new_labels = _apply_spell_corruption(words, ["O"] * len(words))
                else:
                    continue
                
                # Validate
                if skip_validation or _validate_example(new_words, new_labels):
                    # Verify it actually has the right type
                    has_type = any(l.split("-")[1].lower() == typ for l in new_labels if l.startswith("B-"))
                    if has_type:
                        new_examples.append((new_words, new_labels))
            
            print(f"    Generated {len(new_examples)} fresh examples from clean windows")
            balanced.extend(new_examples)
    
    random.shuffle(balanced)
    return balanced


# --- GRAM Corruptions ---

def _apply_gram_corruption(words: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Apply one grammar corruption."""
    strategies = [
        _drop_article,
        _duplicate_word,
        _wrong_preposition,
        _wrong_modal,
        _wrong_connective,
    ]
    # Don't drop articles if the text is too short
    if len(words) < 5:
        strategies.remove(_drop_article)
    
    strategy = random.choice(strategies)
    return strategy(words, labels)


def _drop_article(words: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Remove an article, label the following word."""
    candidates = [i for i, w in enumerate(words) if w.lower().strip(".,;:") in CONFIG.articles]
    if not candidates or len(words) <= 3:
        return words, labels
    
    idx = random.choice(candidates)
    new_words = words[:idx] + words[idx + 1:]
    new_labels = labels[:idx] + labels[idx + 1:]
    
    # Label the word after the gap
    if idx < len(new_labels):
        new_labels[idx] = "B-GRAM"
    
    return new_words, new_labels


def _duplicate_word(words: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Duplicate a word."""
    # Prefer content words (not articles, prepositions)
    eligible = []
    for i, w in enumerate(words):
        if labels[i] == "O" and len(w) > 2:
            lower_w = w.lower().strip(".,;:")
            if lower_w not in CONFIG.articles and lower_w not in CONFIG.wrong_prepositions:
                eligible.append(i)
    
    if not eligible or len(words) >= 200:
        return words, labels
    
    idx = random.choice(eligible)
    new_words = words[:idx] + [words[idx]] + words[idx:]
    new_labels = labels[:idx] + ["B-GRAM"] + ["I-GRAM"] + labels[idx + 1:]
    
    return new_words, new_labels


def _wrong_preposition(words: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Replace a preposition with the wrong one."""
    candidates = [
        i for i, w in enumerate(words)
        if labels[i] == "O" and w.lower().strip(".,;:") in CONFIG.wrong_prepositions
    ]
    if not candidates:
        return words, labels
    
    idx = random.choice(candidates)
    key = words[idx].lower().strip(".,;:")
    words = list(words)
    labels = list(labels)
    words[idx] = random.choice(CONFIG.wrong_prepositions[key])
    labels[idx] = "B-GRAM"
    
    return words, labels


def _wrong_modal(words: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Replace a legal modal verb (shall/may)."""
    candidates = [
        i for i, w in enumerate(words)
        if labels[i] == "O" and w.lower().strip(".,;:") in CONFIG.legal_modals
    ]
    if not candidates:
        return words, labels
    
    idx = random.choice(candidates)
    key = words[idx].lower().strip(".,;:")
    words = list(words)
    labels = list(labels)
    words[idx] = random.choice(CONFIG.legal_modals[key])
    labels[idx] = "B-GRAM"
    
    return words, labels


def _wrong_connective(words: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Replace legal connectives (provided/notwithstanding)."""
    candidates = [
        i for i, w in enumerate(words)
        if labels[i] == "O" and w.lower().strip(".,;:") in CONFIG.legal_connectives
    ]
    if not candidates:
        return words, labels
    
    idx = random.choice(candidates)
    key = words[idx].lower().strip(".,;:")
    words = list(words)
    labels = list(labels)
    words[idx] = random.choice(CONFIG.legal_connectives[key])
    labels[idx] = "B-GRAM"
    
    return words, labels


# --- CITE Corruptions ---

def _apply_cite_corruption(words: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Corrupt a citation to a plausible wrong section."""
    line = " ".join(words)
    matches = [(pattern, m) for pattern, _ in CITATION_PATTERNS for m in re.finditer(pattern, line)]
    if not matches:
        return words, labels

    _, match = random.choice(matches)
    matched_text = match.group(0)
    match_words = matched_text.split()
    start_idx = _find_word_subsequence(words, match_words)
    if start_idx is None:
        return words, labels

    span_indices = range(start_idx, start_idx + len(match_words))
    if any(labels[i] != "O" for i in span_indices):
        return words, labels

    words = list(words)
    labels = list(labels)

    # Find the section number
    number_idx = None
    for i in span_indices:
        token = words[i].strip(".,;:")
        if token.isdigit() or re.match(r"^\d+[A-Z]?$", token):
            number_idx = i
            break

    if number_idx is None:
        return words, labels

    # Corrupt to a nearby section (much more realistic)
    digits = re.match(r"\d+", words[number_idx])
    if not digits:
        return words, labels
    
    current_num = int(digits.group())
    
    # 70% chance: nearby but plausible (e.g., 302 -> 304)
    # 30% chance: clearly wrong (e.g., 302 -> 872)
    if random.random() < 0.7:
        # Nearby: ±1-10, avoid 0 and negative
        offsets = list(range(1, 6)) + list(range(-5, 0))
        offset = random.choice(offsets)
        if current_num + offset <= 0:
            offset = random.choice([1, 2, 3])
        new_num = current_num + offset
    else:
        # Far wrong
        new_num = current_num + random.randint(500, 900)
    
    new_number_str = str(new_num)
    words[number_idx] = words[number_idx].replace(digits.group(), new_number_str, 1)

    for i, idx in enumerate(span_indices):
        labels[idx] = "B-CITE" if i == 0 else "I-CITE"

    return words, labels


def _find_word_subsequence(words: List[str], sub: List[str]) -> Optional[int]:
    """Find start index of subsequence."""
    for i in range(len(words) - len(sub) + 1):
        if words[i:i + len(sub)] == sub:
            return i
    return None


# --- SPELL Corruptions ---

def _apply_spell_corruption(words: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """Apply a spelling typo."""
    # Prefer longer words or legal terms
    legal_terms = {"section", "act", "court", "judge", "offence", "punishment", 
                   "sentence", "appeal", "witness", "evidence", "trial"}
    
    eligible = []
    for i, w in enumerate(words):
        if labels[i] == "O" and len(w) >= 3:
            if w.lower() in legal_terms:
                eligible.insert(0, i)  # Prioritize legal terms
            else:
                eligible.append(i)
    
    if not eligible:
        return words, labels
    
    idx = random.choice(eligible[:20])  # Limit to first 20 candidates
    words = list(words)
    labels = list(labels)
    words[idx] = _typo(words[idx])
    labels[idx] = "B-SPELL"
    
    return words, labels


def _typo(word: str) -> str:
    """Apply a keyboard-based typo."""
    if len(word) < 2:
        return word
    
    strategy = random.choice(["swap", "delete", "insert", "substitute"])
    
    # Don't corrupt the first or last char more often
    pos = random.randint(0, len(word) - 1)
    if random.random() < 0.3:  # 30% chance to avoid edges
        pos = random.randint(1, len(word) - 2)
    
    if strategy == "swap" and len(word) >= 2:
        pos = random.randint(0, len(word) - 2)
        chars = list(word)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return "".join(chars)
    
    if strategy == "delete":
        if len(word) <= 3:  # Don't make words too short
            return _typo(word)  # Retry
        return word[:pos] + word[pos + 1:]
    
    if strategy == "insert":
        neighbors = CONFIG.qwerty_neighbors.get(word[pos].lower(), word[pos])
        return word[:pos] + random.choice(neighbors) + word[pos:]
    
    # substitute
    neighbors = CONFIG.qwerty_neighbors.get(word[pos].lower(), word[pos])
    return word[:pos] + random.choice(neighbors) + word[pos + 1:]


# --- Statistics and Manifest ---

def _report_statistics(splits: Dict[str, List[Tuple[List[str], List[str]]]]) -> Dict[str, Any]:
    """Print comprehensive dataset statistics and return stats dict."""
    total_examples = sum(len(ex) for ex in splits.values())
    print(f"\n{'='*50}")
    print(f"Dataset Statistics")
    print(f"{'='*50}")
    print(f"Total examples: {total_examples}")
    
    # Per-split counts
    for name, split in splits.items():
        if split:
            print(f"  {name}: {len(split)} ({len(split)/total_examples*100:.1f}%)")
    
    # Token and label stats
    label_counts = Counter()
    token_counts = []
    corruption_counts = defaultdict(int)
    
    for split in splits.values():
        for words, labels in split:
            token_counts.append(len(words))
            label_counts.update(labels)
            
            # Count corruption types
            seen = set()
            for label in labels:
                if label.startswith("B-"):
                    typ = label.split("-")[1].lower()
                    if typ not in seen:
                        corruption_counts[typ] += 1
                        seen.add(typ)
    
    # Token stats
    avg_tokens = 0
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"\nToken Statistics:")
        print(f"  Avg tokens/example: {avg_tokens:.1f}")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
    
    # Label distribution
    total_labels = sum(label_counts.values())
    print(f"\nLabel Distribution:")
    for label, count in sorted(label_counts.items()):
        if count > 0:
            pct = count / total_labels * 100
            print(f"  {label:10s}: {count:8d} ({pct:5.2f}%)")
    
    # Corruption type distribution
    print(f"\nCorruption Type Distribution:")
    for typ, count in sorted(corruption_counts.items()):
        if count > 0:
            pct = count / len(splits["train"]) * 100
            print(f"  {typ:10s}: {count:8d} ({pct:5.1f}% of examples)")
    
    # Clean examples (no corruption)
    clean_count = 0
    for split in splits.values():
        for words, labels in split:
            if all(l == "O" for l in labels):
                clean_count += 1
    print(f"\nClean examples: {clean_count} ({clean_count/total_examples*100:.1f}%)")
    
    print(f"{'='*50}\n")
    
    # Return stats for manifest
    return {
        "total_examples": total_examples,
        "splits": {name: len(split) for name, split in splits.items()},
        "avg_tokens": avg_tokens,
        "label_distribution": dict(label_counts),
        "corruption_distribution": dict(corruption_counts),
        "clean_examples": clean_count,
    }


def _generate_manifest(
    args: argparse.Namespace,
    pdfs_used: List[str],
    stats: Dict[str, Any],
    splits: Dict[str, List[Tuple[List[str], List[str]]]],
    checksums: Dict[str, str]
) -> None:
    """Generate a manifest file with dataset provenance."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "corpus_path": str(args.corpus),
        "pdfs_used": pdfs_used,
        "num_pdfs": len(pdfs_used),
        "examples": stats["total_examples"],
        "splits": stats["splits"],
        "label_distribution": stats["label_distribution"],
        "corruption_distribution": stats["corruption_distribution"],
        "clean_examples": stats["clean_examples"],
        "avg_tokens_per_example": stats["avg_tokens"],
        "corruption_weights": CONFIG.corruption_weights,
        "corruption_rate": CONFIG.corruption_rate,
        "window_size": CONFIG.window_size,
        "window_stride": CONFIG.window_stride,
        "min_paragraph_words": CONFIG.min_paragraph_words,
        "max_paragraph_words": CONFIG.max_paragraph_words,
        "checksums": checksums,
        "generator_version": "2.1",
    }
    
    # Try to get git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        manifest["git_commit"] = commit
    except (subprocess.SubprocessError, FileNotFoundError):
        manifest["git_commit"] = None
    
    # Write manifest
    manifest_path = args.out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()