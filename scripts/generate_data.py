"""
generates synthetic training data for NyayAI's token classifier by deliberately
corrupting real, verified legal text pulled from the parsed corpus.

Uses the actual act-specific parsers (corpus/parsers/ipc.py, bns.py, bnss.py,
...) via corpus/parser.py's parse_act(), NOT raw OCR text - a parsed
Section's body already has running headers, the table of contents, and
footnote junk stripped out (see corpus/pdf_utils.py), so there's nothing
left to heuristically re-clean the way raw OCR extraction would need.
This also means the generator automatically scopes itself to whichever
acts actually have a working parser right now (corpus.parser.SUPPORTED_ACTS)
- it won't silently try to train on an act's PDF that has no parser yet,
the way globbing for any *.pdf under a directory would.

Groups each Section's body into a training paragraph directly - one real,
numbered section of the Act per paragraph, not a heuristically-guessed
OCR paragraph boundary. Applies realistic legal errors.

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
    uv run python scripts/generate_data.py --corpus corpus/sources/ipc/ipc.pdf --act IPC --out data/training
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

from corpus.parser import parse_act, SUPPORTED_ACTS
from corpus.schemas import Section
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


# Global config instance
CONFIG = GeneratorConfig()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True,
                       help="path to corpus/sources/ (expects <act>/ subdirs, e.g. ipc/, bns/, bnss/) "
                            "or a single PDF file (requires --act)")
    parser.add_argument("--act", type=str, default=None,
                       help="required when --corpus points to a single PDF file - which act it is "
                            "(must be one of corpus.parser.SUPPORTED_ACTS)")
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

    # Figure out which (act, pdf_path) pairs to parse. Single-file mode
    # needs an explicit --act since a PDF's filename alone doesn't
    # reliably tell us which act it is. Directory mode auto-discovers
    # one PDF per act under SUPPORTED_ACTS - the SAME registry
    # corpus/parser.py itself uses, so this script never tries to
    # "parse" an act that doesn't actually have a parser yet (unlike
    # the old raw-PDF-glob approach, which would happily hand ANY *.pdf
    # under the directory to ocr.pipeline.extract regardless of whether
    # a real parser existed for it).
    acts_to_process: List[Tuple[str, Path]] = []
    if args.corpus.is_file():
        if not args.act:
            parser.error("--act is required when --corpus points to a single PDF file")
        act = args.act.strip().upper()
        if act not in SUPPORTED_ACTS:
            parser.error(f"--act '{args.act}' has no registered parser. Supported: {SUPPORTED_ACTS}")
        acts_to_process = [(act, args.corpus)]
    else:
        for act in SUPPORTED_ACTS:
            pdf_path = _find_pdf_for_act(args.corpus, act)
            if pdf_path is None:
                print(f"  skipping {act}: no PDF found under {args.corpus / act.lower()}/")
                continue
            acts_to_process.append((act, pdf_path))

    print(f"Found {len(acts_to_process)} act(s) to process:")
    for act, pdf_path in acts_to_process:
        print(f"  - {act}: {pdf_path}")

    # Parse each act and build paragraphs directly from Section bodies -
    # no OCR, no heuristic paragraph-boundary guessing. See
    # _build_paragraphs_from_sections's docstring for why.
    paragraphs = []
    pdfs_used = []
    for act, pdf_path in acts_to_process:
        print(f"Processing {act}: {pdf_path.name}")
        try:
            sections = parse_act(pdf_path, act)
            new_paragraphs = _build_paragraphs_from_sections(sections)
            if new_paragraphs:
                paragraphs.extend(new_paragraphs)
                pdfs_used.append({
                    "act": act,
                    "pdf": str(pdf_path),
                    "sections_parsed": len(sections),
                    "paragraphs_used": len(new_paragraphs),
                })
                print(f"  Parsed {len(sections)} sections -> {len(new_paragraphs)} usable paragraphs")
        except Exception as e:
            print(f"  ERROR processing {act} ({pdf_path.name}): {e}")
            continue

    print(f"\nTotal paragraphs extracted: {len(paragraphs)}")

    # split at the PARAGRAPH level, before any window is ever built - not
    # at the individual-example level after the fact.
    #
    # windows overlap by 50% (window_stride=64 against window_size=128),
    # and were previously pooled together across ALL sections before
    # being shuffled and sliced into train/val/test. that meant two
    # windows sharing most of their words could land on opposite sides
    # of the split - and the rebalancer could regenerate a "test"
    # example by resampling a window majority-overlapping with something
    # already in "train", via its own global clean_windows pool. either
    # way, the model could see most of a held-out example's actual
    # content during training, which would quietly inflate
    # train/evaluate.py's eval numbers without meaning anything about
    # real generalization.
    #
    # splitting the PARAGRAPHS first closes this off structurally: since
    # a paragraph is one whole Section body, and windows are only ever
    # built from paragraphs already committed to one split, no window
    # generated for train can share source text with a window in val or
    # test. windowing, corruption, validation, and rebalancing all then
    # run independently per split, using only that split's own pool -
    # rebalancing against the global pool would just reopen the same
    # leak through regeneration instead.
    #
    # split sizes end up approximately (not exactly) 80/10/10 by example
    # count, since longer sections produce more windows than shorter
    # ones - that's the correct tradeoff for avoiding leakage over
    # hitting an exact ratio.
    random.shuffle(paragraphs)
    n_paragraphs = len(paragraphs)
    n_train_p = int(n_paragraphs * args.train_split)
    n_val_p = int(n_paragraphs * args.val_split)
    paragraph_splits = {
        "train": paragraphs[:n_train_p],
        "val": paragraphs[n_train_p:n_train_p + n_val_p],
        "test": paragraphs[n_train_p + n_val_p:],
    }
    split_fractions = {
        "train": args.train_split,
        "val": args.val_split,
        "test": max(0.0, 1 - args.train_split - args.val_split),
    }

    splits = {}
    total_skipped = 0
    for split_name, split_paragraphs in paragraph_splits.items():
        clean_windows = []
        for para in split_paragraphs:
            clean_windows.extend(_build_windows(para))

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
        total_skipped += skipped

        # per-type target scales with this split's share of
        # --min-examples, so val/test (typically 10% each) aren't held
        # to the same absolute per-type target as train - floor of 20
        # so a small split still gets SOME rebalancing rather than none.
        split_min = max(20, round(args.min_examples * split_fractions[split_name]))
        examples = _rebalance_with_regeneration(examples, clean_windows, split_min, args.no_validate)
        random.shuffle(examples)
        splits[split_name] = examples

        print(f"  {split_name}: {len(split_paragraphs)} sections -> {len(clean_windows)} windows -> {len(examples)} examples")

    if total_skipped > 0:
        print(f"Skipped {total_skipped} invalid examples (across all splits)")

    total_examples = sum(len(ex) for ex in splits.values())
    if total_examples < args.min_examples:
        print(f"Warning: Only {total_examples} total examples generated. Expected at least {args.min_examples}.")
        print("Consider adding more PDFs or reducing filtering.")

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


def _find_pdf_for_act(corpus_dir: Path, act: str) -> Optional[Path]:
    """finds the source PDF for a given act, following this project's
    corpus/sources/<act>/ convention (see bns.py's docstring, and
    test_parser.py's corpus/sources/ipc/ipc.pdf). picks the first .pdf
    found if there's more than one - each act is expected to have
    exactly one source PDF."""
    act_dir = corpus_dir / act.lower()
    if not act_dir.is_dir():
        return None
    pdfs = sorted(act_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def _build_paragraphs_from_sections(sections: List[Section]) -> List[str]:
    """turns parsed Section bodies directly into training paragraphs.

    replaces the old raw-OCR heuristics entirely (paragraph-boundary
    guessing off vertical gaps/ALL-CAPS headers/numbered-line starts,
    plus a metadata-line filter for page numbers and running headers) -
    a parsed Section's body has ALL of that already stripped out by
    pdf_utils.py and the act parser itself (running headers via
    remove_repeated_headers, the TOC via each parser's own split logic,
    footnote junk via _resolve_footnote_markers). there's nothing left
    to heuristically clean up here.

    each Section is its own natural paragraph unit - one real, numbered
    provision of the Act, not a guessed OCR paragraph boundary. no need
    to additionally cap length the way the old max_paragraph_words split
    did for raw OCR text either: _build_windows already slices anything
    longer than window_size into multiple training examples regardless
    of how long the paragraph handed to it is."""
    paragraphs = []
    for section in sections:
        body = section.body.strip()
        if not body:
            continue
        if len(body.split()) < CONFIG.min_paragraph_words:
            # too short to be useful training text - e.g. a repealed
            # section's stub body, which is just "[<title>]"
            continue
        paragraphs.append(body)
    return paragraphs


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


def _typo(word: str, _exclude: Optional[Set[str]] = None) -> str:
    """Apply a keyboard-based typo."""
    if len(word) < 2:
        return word

    exclude = _exclude or set()
    available = [s for s in ("swap", "delete", "insert", "substitute") if s not in exclude]
    if not available:
        # every strategy already ruled out on retry (only "delete" ever
        # retries, and it excludes itself each time - see below) - give
        # up and return the word unchanged rather than recursing forever
        return word

    strategy = random.choice(available)
    
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
            # bounded retry: exclude "delete" this time instead of an
            # unbounded recursive call - guarantees termination within
            # one extra call, since no other branch ever retries
            return _typo(word, _exclude=exclude | {"delete"})
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
    acts_used: List[Dict[str, Any]],
    stats: Dict[str, Any],
    splits: Dict[str, List[Tuple[List[str], List[str]]]],
    checksums: Dict[str, str]
) -> None:
    """Generate a manifest file with dataset provenance."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "corpus_path": str(args.corpus),
        "acts_used": acts_used,
        "num_acts": len(acts_used),
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
        "generator_version": "3.0",  # bumped: now sourced from parsed Sections, not raw OCR
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