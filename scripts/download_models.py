"""
one-time setup script. handles the two things that ARE safe to fully
automate, and gives clear manual instructions for the one thing that isn't.

automated:
  - spacy's en_core_web_sm model (rules/entity_checker.py)
  - InLegalBERT base weights, pre-cached locally (optional --skip-hf-cache)

NOT automated: downloading the actual Act PDFs (IPC/BNS/BNSS/CPC/Constitution)
from IndiaCode. IndiaCode's repository (a DSpace instance) uses opaque
per-document bitstream IDs with no stable, predictable URL pattern - while
researching this I found two DIFFERENT bitstream IDs pointing at what
should be the same IPC PDF. hardcoding a guessed link here risks silently
pointing at a stale or wrong version of an actual legal document - exactly
what this project's "verified sources only" principle (see
corpus/data/ipc_bns_mapping.py) is meant to guard against. safer to grab
these by hand once and confirm you have the current version.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_sm"

REQUIRED_ACTS = {
    "IPC": "Indian Penal Code, 1860",
    "BNS": "Bharatiya Nyaya Sanhita, 2023",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "CPC": "Code of Civil Procedure, 1908",
    "CONSTITUTION": "Constitution of India, 1950",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-spacy", action="store_true")
    parser.add_argument("--skip-hf-cache", action="store_true")
    args = parser.parse_args()

    if not args.skip_spacy:
        install_spacy_model()

    if not args.skip_hf_cache:
        cache_huggingface_model()

    check_act_pdfs()


def install_spacy_model():
    logger.info(f"installing spacy model {SPACY_MODEL}...")
    result = subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL])
    if result.returncode != 0:
        logger.error(f"spacy model install failed (exit code {result.returncode})")
    else:
        logger.info("spacy model installed")


def cache_huggingface_model():
    logger.info(f"pre-caching {settings.bert_checkpoint} from HuggingFace...")
    try:
        from transformers import AutoTokenizer, AutoModel
        AutoTokenizer.from_pretrained(settings.bert_checkpoint)
        AutoModel.from_pretrained(settings.bert_checkpoint)
        logger.info("InLegalBERT cached locally")
    except Exception as e:
        logger.warning(f"couldn't pre-cache InLegalBERT ({e}) - it'll download on first real use instead")


def check_act_pdfs():
    """tells you what's missing and where to look, rather than guessing a link - see module docstring."""
    missing = []
    for act, full_name in REQUIRED_ACTS.items():
        act_dir = Path("corpus/sources") / act.lower()
        pdfs = list(act_dir.glob("*.pdf")) if act_dir.exists() else []
        if not pdfs:
            missing.append((act, full_name))

    if not missing:
        logger.info("all act PDFs present in corpus/sources/")
        return

    print("\nmissing act PDFs - download these manually from indiacode.nic.in:")
    for act, full_name in missing:
        print(f'  {act:14s} search indiacode.nic.in for "{full_name}"')
        print(f"                 save into corpus/sources/{act.lower()}/")
    print(
        "\nIndiaCode's repository doesn't have a stable per-act URL to script "
        "against reliably - safer to grab these by hand once and verify "
        "you have the current version than to hardcode a link that might "
        "silently point at a stale one.\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
