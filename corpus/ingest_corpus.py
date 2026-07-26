"""
thin CLI wrapper around corpus/ingest.py.

usage:
    uv run python scripts/ingest_corpus.py --act ipc
    uv run python scripts/ingest_corpus.py --act BNS --force
    uv run python scripts/ingest_corpus.py --all
"""

import argparse

from corpus.ingest import ingest_act
from corpus.embeddings import PassageEmbedder
from config.constants import ACTS

ALL_ACTS = [act.lower() for act in ACTS]


def main():
    parser = argparse.ArgumentParser()
    # type=str.lower allows case-insensitive CLI inputs (e.g., --act IPC or --act ipc)
    parser.add_argument("--act", type=str.lower, choices=ALL_ACTS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.act and not args.all:
        parser.error("pass --act <name> or --all")

    acts = ALL_ACTS if args.all else [args.act]

    # load InLegalBERT once, reuse across acts - it's the slow part of the pipeline
    embedder = PassageEmbedder()

    for act in acts:
        ingest_act(act, force=args.force, embedder=embedder)


if __name__ == "__main__":
    main()