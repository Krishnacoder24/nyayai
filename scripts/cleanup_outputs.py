"""
deletes uploaded PDFs and generated outputs (annotated PDF, JSON report,
HTML report) older than settings.output_retention_days. meant to run on a
schedule (cron, systemd timer, etc.) - there's no Celery beat schedule set
up in this project yet, so a plain script invoked periodically is the
simpler option (see services/storage.py for the file layout this reads).

each file's own last-modified time decides whether it's deleted - files
belonging to the same job_id are written within moments of each other by
services/analysis.py, so there's no need to group them by job_id first.

usage:
    uv run python scripts/cleanup_outputs.py              # deletes old files
    uv run python scripts/cleanup_outputs.py --dry-run     # lists what
                                                            # would be deleted,
                                                            # deletes nothing
    uv run python scripts/cleanup_outputs.py --retention-days 7   # override
                                                                   # the configured window
"""

import argparse
import time
from pathlib import Path

from config.settings import settings

SECONDS_PER_DAY = 86400


def find_stale_files(directory: Path, retention_days: int) -> list[Path]:
    """files in `directory` (non-recursive) whose mtime is older than the
    retention window. missing directory -> empty list, not an error, since
    a fresh checkout won't have data/uploads or data/outputs yet."""
    if not directory.exists():
        return []

    cutoff = time.time() - (retention_days * SECONDS_PER_DAY)
    return [
        path for path in directory.iterdir()
        if path.is_file() and path.stat().st_mtime < cutoff
    ]


def cleanup(retention_days: int, dry_run: bool) -> None:
    targets = [Path(settings.uploads_dir), Path(settings.outputs_dir)]

    stale_files = []
    for directory in targets:
        stale_files.extend(find_stale_files(directory, retention_days))

    if not stale_files:
        print(f"no files older than {retention_days} day(s) found.")
        return

    total_bytes = 0
    for path in stale_files:
        size = path.stat().st_size
        total_bytes += size
        action = "[dry-run] would delete" if dry_run else "deleting"
        print(f"{action}: {path} ({size / 1024:.1f} KB)")
        if not dry_run:
            path.unlink()

    verb = "would free" if dry_run else "freed"
    print(f"\n{len(stale_files)} file(s), {total_bytes / (1024 * 1024):.2f} MB {verb}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list files that would be deleted without actually deleting them",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=settings.output_retention_days,
        help=f"delete files older than this many days (default: {settings.output_retention_days}, "
             "from OUTPUT_RETENTION_DAYS / config/settings.py)",
    )
    args = parser.parse_args()

    cleanup(retention_days=args.retention_days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()