#!/usr/bin/env python3
"""
release_prep.py

Prepares the repository for release by removing development-only
directories/files that should not be shipped in a release package.

Removed items (relative to the repository root, i.e. the directory
containing this script):
    - tests/
    - testapp/
    - scripts/
    - val_cfg/
    - ci/
    - dist/
    - CONTRIBUTION.md
    - examples_gui/
    - THIRD_PARTY_LICENSES.md

Usage:
    python release_prep.py          # perform the removal
    python release_prep.py --dry-run  # show what would be removed, without deleting anything
"""

import argparse
import shutil
import sys
from pathlib import Path

# Items to remove, relative to the repository root.
ITEMS_TO_REMOVE = [
    "tests",
    "testapp",
    "scripts",
    "val_cfg",
    "ci",
    "dist",
    "CONTRIBUTION.md",
    "examples_gui",
    "THIRD_PARTY_LICENSES.md"
]


def remove_path(path: Path, dry_run: bool) -> None:
    """Remove a file or directory, respecting dry_run mode."""
    if not path.exists():
        print(f"[skip]    {path} (does not exist)")
        return

    if dry_run:
        kind = "directory" if path.is_dir() else "file"
        print(f"[dry-run] would remove {kind}: {path}")
        return

    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            print(f"[removed] directory: {path}")
        else:
            path.unlink()
            print(f"[removed] file: {path}")
    except OSError as exc:
        print(f"[error]   failed to remove {path}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove development-only files/directories in preparation for a release."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be removed, without actually deleting anything.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    print(f"Repository root: {repo_root}")
    if args.dry_run:
        print("Running in DRY-RUN mode. No files will be deleted.\n")
    else:
        print("Running in LIVE mode. The following items will be permanently deleted.\n")

    for item in ITEMS_TO_REMOVE:
        remove_path(repo_root / item, args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
