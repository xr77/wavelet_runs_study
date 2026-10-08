"""Reject notebooks, data, results, and local backups from tracked files or a ZIP."""

import argparse
import subprocess
import zipfile
from pathlib import PurePosixPath

ALLOWED_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".cff"}
ALLOWED_NAMES = {"LICENSE", ".gitignore", ".gitattributes", "requirements.txt", "MANIFEST.in"}
FORBIDDEN_PARTS = {"data", "outputs", "results", ".local_archive", ".venv", "__pycache__"}


def rejected_paths(paths):
    rejected = []
    for name in paths:
        path = PurePosixPath(name)
        if name.endswith("/"):
            continue
        if set(path.parts) & FORBIDDEN_PARTS:
            rejected.append(name)
        elif path.suffix not in ALLOWED_SUFFIXES and path.name not in ALLOWED_NAMES:
            rejected.append(name)
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive")
    args = parser.parse_args()
    if args.archive:
        with zipfile.ZipFile(args.archive) as archive:
            paths = archive.namelist()
    else:
        paths = subprocess.check_output(["git", "ls-files", "-z"], text=True).split("\0")
        paths = [name for name in paths if name]
    rejected = rejected_paths(paths)
    if rejected:
        parser.exit(1, "Forbidden release files:\n" + "\n".join(rejected) + "\n")
    print(f"Release check passed: {len(paths)} entries; source and documentation only.")


if __name__ == "__main__":
    main()
