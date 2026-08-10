#!/usr/bin/env python3
"""Validate Git commit subjects against the Conventional Commits header."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


HEADER = re.compile(
    r"^[a-z][a-z0-9-]*(\([a-z0-9][a-z0-9._/-]*\))?!?: [^\s].+$"
)


def subjects_for_range(commit_range: str) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--format=%s", commit_range],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--message-file", type=Path)
    group.add_argument("--range", dest="commit_range")
    args = parser.parse_args()

    if args.message_file:
        text = args.message_file.read_text(encoding="utf-8")
        subjects = [text.splitlines()[0]] if text.splitlines() else []
    else:
        subjects = subjects_for_range(args.commit_range)

    invalid = [subject for subject in subjects if not HEADER.fullmatch(subject)]
    if invalid:
        print("Invalid Conventional Commit subject(s):", file=sys.stderr)
        for subject in invalid:
            print(f"  - {subject}", file=sys.stderr)
        print(
            "Expected: type(scope): description (scope is optional).",
            file=sys.stderr,
        )
        return 1

    print(f"Validated {len(subjects)} Conventional Commit subject(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
