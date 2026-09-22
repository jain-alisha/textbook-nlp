#!/usr/bin/env python3
"""
Strip page-footer watermark boilerplate and decorative bullet-divider noise
from an already-extracted paragraphs.csv, in place.

Usage:
    python scripts/clean_paragraphs.py --name cpm_course1_ms cpm_course2_ms ...
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

MIN_PARAGRAPH_LEN = 30

# Matches the CPM page-footer watermark (both the normal and the
# missing-space "forcommercial" variant seen in the extracted text).
WATERMARK_RE = re.compile(
    r"©\s*CPM Educational Program.{0,400}?Do not share\.?",
    re.DOTALL,
)

# Runs of 4+ decorative bullet/divider dots between section headers.
BULLET_RUN_RE = re.compile(r"(?:[•·]\s*){4,}")

ZERO_WIDTH_SPACE_RE = re.compile("​")

WHITESPACE_RE = re.compile(r"\s+")


def clean(paragraph: str) -> str:
    text = WATERMARK_RE.sub(" ", paragraph)
    text = BULLET_RUN_RE.sub(" ", text)
    text = ZERO_WIDTH_SPACE_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_book(name: str) -> None:
    path = Path("data") / name / "paragraphs.csv"
    if not path.exists():
        print(f"  ✗ {name}: {path} not found, skipping")
        return

    with path.open(newline="", encoding="utf-8") as f:
        rows = [(r["paragraph"], r.get("source", "")) for r in csv.DictReader(f)]

    seen: set[str] = set()
    cleaned: list[tuple[str, str]] = []
    dropped_short = 0
    dropped_dupe = 0

    for raw, source in rows:
        text = clean(raw)
        if len(text) < MIN_PARAGRAPH_LEN:
            dropped_short += 1
            continue
        if text in seen:
            dropped_dupe += 1
            continue
        seen.add(text)
        cleaned.append((text, source))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paragraph", "source"])
        for text, source in cleaned:
            writer.writerow([text, source])

    print(
        f"  ✓ {name}: {len(rows)} -> {len(cleaned)} paragraphs "
        f"(dropped {dropped_short} too-short, {dropped_dupe} duplicate)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", nargs="+", required=True,
                         help="One or more textbook identifiers under data/.")
    args = parser.parse_args()

    for name in args.name:
        clean_book(name)


if __name__ == "__main__":
    main()
