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

# CPM's running page header ("Core Connections Course 2", "Core Connections
# Geometry", ...) fused onto the start of the following paragraph's text on
# alternating pages -- e.g. "Core Connections Course 2 1-51. Janelle wants...".
# Surveyed all five books this applies to (2026-09-25): the header is always this
# exact fixed string, immediately followed by real content in every instance
# checked -- a page number, a section label ("Closure", "Review & Preview"), or
# straight into prose. A strict prefix match is safe here because the string
# itself never overlaps with genuine paragraph content; it is not a heuristic
# guess at where a header "probably" ends.
CPM_HEADER_RE = re.compile(
    r"^\s*Core Connections (?:Course [123]|Algebra [12]|Geometry)\s*"
)


def clean(paragraph: str) -> str:
    text = WATERMARK_RE.sub(" ", paragraph)
    text = CPM_HEADER_RE.sub("", text)
    text = BULLET_RUN_RE.sub(" ", text)
    text = ZERO_WIDTH_SPACE_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_book(name: str, filename: str = "paragraphs.csv") -> None:
    path = Path("data") / name / filename
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
    parser.add_argument("--also-stitched", action="store_true",
                         help="Also clean paragraphs_stitched.csv where present -- "
                              "needed because stitching ran before this header-strip "
                              "existed, so the header text is baked into the merges.")
    args = parser.parse_args()

    for name in args.name:
        clean_book(name)
        if args.also_stitched:
            stitched = Path("data") / name / "paragraphs_stitched.csv"
            if stitched.exists():
                clean_book(name, "paragraphs_stitched.csv")


if __name__ == "__main__":
    main()
