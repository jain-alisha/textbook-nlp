#!/usr/bin/env python3
"""
NOT USED — abandoned, kept for reference only.

Rule-based paragraph stitcher — free, instant, no API calls.

Tested on Saxon (2026-09-25) and found to damage Era 4 data: on a 60-paragraph
sample it merged 48% of paragraphs together, fusing separate numbered problems
(3-8) into a single blob and destroying 5 problem-number markers. The
NEW_PROBLEM_RE-style boundary logic this relies on only recognizes CPM's
"1-51." numbering, not Saxon's "7." style, so it over-merges on any non-CPM
book. Decision: skip stitching entirely rather than fix the heuristics — see
README changelog for the full writeup. Nothing in the pipeline calls this
script; do not wire it back in without re-deriving a per-series boundary rule
first.

Merges a paragraph into the next one when it looks like a continuation:
  - ends without terminal punctuation (no . ? ! :)  AND is short (<120 chars)
  - OR starts with a lowercase letter (mid-sentence split)
  - OR starts with a continuation word (and, or, but, then, so, which, that...)
  - OR is a lone label like "a." "b." "1." that belongs with the next chunk

Usage:
    python scripts/stitch_cheap.py --name saxon_course2
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


# ── Heuristics ────────────────────────────────────────────────────────────────

TERMINAL_PUNCT = re.compile(r'[.?!:]$')
CONTINUATION_STARTS = re.compile(
    r'^(and|or|but|then|so|which|that|where|when|if|as|because|however|therefore|'
    r'thus|hence|whereas|while|although|since|unless|until|for|nor)\b',
    re.IGNORECASE
)
LONE_LABEL = re.compile(r'^[\*\•]?\s*[a-zA-Z0-9]{1,2}[.)]\s*$')
SHORT_THRESHOLD = 120


def should_merge(current: str, nxt: str) -> bool:
    cur = current.strip()
    nx  = nxt.strip()

    if not cur or not nx:
        return False

    # Lone label: "a." "b." "1." — belongs with what follows
    if LONE_LABEL.match(cur):
        return True

    # Next chunk starts lowercase → mid-sentence split
    if nx and nx[0].islower():
        return True

    # Next chunk starts with a continuation word
    if CONTINUATION_STARTS.match(nx):
        return True

    # Current ends without terminal punctuation AND is short
    if not TERMINAL_PUNCT.search(cur) and len(cur) < SHORT_THRESHOLD:
        return True

    return False


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_paragraphs(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["paragraph"].strip() for row in reader if row["paragraph"].strip()]


def save_paragraphs(paragraphs: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["paragraph"])
        for p in paragraphs:
            writer.writerow([p])


# ── Main ──────────────────────────────────────────────────────────────────────

def stitch(paragraphs: list[str]) -> list[str]:
    if not paragraphs:
        return []

    result = [paragraphs[0]]

    for nxt in paragraphs[1:]:
        if should_merge(result[-1], nxt):
            result[-1] = result[-1].rstrip() + " " + nxt.lstrip()
        else:
            result.append(nxt)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based paragraph stitcher (free)")
    parser.add_argument("--name", required=True, help="Textbook identifier")
    args = parser.parse_args()

    data_dir   = Path("data") / args.name
    input_path = data_dir / "paragraphs.csv"
    out_path   = data_dir / "paragraphs_stitched.csv"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run extract.py first.")
        return

    print(f"Loading {input_path}...")
    paragraphs = load_paragraphs(input_path)
    print(f"  Loaded {len(paragraphs)} paragraphs")

    stitched = stitch(paragraphs)

    save_paragraphs(stitched, out_path)

    reduction = len(paragraphs) - len(stitched)
    print(f"\nSTITCHING COMPLETE")
    print(f"  Original:  {len(paragraphs)}")
    print(f"  Stitched:  {len(stitched)}")
    print(f"  Merged:    {reduction} ({100*reduction/len(paragraphs):.1f}% reduction)")
    print(f"  Output:    {out_path}")
    print(f"\nNext step:")
    print(f"  python scripts/run_pipeline.py --name {args.name} "
          f"--paragraphs paragraphs_stitched.csv --quota-wait 0")


if __name__ == "__main__":
    main()
