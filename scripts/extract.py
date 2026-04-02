"""
Extract paragraphs from a textbook PDF using PyMuPDF.

Usage:
    python scripts/extract.py pdfs/saxon_course1.pdf --name saxon_course1
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


MIN_PARAGRAPH_LEN = 20

# Patterns that indicate a line is a header or page number rather than content.
_SKIP_LINE_RE = re.compile(
    r"""
    ^\s*\d+\s*$                      # bare page number
    | ^\s*page\s+\d+\s*$             # "Page 42"
    | ^\s*chapter\s+\d+              # "Chapter 3"
    | ^\s*section\s+\d+[\.\d]*\s*$   # "Section 1.2"
    | ^\s*lesson\s+\d+               # "Lesson 14"
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_skip_line(line: str) -> bool:
    return bool(_SKIP_LINE_RE.match(line))


_TERMINAL_PUNCT = frozenset('.?!"\')')

# Matches structural artifacts: TOC entries, answer keys, lesson reference tables
_STRUCTURAL_RE = re.compile(
    r'\d+\.\d+\.\d+'       # lesson codes like 9.1.1
    r'|Problems\s+\d'      # "Problems 4-22"
    r'|Lesson\s+\d'        # "Lesson 8"
    r'|Section\s+\d'       # "Section 6"
    r'|\bMN:\s*\d'         # "MN: 8.1.2"
)


def _ends_complete(text: str) -> bool:
    """Return True if text ends with terminal punctuation."""
    t = text.rstrip()
    return bool(t) and t[-1] in _TERMINAL_PUNCT


def _is_structural(text: str) -> bool:
    """Return True if chunk looks like a TOC entry, answer key, or reference
    table rather than prose — these should never trigger a merge."""
    if _STRUCTURAL_RE.search(text):
        return True
    # ends with a bare digit (page number, answer code, section number)
    t = text.rstrip()
    return bool(t) and t[-1].isdigit()


def _punctuation_merge(chunks: list[str]) -> list[str]:
    """
    Pass 1 — free, no API calls.
    Merge consecutive chunks where the previous chunk ends mid-sentence,
    but skip structural artifacts (TOC entries, answer keys, reference tables)
    which legitimately end without terminal punctuation.
    """
    if not chunks:
        return chunks
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        tail = merged[-1]
        if not _ends_complete(tail) and not _is_structural(tail):
            merged[-1] = tail.rstrip() + " " + chunk.lstrip()
        else:
            merged.append(chunk)
    return merged


def extract_paragraphs(pdf_path: Path) -> list[str]:
    doc = fitz.open(str(pdf_path))
    raw_chunks: list[str] = []

    for page in doc:
        text = page.get_text("text")
        # Split on blank lines to get natural paragraph breaks.
        blocks = re.split(r"\n{2,}", text)
        for block in blocks:
            # Collapse internal newlines into spaces.
            lines = block.splitlines()
            kept = [ln for ln in lines if not _is_skip_line(ln)]
            para = " ".join(kept).strip()
            # Normalise whitespace.
            para = re.sub(r"\s+", " ", para)
            if len(para) >= MIN_PARAGRAPH_LEN:
                raw_chunks.append(para)

    doc.close()

    # Pass 1: punctuation-based merge (free, catches ~13% of truncations)
    merged = _punctuation_merge(raw_chunks)

    # Re-apply minimum length filter after merging
    merged = [p for p in merged if len(p) >= MIN_PARAGRAPH_LEN]

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paragraphs from a PDF.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file.")
    parser.add_argument(
        "--name",
        required=True,
        help="Short identifier for this textbook (e.g. saxon_course1).",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: {args.pdf} not found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("data") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paragraphs.csv"

    paragraphs = extract_paragraphs(args.pdf)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paragraph"])
        for para in paragraphs:
            writer.writerow([para])

    print(f"Extracted {len(paragraphs)} paragraphs -> {out_path}")


if __name__ == "__main__":
    main()
