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
_MAX_INCOMPLETE_LEN = 300  # only merge if the incomplete chunk is short


def _ends_complete(text: str) -> bool:
    """Return True if text ends with terminal punctuation."""
    t = text.rstrip()
    return bool(t) and t[-1] in _TERMINAL_PUNCT


def _punctuation_merge(chunks: list[str]) -> list[str]:
    """
    Pass 1: merge consecutive chunks where the previous chunk ends mid-sentence
    AND is short enough to be a genuine truncation (not a TOC entry or label).
    The length cap prevents chaining TOC entries and section headers together.
    """
    if not chunks:
        return chunks
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        tail = merged[-1]
        if not _ends_complete(tail) and len(tail) <= _MAX_INCOMPLETE_LEN:
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

    # Pass 1: punctuation-based merge (free, no API calls needed)
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
