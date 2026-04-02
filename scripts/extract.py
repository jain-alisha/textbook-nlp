#!/usr/bin/env python3
"""
Extract paragraphs from a textbook PDF using Google Gemini.

Gemini reads the PDF visually and understands layout, producing semantically
coherent paragraph chunks. Falls back to PyMuPDF if no GEMINI_API_KEY is set
or if Gemini fails.

Usage:
    python scripts/extract.py pdfs/cpm_algebra2.pdf --name cpm_algebra2

Environment:
    GEMINI_API_KEY=your_key_here   (in .env)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MIN_PARAGRAPH_LEN = 30

GEMINI_PROMPT = """You are processing a math textbook PDF. Extract all instructional paragraphs
from this document as a JSON array of strings.

Rules:
- Each string should be one coherent pedagogical unit: a problem, an explanation, a worked
  example, a definition, a student dialogue, or an instruction set
- Keep problem setups and their follow-up questions together in one chunk where they clearly
  belong together (e.g. a narrative setup followed by parts a, b, c)
- Exclude: page numbers, running headers/footers, table of contents entries, index entries,
  answer keys at the back of the book
- Include: problem text, explanations, worked examples, student dialogues, definitions,
  margin notes, math notes boxes, learning log prompts
- Preserve the actual text faithfully, do not paraphrase or summarize
- Minimum chunk length: 30 characters

Return ONLY a valid JSON array of strings. No markdown, no commentary, no preamble.
Example format: ["paragraph one text", "paragraph two text", ...]
"""


def extract_gemini(pdf_path: Path, api_key: str) -> list[str]:
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("Run: pip install google-generativeai")

    genai.configure(api_key=api_key)

    print(f"  Uploading {pdf_path.name} to Gemini...")
    upload = genai.upload_file(path=str(pdf_path), display_name=pdf_path.name)

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        print("  Extracting paragraphs (this may take 1-3 minutes)...")
        response = model.generate_content(
            [upload, GEMINI_PROMPT],
            request_options={"timeout": 600},
            generation_config={"temperature": 0.0},
        )
        raw = (response.text or "").strip()
    finally:
        try:
            genai.delete_file(upload.name)
        except Exception:
            pass

    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    try:
        paragraphs = json.loads(raw)
        if not isinstance(paragraphs, list):
            raise ValueError("Response is not a JSON array")
        return [
            str(p).strip() for p in paragraphs
            if str(p).strip() and len(str(p).strip()) >= MIN_PARAGRAPH_LEN
        ]
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Failed to parse Gemini response: {e}\nRaw: {raw[:500]}")


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


def extract_pymupdf(pdf_path: Path) -> list[str]:
    """Fallback: PyMuPDF with double-newline splitting only (no single-newline)."""
    try:
        import fitz
    except ImportError:
        raise RuntimeError("Run: pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    chunks: list[str] = []

    for page in doc:
        text = page.get_text("text")
        # Split on double newlines only — keeps pages whole if needed,
        # but avoids the 9000-chunk problem from single-newline splitting
        blocks = re.split(r"\n{2,}", text)
        for block in blocks:
            lines = block.splitlines()
            kept = [ln for ln in lines if not _is_skip_line(ln)]
            para = " ".join(kept).strip()
            para = re.sub(r"\s+", " ", para)
            if len(para) >= MIN_PARAGRAPH_LEN:
                chunks.append(para)

    doc.close()
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract paragraphs from a textbook PDF using Gemini."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file.")
    parser.add_argument("--name", required=True,
                        help="Short identifier (e.g. cpm_algebra2).")
    parser.add_argument("--fallback", action="store_true",
                        help="Force PyMuPDF fallback even if GEMINI_API_KEY is set.")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: {args.pdf} not found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("data") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paragraphs.csv"

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if api_key and not args.fallback:
        print(f"Using Gemini extraction for {args.pdf.name}...")
        try:
            paragraphs = extract_gemini(args.pdf, api_key)
            print(f"  Gemini extracted {len(paragraphs)} paragraphs")
        except Exception as e:
            print(f"  Gemini failed: {e}", file=sys.stderr)
            print("  Falling back to PyMuPDF...", file=sys.stderr)
            paragraphs = extract_pymupdf(args.pdf)
            print(f"  PyMuPDF extracted {len(paragraphs)} paragraphs (lower quality)")
    else:
        if not api_key:
            print("No GEMINI_API_KEY found — using PyMuPDF fallback.")
            print("Add GEMINI_API_KEY to .env for better extraction.")
        paragraphs = extract_pymupdf(args.pdf)
        print(f"PyMuPDF extracted {len(paragraphs)} paragraphs")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paragraph"])
        for para in paragraphs:
            writer.writerow([para])

    print(f"\nSaved {len(paragraphs)} paragraphs -> {out_path}")
    print(f"\nNext step:")
    print(f"  python scripts/stitch.py --name {args.name} --start-fresh")


if __name__ == "__main__":
    main()