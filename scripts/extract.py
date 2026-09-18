#!/usr/bin/env python3
"""
Extract paragraphs from a textbook PDF using Google Gemini.

Processes the PDF in page-range chunks to avoid output token limits.
Falls back to PyMuPDF if no GEMINI_API_KEY is set or Gemini fails.

Usage:
    python scripts/extract.py pdfs/saxon_course1.pdf --name saxon_course1

Environment:
    GEMINI_API_KEY=your_key_here   (in .env)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MIN_PARAGRAPH_LEN = 30
PAGES_PER_CHUNK   = 50   # pages sent to Gemini per call

GEMINI_PROMPT = """You are processing a section of a math textbook PDF. Extract all instructional paragraphs as a JSON array of strings.

Rules:
- Each string = one coherent pedagogical unit: a problem, explanation, worked example, definition, student dialogue, or instruction set
- Keep problem setups and follow-up questions together when they clearly belong (e.g. narrative setup + parts a, b, c)
- Exclude: page numbers, running headers/footers, table of contents entries, index entries, answer keys
- Include: problem text, explanations, worked examples, student dialogues, definitions, margin notes, math notes boxes
- Preserve text faithfully — do not paraphrase
- Minimum chunk length: 30 characters

Return ONLY a valid JSON array of strings. No markdown, no commentary.
Example: ["paragraph one", "paragraph two"]
"""


def extract_gemini_chunked(pdf_path: Path, api_key: str, pages_per_chunk: int = PAGES_PER_CHUNK) -> list[str]:
    try:
        import google.generativeai as genai
        import fitz
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}. Run: pip install google-generativeai pymupdf")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    all_paragraphs: list[str] = []
    chunk_num = 0
    page = 0

    while page < total_pages:
        chunk_num += 1
        page_end = min(page + pages_per_chunk, total_pages)
        print(f"  Chunk {chunk_num}: pages {page+1}–{page_end} of {total_pages}...")

        # Extract page range to a temporary PDF in memory
        src = fitz.open(str(pdf_path))
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(src, from_page=page, to_page=page_end - 1)
        src.close()

        buf = io.BytesIO(chunk_doc.tobytes())
        chunk_doc.close()

        # Upload chunk to Gemini
        upload = genai.upload_file(
            path=buf,
            display_name=f"{pdf_path.stem}_p{page+1}-{page_end}.pdf",
            mime_type="application/pdf",
        )

        try:
            response = model.generate_content(
                [upload, GEMINI_PROMPT],
                request_options={"timeout": 300},
                generation_config={"temperature": 0.0},
            )
            # Check for copyright refusal (finish_reason == 4)
            candidate = response.candidates[0] if response.candidates else None
            if candidate and candidate.finish_reason == 4:
                print(f"    ⚠ Copyright refusal on chunk {chunk_num} — using PyMuPDF for these pages")
                src2 = fitz.open(str(pdf_path))
                for pnum in range(page, page_end):
                    pg = src2[pnum]
                    text = pg.get_text("text")
                    blocks = re.split(r"\n{2,}", text)
                    for block in blocks:
                        lines = block.splitlines()
                        kept = [ln for ln in lines if not _is_skip_line(ln)]
                        para = " ".join(kept).strip()
                        para = re.sub(r"\s+", " ", para)
                        if len(para) >= MIN_PARAGRAPH_LEN:
                            all_paragraphs.append(para)
                src2.close()
                page = page_end
                continue
            raw = (response.text or "").strip()
        finally:
            try:
                genai.delete_file(upload.name)
            except Exception:
                pass

        # Parse response — use a more lenient JSON extraction
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        # Try to salvage truncated JSON by finding the last complete string
        try:
            chunk_paras = json.loads(raw)
            if not isinstance(chunk_paras, list):
                raise ValueError("Not a list")
        except (json.JSONDecodeError, ValueError):
            # Try to recover partial JSON — find last complete quoted string
            matches = re.findall(r'"((?:[^"\\]|\\.)*)"\s*(?:,|\])', raw)
            if matches:
                chunk_paras = matches
                print(f"    ⚠ Partial JSON recovered: {len(chunk_paras)} items")
            else:
                print(f"    ⚠ Parse error on chunk {chunk_num} — skipping")
                print(f"    Raw preview: {raw[:150]}")
                page = page_end
                continue

        good = [str(p).strip() for p in chunk_paras
                if str(p).strip() and len(str(p).strip()) >= MIN_PARAGRAPH_LEN]
        all_paragraphs.extend(good)
        print(f"    → {len(good)} paragraphs extracted")

        page = page_end

        # Brief pause between chunks to avoid rate limits
        if page < total_pages:
            time.sleep(2)

    return all_paragraphs


# ── PyMuPDF fallback ──────────────────────────────────────────────────────────

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
    try:
        import fitz
    except ImportError:
        raise RuntimeError("Run: pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    chunks: list[str] = []
    for page in doc:
        text = page.get_text("text")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract paragraphs from a textbook PDF using Gemini."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file.")
    parser.add_argument("--name", required=True,
                        help="Short identifier (e.g. saxon_course1).")
    parser.add_argument("--fallback", action="store_true",
                        help="Force PyMuPDF fallback even if GEMINI_API_KEY is set.")
    parser.add_argument("--chunk-size", type=int, default=PAGES_PER_CHUNK,
                        help=f"Pages per Gemini chunk (default: {PAGES_PER_CHUNK}).")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: {args.pdf} not found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("data") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paragraphs.csv"

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if api_key and not args.fallback:
        print(f"Using Gemini extraction for {args.pdf.name} ({args.chunk_size} pages/chunk)...")
        try:
            paragraphs = extract_gemini_chunked(args.pdf, api_key, args.chunk_size)
            print(f"\n  Gemini extracted {len(paragraphs)} paragraphs total")
        except Exception as e:
            print(f"\n  Gemini failed: {e}", file=sys.stderr)
            print("  Falling back to PyMuPDF...", file=sys.stderr)
            paragraphs = extract_pymupdf(args.pdf)
            print(f"  PyMuPDF extracted {len(paragraphs)} paragraphs (lower quality)")
    else:
        if not api_key:
            print("No GEMINI_API_KEY — using PyMuPDF fallback.")
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