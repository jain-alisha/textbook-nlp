#!/usr/bin/env python3
"""
Extract paragraphs from a textbook PDF using Google Gemini.

Processes the PDF in page-range chunks, several at a time. A chunk Gemini can't
handle (timeout, truncated output, refusal) is retried, then split in half, and
only as a last resort extracted with PyMuPDF — for those pages alone. If too much
of the book ends up on PyMuPDF the run aborts rather than overwrite good data.

Usage:
    python scripts/extract.py pdfs/saxon_course1_ms.pdf --name saxon_course1_ms

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MIN_PARAGRAPH_LEN = 30
PAGES_PER_CHUNK   = 50
GEMINI_MODEL      = "gemini-2.5-flash"
MAX_ATTEMPTS      = 3
MIN_SPLIT_PAGES   = 6
DEFAULT_WORKERS   = 3
MAX_FALLBACK_SHARE = 0.2

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


# ── Gemini ────────────────────────────────────────────────────────────────────

class DailyQuotaExhausted(Exception):
    """Retrying or splitting can't help; every further call fails until reset."""


def _parse(raw: str | None) -> list[str] | None:
    raw = re.sub(r"```(?:json)?", "", raw or "").replace("```", "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [s for s in (str(p).strip() for p in data) if len(s) >= MIN_PARAGRAPH_LEN]


def _gemini_chunk(client, data: bytes, label: str) -> tuple[str, list[str]]:
    """Returns ("ok", paragraphs), or a failure status with no paragraphs."""
    from google.genai import types

    # Thinking stays at the model default despite its cost: grouping a problem's
    # setup with its parts and dropping headers/footers depend on it. With
    # thinking_budget=0 (or even 1024) output fragmented ~2.5x and kept the
    # per-page copyright footer.
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=list[str],
    )
    upload = None
    status = "failed"
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                if upload is None:
                    upload = client.files.upload(
                        file=io.BytesIO(data),
                        config=types.UploadFileConfig(mime_type="application/pdf",
                                                      display_name=f"{label}.pdf"),
                    )
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=[upload, GEMINI_PROMPT], config=config)
            except Exception as e:
                if "PerDay" in str(e):
                    raise DailyQuotaExhausted(str(e)) from e
                print(f"    {label}: attempt {attempt}/{MAX_ATTEMPTS} failed: {str(e)[:140]}")
                time.sleep(15 * attempt)
                continue

            cand = resp.candidates[0] if resp.candidates else None
            reason = cand.finish_reason if cand else None
            if reason == types.FinishReason.MAX_TOKENS:
                return "too_long", []
            paras = _parse(resp.text)
            if paras is not None:
                return "ok", paras
            if reason not in (None, types.FinishReason.STOP):
                return "blocked", []   # e.g. RECITATION on copyrighted text
            print(f"    {label}: attempt {attempt}/{MAX_ATTEMPTS} returned unparseable output")
            status = "unparseable"
        return status, []
    finally:
        if upload is not None:
            try:
                client.files.delete(name=upload.name)
            except Exception:
                pass


def _chunk_bytes(doc, start: int, end: int) -> bytes:
    import fitz
    sub = fitz.open()
    sub.insert_pdf(doc, from_page=start, to_page=end - 1)
    data = sub.tobytes()
    sub.close()
    return data


def extract_gemini_chunked(pdf_path: Path, api_key: str, pages_per_chunk: int,
                           workers: int) -> tuple[list[tuple[str, str]], list[tuple[int, int]], int]:
    """Returns ([(paragraph, extractor)] in page order, fallback ranges, total pages).

    Each paragraph carries the extractor that produced it so a mixed book's
    provenance is recorded in the data rather than only in the run log.
    """
    import fitz
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key,
                          http_options=types.HttpOptions(timeout=600_000))
    # PyMuPDF isn't thread-safe, so the document is only touched on this thread;
    # workers receive ready-made chunk bytes and do network I/O only.
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    pending = [(s, min(s + pages_per_chunk, total_pages))
               for s in range(0, total_pages, pages_per_chunk)]
    done: dict[int, tuple[int, list[str]]] = {}
    fallback: list[tuple[int, int]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending:
            jobs = [(s, e, _chunk_bytes(doc, s, e)) for s, e in pending]
            pending = []
            labels = [f"{pdf_path.stem}_p{s+1}-{e}" for s, e, _ in jobs]
            for s, e, _ in jobs:
                print(f"  Chunk pages {s+1}–{e} of {total_pages}...")
            outcomes = pool.map(lambda j, lbl: _gemini_chunk(client, j[2], lbl), jobs, labels)

            for (s, e, _), (status, paras) in zip(jobs, outcomes):
                if status == "ok":
                    done[s] = (e, [(p, "gemini") for p in paras])
                    print(f"    pages {s+1}–{e}: {len(paras)} paragraphs")
                elif e - s > MIN_SPLIT_PAGES:
                    mid = (s + e) // 2
                    print(f"    pages {s+1}–{e}: {status} — splitting in half")
                    pending += [(s, mid), (mid, e)]
                else:
                    print(f"    pages {s+1}–{e}: {status} — using PyMuPDF for these pages")
                    done[s] = (e, [(p, "pymupdf") for p in _pymupdf_pages(doc, s, e)])
                    fallback.append((s, e))

    doc.close()
    paragraphs = [p for s in sorted(done) for p in done[s][1]]
    return paragraphs, sorted(fallback), total_pages


# ── PyMuPDF ───────────────────────────────────────────────────────────────────

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


def _pymupdf_pages(doc, start: int, end: int) -> list[str]:
    out: list[str] = []
    for pnum in range(start, end):
        for block in re.split(r"\n{2,}", doc[pnum].get_text("text")):
            kept = [ln for ln in block.splitlines() if not _SKIP_LINE_RE.match(ln)]
            para = re.sub(r"\s+", " ", " ".join(kept)).strip()
            if len(para) >= MIN_PARAGRAPH_LEN:
                out.append(para)
    return out


def extract_pymupdf(pdf_path: Path) -> list[str]:
    import fitz
    doc = fitz.open(str(pdf_path))
    paras = _pymupdf_pages(doc, 0, len(doc))
    doc.close()
    return paras


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract paragraphs from a textbook PDF using Gemini."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file.")
    parser.add_argument("--name", required=True,
                        help="Short identifier (e.g. saxon_course1_ms).")
    parser.add_argument("--fallback", action="store_true",
                        help="Force PyMuPDF for the whole book even if GEMINI_API_KEY is set.")
    parser.add_argument("--chunk-size", type=int, default=PAGES_PER_CHUNK,
                        help=f"Pages per Gemini chunk (default: {PAGES_PER_CHUNK}).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Chunks sent to Gemini concurrently (default: {DEFAULT_WORKERS}).")
    parser.add_argument("--allow-fallback", action="store_true",
                        help=f"Save even if more than {MAX_FALLBACK_SHARE:.0%} of pages "
                             "had to use PyMuPDF.")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: {args.pdf} not found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("data") / args.name
    out_path = out_dir / "paragraphs.csv"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if api_key and not args.fallback:
        print(f"Using Gemini extraction for {args.pdf.name} "
              f"({args.chunk_size} pages/chunk, {args.workers} workers)...")
        try:
            paragraphs, fallback, total_pages = extract_gemini_chunked(
                args.pdf, api_key, args.chunk_size, args.workers)
        except DailyQuotaExhausted:
            print(f"\nERROR: Gemini daily request quota exhausted; nothing saved. "
                  f"Re-run after the quota resets.", file=sys.stderr)
            sys.exit(3)
        fb_pages = sum(e - s for s, e in fallback)
        share = fb_pages / total_pages if total_pages else 0.0
        print(f"\n  {len(paragraphs)} paragraphs; {fb_pages}/{total_pages} pages "
              f"({share:.0%}) fell back to PyMuPDF")
        for s, e in fallback:
            print(f"    fallback: pages {s+1}–{e}")
        if share > MAX_FALLBACK_SHARE and not args.allow_fallback:
            print(f"\nERROR: more than {MAX_FALLBACK_SHARE:.0%} of pages fell back to PyMuPDF; "
                  f"not overwriting {out_path}. Re-run later, or pass --allow-fallback.",
                  file=sys.stderr)
            sys.exit(2)
    else:
        if not api_key:
            print("No GEMINI_API_KEY — using PyMuPDF fallback.")
        paragraphs = [(p, "pymupdf") for p in extract_pymupdf(args.pdf)]
        print(f"PyMuPDF extracted {len(paragraphs)} paragraphs")

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paragraph", "source"])
        for para, source in paragraphs:
            writer.writerow([para, source])

    print(f"\nSaved {len(paragraphs)} paragraphs -> {out_path}")
    print(f"\nNext step:")
    print(f"  python scripts/stitch.py --name {args.name} --start-fresh")


if __name__ == "__main__":
    main()
