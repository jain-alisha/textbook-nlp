#!/usr/bin/env python3
"""
Gemini PDF Reader - Extract paragraphs from textbook PDFs.

Usage:
    python gemini_reader.py <pdf_path> --output paragraph_clusters.csv

Reads PDF via Gemini API (if GEMINI_ENABLED=1) or falls back to PyMuPDF extraction.
Saves cleaned paragraph chunks to CSV for reuse in classification.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import List

import fitz
import numpy as np
import spacy
from dotenv import load_dotenv
from sklearn.cluster import KMeans

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

load_dotenv()

DEFAULT_READER_MODEL = "models/gemini-flash-latest"
DEFAULT_PROMPT = (
    "Extract the main instructional paragraphs from this PDF. "
    "Return a JSON array of strings where each string is a cleaned paragraph. "
    "Do not add commentary outside of the JSON array."
)


# ============================================================================
# Gemini-based extraction
# ============================================================================

def extract_paragraphs_gemini(
    pdf_path: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    prompt: str | None = None,
    request_timeout: int = 600,
) -> List[str]:
    """Extract paragraphs from a PDF using Google Gemini."""
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Install google-generativeai to use Gemini extraction.")

    key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not key:
        raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY or pass api_key.")

    genai.configure(api_key=key)

    model_name = model or os.environ.get("GEMINI_READER_MODEL", DEFAULT_READER_MODEL)
    prompt_text = prompt or os.environ.get("GEMINI_READER_PROMPT", DEFAULT_PROMPT)

    upload = genai.upload_file(path=str(pdf_path), display_name=Path(pdf_path).name)
    try:
        reader = genai.GenerativeModel(model_name)
        response = reader.generate_content(
            [upload, prompt_text],
            request_options={"timeout": request_timeout},
            generation_config={"temperature": 0.0},
        )
    finally:
        try:
            genai.delete_file(upload.name)
        except Exception:
            pass

    raw_text = _response_text(response).strip()
    paragraphs = _parse_paragraphs(raw_text)
    if not paragraphs:
        raise RuntimeError("Gemini extraction returned no paragraphs.")
    return paragraphs


def _response_text(response) -> str:
    if hasattr(response, "text") and response.text:
        return response.text

    parts: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _parse_paragraphs(blob: str) -> List[str]:
    if not blob:
        return []

    start = blob.find("[")
    end = blob.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = blob[start : end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                collected: List[str] = []
                for item in data:
                    if isinstance(item, str):
                        text = item.strip()
                    elif isinstance(item, dict):
                        text = str(item.get("text", "")).strip()
                    else:
                        text = str(item).strip()
                    if text:
                        collected.append(text)
                if collected:
                    return collected
        except json.JSONDecodeError:
            pass

    paragraphs = [p.strip() for p in blob.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs

    return [line.strip() for line in blob.splitlines() if line.strip()]


# ============================================================================
# PyMuPDF fallback extraction
# ============================================================================

nlp = spacy.load('en_core_web_sm')


def cluster_columns(blocks, n_columns=2):
    """Clusters blocks into columns based on x-center using k-means."""
    if len(blocks) < n_columns:
        return [blocks]
    x_centers = np.array([[(b['bbox'][0] + b['bbox'][2]) / 2] for b in blocks])
    kmeans = KMeans(n_clusters=n_columns, n_init=10, random_state=42)
    col_labels = kmeans.fit_predict(x_centers)
    columns = {i: [] for i in range(n_columns)}
    for b, col in zip(blocks, col_labels):
        columns[col].append(b)
    sorted_cols = sorted(columns.items(), key=lambda x: np.mean([(b['bbox'][0] + b['bbox'][2]) / 2 for b in x[1]]) if x[1] else float('inf'))
    return [blocks for _, blocks in sorted_cols]


def extract_paragraphs_pymupdf_range(
    pdf_path: str,
    start_page: int = 0,
    end_page: int | None = None,
    page_frac: float = 1.0,
    gap_thresh: int = 15,
    n_columns: int = 2
) -> List[str]:
    """Extract paragraphs using PyMuPDF with column clustering from a page range."""
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    
    if end_page is None:
        end_page = n_pages
    end_page = min(end_page, n_pages)
    start_page = max(0, start_page)
    
    sample_n = max(1, int((end_page - start_page) * page_frac))
    pages = list(range(start_page, end_page))[:sample_n]

    paras = []
    for i in pages:
        page = doc[i]
        blocks = [b for b in page.get_text("dict", sort=True)["blocks"] if b["type"] == 0]
        if not blocks:
            continue
        col_blocks = cluster_columns(blocks, n_columns=n_columns)
        for blocks_in_col in col_blocks:
            blocks_in_col.sort(key=lambda b: b["bbox"][1])
            cluster = []
            prev_bot = None
            for b in blocks_in_col:
                txt_chunks = []
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        if "text" in span:
                            txt_chunks.append(span["text"])
                txt = " ".join(txt_chunks)
                txt = re.sub(r'[\n\r]+', ' ', txt).strip()
                txt = re.sub(r'\s+', ' ', txt)
                if len(txt) < 30 or txt.isupper() or txt.endswith(':'):
                    continue
                if txt.count('.') < 1 or re.fullmatch(r'[\d\W_]+', txt):
                    continue
                y0, y1 = b['bbox'][1], b['bbox'][3]
                if prev_bot is None or (y0 - prev_bot) <= gap_thresh:
                    cluster.append(txt)
                else:
                    if cluster:
                        paras.append(" ".join(cluster))
                    cluster = [txt]
                prev_bot = y1
            if cluster:
                paras.append(" ".join(cluster))

    cleaned = []
    for p in paras:
        doc_spacy = nlp(p)
        full = " ".join(sent.text.strip() for sent in doc_spacy.sents if len(sent.text.strip()) > 20)
        if len(full) > 60:
            cleaned.append(re.sub(r'\s+', ' ', full))
    return cleaned


def extract_paragraphs_pymupdf(
    pdf_path: str,
    page_frac: float = 1.0,
    gap_thresh: int = 15,
    n_columns: int = 2
) -> List[str]:
    """Extract paragraphs using PyMuPDF with column clustering (backward compatibility)."""
    return extract_paragraphs_pymupdf_range(
        pdf_path, 
        start_page=0, 
        end_page=None, 
        page_frac=page_frac,
        gap_thresh=gap_thresh,
        n_columns=n_columns
    )


# ============================================================================
# Main pipeline
# ============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract paragraphs from PDF")
    parser.add_argument("pdf", type=Path, help="Path to textbook PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paragraph_clusters.csv"),
        help="Output CSV path (default: paragraph_clusters.csv)"
    )
    parser.add_argument(
        "--page-fraction",
        type=float,
        default=float(os.environ.get("PAGE_SAMPLE_FRAC", "1.0")),
        help="Fraction of pages to process (0-1, default: 1.0)"
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Force Gemini extraction (default: auto via GEMINI_ENABLED env)"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="Starting page number (0-indexed, default: 0)"
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Ending page number (exclusive, default: all pages)"
    )
    args = parser.parse_args(argv)

    paragraphs = []
    use_gemini = args.use_gemini or os.environ.get('GEMINI_ENABLED', '').lower() in ('1', 'true', 'yes')

    if use_gemini:
        try:
            print('Attempting Gemini PDF extraction...')
            paragraphs = extract_paragraphs_gemini(
                str(args.pdf),
                api_key=os.environ.get('GEMINI_API_KEY')
            )
            print(f'Gemini returned {len(paragraphs)} paragraphs.')
        except Exception as e:
            print(f'Gemini extraction failed: {e}')
            print('Falling back to local PyMuPDF extraction...')

    if not paragraphs:
        print(f'Using PyMuPDF extraction (pages {args.start_page}-{args.end_page or "end"})...')
        paragraphs = extract_paragraphs_pymupdf_range(
            str(args.pdf),
            start_page=args.start_page,
            end_page=args.end_page,
            page_frac=args.page_fraction
        )

    print(f"Extracted {len(paragraphs)} paragraph clusters.")

    mode = 'a' if args.append else 'w'
    write_header = not (args.append and args.output.exists())
    
    with args.output.open(mode, encoding='utf8', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["paragraph"])
        for p in paragraphs:
            w.writerow([p])

    print(f"{'Appended' if args.append else 'Saved'} paragraphs to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
