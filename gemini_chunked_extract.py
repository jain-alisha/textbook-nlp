#!/usr/bin/env python3
"""
Extract paragraphs from large PDFs using Gemini API in chunks.
Splits PDF into smaller temporary files and processes each chunk separately.
"""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import fitz
from dotenv import load_dotenv

from gemini_reader import extract_paragraphs_gemini

load_dotenv()


def extract_pdf_chunk(input_pdf: str, start_page: int, end_page: int, output_pdf: str):
    """Extract a page range from PDF and save to new file."""
    doc = fitz.open(input_pdf)
    chunk_doc = fitz.open()
    
    for page_num in range(start_page, min(end_page, len(doc))):
        chunk_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    
    chunk_doc.save(output_pdf)
    chunk_doc.close()
    doc.close()


def main():
    parser = argparse.ArgumentParser(description="Extract paragraphs from large PDF using Gemini in chunks")
    parser.add_argument("pdf", type=Path, help="Path to textbook PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paragraph_clusters.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Pages per chunk (default: 50)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (overrides GEMINI_API_KEY env)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set", file=sys.stderr)
        return 1

    # Get total pages
    doc = fitz.open(str(args.pdf))
    total_pages = len(doc)
    doc.close()
    
    print(f"Total pages: {total_pages}")
    print(f"Chunk size: {args.chunk_size} pages")
    print(f"Output: {args.output}")
    print()

    # Remove existing output
    if args.output.exists():
        args.output.unlink()

    all_paragraphs = []
    
    # Process in chunks
    with tempfile.TemporaryDirectory() as tmpdir:
        for start_page in range(0, total_pages, args.chunk_size):
            end_page = min(start_page + args.chunk_size, total_pages)
            
            print(f"Processing pages {start_page}-{end_page}...")
            
            # Create temporary chunk PDF
            chunk_pdf = Path(tmpdir) / f"chunk_{start_page}_{end_page}.pdf"
            extract_pdf_chunk(str(args.pdf), start_page, end_page, str(chunk_pdf))
            
            # Extract with Gemini
            try:
                paragraphs = extract_paragraphs_gemini(
                    str(chunk_pdf),
                    api_key=api_key,
                    request_timeout=300
                )
                print(f"  Extracted {len(paragraphs)} paragraphs from chunk")
                all_paragraphs.extend(paragraphs)
            except Exception as e:
                print(f"  Error processing chunk: {e}")
                print(f"  Skipping pages {start_page}-{end_page}")
                continue
            
            print()

    # Write all results
    print(f"Total paragraphs extracted: {len(all_paragraphs)}")
    
    with args.output.open('w', encoding='utf8', newline='') as f:
        w = csv.writer(f)
        w.writerow(["paragraph"])
        for p in all_paragraphs:
            w.writerow([p])
    
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
