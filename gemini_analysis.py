#!/usr/bin/env python3
"""
Gemini Textbook Analysis - Classify textbook paragraphs using Gemini.

Usage:
    # From pre-extracted CSV:
    python gemini_analysis.py paragraph_clusters.csv --output analysis_results.csv

    # Or directly from PDF (will extract first):
    python gemini_analysis.py textbook.pdf --output analysis_results.csv

Classifies paragraphs into categories defined in seed_phrases.csv using Gemini API.
Output format: [paragraph, category]
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions

# Import reader for PDF fallback
from gemini_reader import extract_paragraphs_gemini, extract_paragraphs_pymupdf

load_dotenv()

DEFAULT_CLASSIFIER_MODEL = "models/gemini-flash-latest"
DEFAULT_READER_MODEL = "models/gemini-flash-latest"
DEFAULT_SAMPLE_FRACTION = 0.4
DEFAULT_EXAMPLES_PER_CATEGORY = 4
DEFAULT_SLEEP = 0.75

STOPWORDS = {
    "the", "and", "with", "that", "from", "this", "your", "into",
    "about", "above", "their", "which", "should", "different",
    "correct", "wrong", "right", "show", "explain", "steps",
    "solution", "rewrite", "answer",
}


def read_seed_phrases(path: Path) -> Dict[str, List[str]]:
    """Load seed phrases grouped by category from CSV."""
    buckets: Dict[str, List[str]] = {}
    with path.open(encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phrase = (row.get("phrase") or "").strip()
            category = (row.get("category") or "").strip()
            if not phrase or not category:
                continue
            buckets.setdefault(category, []).append(phrase)
    if not buckets:
        raise RuntimeError(f"No seed phrases found in {path}")
    return buckets


def _keyword_candidates(samples: Sequence[str], limit: int = 6) -> List[str]:
    """Extract common keywords from sample phrases."""
    tokens: List[str] = []
    for sample in samples:
        for word in re.findall(r"[A-Za-z']+", sample.lower()):
            if len(word) < 4 or word in STOPWORDS:
                continue
            tokens.append(word)
    if not tokens:
        return []
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(limit)]


def build_category_context(
    categories: Dict[str, List[str]],
    examples_per_category: int,
) -> str:
    """Build descriptive context for Gemini classifier from seed phrases."""
    blocks: List[str] = []
    for name, phrases in sorted(categories.items()):
        keywords = _keyword_candidates(phrases)
        keyword_text = ", ".join(keywords) if keywords else "See examples"
        examples = phrases[:examples_per_category] if phrases else []
        example_lines = "\n".join(f"  - {example}" for example in examples)
        block = (
            f"Category: {name}\n"
            f"Detailed description: Focus on prompts about {keyword_text}.\n"
            f"Representative prompts:\n{example_lines if example_lines else '  - (no examples available)'}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def configure_client(api_key: str | None) -> None:
    """Configure Gemini API client."""
    key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not key:
        raise RuntimeError(
            "Gemini API key missing. Set GEMINI_API_KEY or pass --api-key explicitly!"
        )
    genai.configure(api_key=key)


def load_paragraphs_from_csv(csv_path: Path) -> List[str]:
    """Load paragraphs from pre-extracted CSV."""
    paragraphs = []
    with csv_path.open(encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get('paragraph') or '').strip()
            if text:
                paragraphs.append(text)
    return paragraphs


def extract_paragraphs_from_pdf(
    pdf_path: Path,
    *,
    reader_model: str,
    api_key: str | None,
    page_fraction: float,
) -> List[str]:
    """Extract paragraphs from PDF using Gemini or PyMuPDF fallback."""
    try:
        return extract_paragraphs_gemini(
            str(pdf_path),
            api_key=api_key,
            model=reader_model,
        )
    except Exception as exc:
        print(f"Gemini PDF extraction failed ({exc}). Falling back to PyMuPDF.", file=sys.stderr)
        return extract_paragraphs_pymupdf(str(pdf_path), page_frac=page_fraction)


def select_sample(paragraphs: Sequence[str], fraction: float, seed: int | None) -> List[str]:
    """Sample a fraction of paragraphs randomly."""
    if not paragraphs:
        raise RuntimeError("No paragraphs available for sampling")
    if fraction <= 0 or fraction > 1:
        raise ValueError("Sample fraction must be within (0, 1]")
    rng = random.Random(seed)
    count = max(1, int(len(paragraphs) * fraction))
    return rng.sample(paragraphs, count) if count < len(paragraphs) else list(paragraphs)


def build_request_prompt(category_context: str, paragraph: str) -> str:
    """Build classification prompt for Gemini."""
    return (
        "You classify math textbook paragraphs into the listed categories.\n"
        "Return a JSON object with keys 'category', 'confidence', 'rationale',"
        " and 'supporting_phrases'. Confidence should be between 0 and 1.\n"
        "Only output JSON with double-quoted keys.\n\n"
        f"{category_context}\n\n"
        "Paragraph to classify:\n"
        f'"""{paragraph}"""\n'
        "If nothing fits any category, use 'N/A'."
    )


def classify_paragraph(
    model: genai.GenerativeModel,
    category_context: str,
    paragraph: str,
    *,
    max_retries: int = 3,
    sleep: float = DEFAULT_SLEEP,
) -> Dict[str, object]:
    """Classify a single paragraph using Gemini."""
    prompt = build_request_prompt(category_context, paragraph)
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_NONE"},
                ],
            )
            text = (response.text or "").strip()
            if not text:
                raise ValueError("Empty response from Gemini")
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Gemini response was not a JSON object")
            return data
        except json.JSONDecodeError:
            if attempt == max_retries:
                raise
            time.sleep(sleep)
            continue
        except google_exceptions.ResourceExhausted as err:
            raise RuntimeError(
                "Gemini API quota exhausted; wait before retrying or upgrade your plan."
            ) from err
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(sleep)
    raise RuntimeError("Classification failed after retries")


def write_results(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    """Write classification results to CSV."""
    fieldnames = ["paragraph", "category"]
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemini-powered textbook analysis")
    parser.add_argument("pdf_or_paragraphs", type=Path, help="Path to PDF or pre-extracted paragraphs CSV")
    parser.add_argument(
        "--seed-phrases",
        type=Path,
        default=Path(os.environ.get("SEED_PHRASES_PATH", "seed_phrases.csv")),
        help="CSV file containing 'phrase' and 'category' columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=float(os.environ.get("TEXTBOOK_SAMPLE_FRAC", DEFAULT_SAMPLE_FRACTION)),
        help="Fraction of paragraphs to classify (0-1)",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=int(os.environ.get("EXAMPLES_PER_CATEGORY", DEFAULT_EXAMPLES_PER_CATEGORY)),
        help="Number of seed phrases to show per category",
    )
    parser.add_argument(
        "--classifier-model",
        default=os.environ.get("GEMINI_CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL),
        help="Gemini model name for classification",
    )
    parser.add_argument(
        "--reader-model",
        default=os.environ.get("GEMINI_READER_MODEL", DEFAULT_READER_MODEL),
        help="Gemini model name for PDF extraction",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (overrides GEMINI_API_KEY env)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Reproducible sampling seed",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=float(os.environ.get("GEMINI_REQUEST_SLEEP", DEFAULT_SLEEP)),
        help="Delay between classification retries",
    )
    args = parser.parse_args(argv)

    configure_client(args.api_key)

    categories = read_seed_phrases(args.seed_phrases)
    category_context = build_category_context(categories, args.examples_per_category)

    # Load paragraphs from CSV if provided, otherwise extract from PDF
    input_path = args.pdf_or_paragraphs
    if input_path.suffix.lower() == '.csv':
        print(f"Loading pre-extracted paragraphs from {input_path}")
        paragraphs = load_paragraphs_from_csv(input_path)
        print(f"Loaded {len(paragraphs)} paragraphs from CSV.")
    else:
        print(f"Extracting paragraphs from PDF: {input_path}")
        paragraphs = extract_paragraphs_from_pdf(
            input_path,
            reader_model=args.reader_model,
            api_key=args.api_key,
            page_fraction=args.sample_fraction,
        )
        print(f"Extracted {len(paragraphs)} paragraphs.")

    sample = select_sample(paragraphs, args.sample_fraction, args.random_seed)
    print(f"Selected {len(sample)} paragraphs for classification.")

    model = genai.GenerativeModel(args.classifier_model)

    results: List[Dict[str, object]] = []
    for idx, paragraph in enumerate(sample, start=1):
        print(f"Classifying paragraph {idx}/{len(sample)}")
        try:
            data = classify_paragraph(
                model,
                category_context,
                paragraph,
                sleep=args.sleep,
            )
        except RuntimeError as err:
            print(f"Stopping early: {err}")
            break
        else:
            results.append(
                {
                    "paragraph": paragraph,
                    "category": data.get("category", "Unknown"),
                }
            )

    write_results(args.output, results)
    print(f"Saved {len(results)} classifications to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
