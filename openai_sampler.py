#!/usr/bin/env python3
"""Sample textbook paragraphs and classify them with OpenAI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import requests
from dotenv import load_dotenv

PROMPT = (
    "You analyze math textbook paragraphs and label them with one of these categories: "
    "Incorrect-to-correct revision tasks, Compare-and-contrast examples, Explicit error detection, "
    "Common error alerts, N/A. Return JSON: {\"category\": ..., \"rationale\": ...}."
)

REQUIRED_SNIPPETS: Sequence[str] = (
    "In your work with parabolas, you have developed two forms",
    "The cost of food has been increasing by 4% per year",
    "b. Be sure to test negative numbers. What happens for negative numbers?",
    "Greg was working on his homework. He completed the square to change y = 2x",
    "Yeah, I think you’re wrong, Marta",
)


@dataclass
class ParagraphRecord:
    index: int
    text: str


def load_paragraphs(path: Path) -> List[str]:
    with path.open(encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        paragraphs = [
            (row.get("paragraph") or "").strip()
            for row in reader
            if (row.get("paragraph") or "").strip()
        ]
    if not paragraphs:
        raise RuntimeError(f"No paragraphs found in {path}")
    return paragraphs


def find_required(paragraphs: Sequence[str], snippets: Sequence[str]) -> Dict[str, ParagraphRecord]:
    matches: Dict[str, ParagraphRecord] = {}
    lowered = [p.lower() for p in paragraphs]
    for snippet in snippets:
        snippet_lower = snippet.lower()
        found = False
        for idx, text in enumerate(lowered):
            if snippet_lower in text:
                matches[snippet] = ParagraphRecord(idx, paragraphs[idx])
                found = True
                break
        if not found:
            raise RuntimeError(f"Required snippet not found: {snippet}")
    return matches


def pick_additional(paragraphs: Sequence[str], taken: Sequence[int], total: int) -> List[int]:
    needed = total - len(taken)
    if needed < 0:
        raise ValueError("Total sample cannot be smaller than required count")
    if needed == 0:
        return list(taken)
    taken_set = set(taken)
    center = len(paragraphs) // 2
    ordered_indices = sorted(range(len(paragraphs)), key=lambda i: (abs(i - center), i))
    extras: List[int] = []
    for idx in ordered_indices:
        if idx in taken_set:
            continue
        extras.append(idx)
        if len(extras) >= needed:
            break
    if len(extras) < needed:
        raise RuntimeError("Not enough paragraphs to complete sample")
    combined = sorted(set(taken).union(extras))
    return combined


def call_openai(
    paragraph: str,
    model: str,
    timeout: int,
    max_retries: int,
    retry_delay: float,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": paragraph},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else None
            if status == 429 and attempt < max_retries:
                delay = retry_delay * attempt
                time.sleep(delay)
                continue
            raise

    raise RuntimeError("OpenAI request failed after retries")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample paragraphs and query OpenAI")
    parser.add_argument("--input", type=Path, default=Path("paragraph_clusters.csv"))
    parser.add_argument("--output", type=Path, default=Path("openai_results.jsonl"))
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between successful calls")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args(argv)

    load_dotenv()

    paragraphs = load_paragraphs(args.input)
    required = find_required(paragraphs, REQUIRED_SNIPPETS)
    required_indices = [rec.index for rec in required.values()]
    sample_indices = pick_additional(paragraphs, required_indices, args.sample_size)

    records = [ParagraphRecord(idx, paragraphs[idx]) for idx in sample_indices]

    with args.output.open("w", encoding="utf8") as handle:
        for record in records:
            if args.dry_run:
                result = "DRY_RUN"
            else:
                try:
                    result = call_openai(
                        record.text,
                        args.model,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        retry_delay=args.sleep,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = f"ERROR: {exc}"
            row = {
                "index": record.index,
                "paragraph": record.text,
                "model": args.model,
                "response": result,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Processed paragraph #{record.index}")
            if not args.dry_run and args.sleep:
                time.sleep(args.sleep)

    print(f"Saved results to {args.output}")
    if args.dry_run:
        print("(Dry run: no API calls were made.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
