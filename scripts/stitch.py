#!/usr/bin/env python3
"""
Semantic paragraph stitching pass.

Reads paragraphs.csv, feeds consecutive chunk pairs to qwen, and merges
chunks that belong together into a single pedagogical unit.

Usage:
  python scripts/stitch.py --name cpm_algebra2

Output:
  data/<name>/paragraphs_stitched.csv   — cleaned, merged paragraphs
  data/<name>/stitch_progress.json      — resumable checkpoint

Then re-classify with:
  python scripts/run_pipeline.py --name cpm_algebra2 --paragraphs paragraphs_stitched.csv --quota-wait 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import List, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID     = "qwen/qwen3-32b"
DEFAULT_SLEEP = 1.0

SYSTEM_PROMPT = """You are a document processing assistant. You will be given two consecutive
text chunks extracted from a math textbook PDF. Your job is to decide whether they form a
single continuous passage or are two independent passages.

Merge them if:
- Chunk A ends mid-sentence (no terminal punctuation: . ? ! ) and chunk B continues it
- Chunk A is clearly the narrative setup for chunk B (e.g. introduces characters or a problem
  that chunk B then develops)
- Together they form one coherent problem or instructional unit

Keep them separate if:
- Both chunks are clearly complete and independent (different problems, different topics)
- Chunk B starts a new numbered problem or new topic unrelated to chunk A
- Merging would combine two genuinely different pedagogical moments

Respond with valid JSON only:
{
  "merge": true or false,
  "result": "merged text if merge=true, or leave empty string if merge=false"
}

If merge=false, return exactly: {"merge": false, "result": ""}
If merge=true, return the cleanly merged text with no duplicate content.
"""

def load_paragraphs(path: Path) -> List[str]:
    paras = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("paragraph") or "").strip()
            if text:
                paras.append(text)
    return paras


def load_progress(path: Path) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return {"processed_up_to": 0, "chunks": []}


def save_progress(progress: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def call_qwen(chunk_a: str, chunk_b: str, api_key: str, sleep: float) -> Tuple[bool, str]:
    """Ask qwen whether to merge two consecutive chunks."""
    user_msg = f"CHUNK A:\n{chunk_a}\n\nCHUNK B:\n{chunk_b}"
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 800,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    rate_attempts = 0
    fail_attempts = 0

    while True:
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 429:
                rate_attempts += 1
                wait = min(15 * (2 ** (rate_attempts - 1)), 120)
                print(f"    Rate limited (#{rate_attempts}) — waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            data = json.loads(raw)
            time.sleep(sleep)
            return bool(data.get("merge", False)), str(data.get("result", "")).strip()
        except Exception as e:
            fail_attempts += 1
            print(f"    Error ({fail_attempts}): {str(e)[:80]}")
            if fail_attempts >= 5:
                return False, ""
            time.sleep(10 * fail_attempts)


def stitch(paragraphs: List[str], api_key: str, sleep: float,
           progress: dict, progress_path: Path) -> List[str]:
    """
    Iterate through paragraphs, merging consecutive chunks where appropriate.
    Resumes from progress["processed_up_to"].
    """
    # Rebuild chunks list from progress if resuming
    chunks: List[str] = list(progress.get("chunks", []))
    start_i = int(progress.get("processed_up_to", 0))

    # If starting fresh, seed with first paragraph
    if start_i == 0:
        chunks = [paragraphs[0]]
        start_i = 1

    total = len(paragraphs)

    i = start_i
    while i < total:
        current_tail = chunks[-1]   # last chunk in our output so far
        next_para    = paragraphs[i]

        short_tail = current_tail[:60].replace("\n", " ")
        short_next = next_para[:60].replace("\n", " ")
        print(f"\n[{i}/{total-1}] Checking merge:")
        print(f"  A: {short_tail}...")
        print(f"  B: {short_next}...")

        should_merge, merged_text = call_qwen(current_tail, next_para, api_key, sleep)

        if should_merge and merged_text:
            print(f"  → MERGED")
            chunks[-1] = merged_text   # replace tail with merged version
        else:
            print(f"  → SEPARATE")
            chunks.append(next_para)   # add as new independent chunk

        i += 1
        progress["processed_up_to"] = i
        progress["chunks"] = chunks

        if i % 20 == 0:
            save_progress(progress, progress_path)
            print(f"\n  ── Saved: {i}/{total} processed, {len(chunks)} chunks so far ──")

    save_progress(progress, progress_path)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic paragraph stitching")
    parser.add_argument("--name",        required=True,
                        help="Textbook identifier (e.g. cpm_algebra2)")
    parser.add_argument("--sleep",       type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--start-fresh", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env")
        return 1

    data_dir      = Path("data") / args.name
    input_path    = data_dir / "paragraphs.csv"
    output_path   = data_dir / "paragraphs_stitched.csv"
    progress_path = data_dir / "stitch_progress.json"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run extract.py first.")
        return 1

    if args.start_fresh and progress_path.exists():
        progress_path.unlink()
        print("Starting fresh")

    print(f"Loading paragraphs from {input_path}...")
    paragraphs = load_paragraphs(input_path)
    total = len(paragraphs)
    print(f"Loaded {total} paragraphs")

    progress = load_progress(progress_path)

    # If already complete, just write output
    if progress.get("processed_up_to", 0) >= total and progress.get("chunks"):
        print("Already complete — writing output from saved progress")
        stitched = progress["chunks"]
    else:
        eta = total * (args.sleep + 1.0) / 60
        print(f"Estimated time: {eta:.0f} min ({eta/60:.1f} hrs)")
        print(f"Model: {MODEL_ID}")
        print("━" * 60)
        stitched = stitch(paragraphs, api_key, args.sleep, progress, progress_path)

    # Write output
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["paragraph"])
        for para in stitched:
            writer.writerow([para])

    reduction = total - len(stitched)
    print(f"\n{'━'*60}")
    print(f"STITCHING COMPLETE")
    print(f"  Original chunks:  {total}")
    print(f"  Stitched chunks:  {len(stitched)}")
    print(f"  Merges performed: {reduction} ({100*reduction/total:.1f}% reduction)")
    print(f"  Output: {output_path}")
    print(f"\nNext step:")
    print(f"  python scripts/run_pipeline.py --name {args.name} --quota-wait 0")
    print(f"  (update run_pipeline.py to read paragraphs_stitched.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())