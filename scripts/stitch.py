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
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID     = "qwen/qwen3.8-27b"
DEFAULT_SLEEP = 1.0

# Local backend: free, slower (~4-11s/decision), no API key. Measured against the
# Groq model it agrees on 92% of real-content pairs, but leans toward merging —
# MAX_MERGED_CHARS and the problem-number filter matter more here.
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"

SYSTEM_PROMPT = """You are a document processing assistant. You will be given two consecutive
text chunks extracted from a math textbook PDF. Punctuation-based merging has already been
applied, so both chunks end with complete sentences. Your job is to decide whether they form
a single continuous pedagogical unit or are two independent passages.

Merge them if:
- Chunk A is clearly the narrative setup for chunk B (e.g. introduces named characters or a
  problem scenario that chunk B then develops or resolves)
- Together they form one coherent problem or instructional unit that would lose meaning
  if read separately

Keep them separate if:
- Both chunks are clearly complete and independent (different problems, different topics)
- Chunk B starts a new numbered problem or new topic unrelated to chunk A
- They are adjacent but thematically unrelated

Respond with valid JSON only, and nothing else:
{"merge": true}
or
{"merge": false}

Do not echo the chunks back. Judge only; the text is joined separately.
"""

# A chunk opening a new numbered problem ("1-41.", or with a running header,
# "Core Connections Course 1 2-104.") starts its own unit, so it is not a
# continuation of the chunk before it. Deciding these in code skips roughly half
# the API calls on CPM books.
NEW_PROBLEM_RE = re.compile(r"^\s*(?:[A-Za-z][A-Za-z0-9 ]{0,40}?\s)?\d+-\d+\.\s")

# ...except when the extractor split a problem across pages and restated its
# number, e.g. "1-2. Continued from previous page. b. ...", which must still merge.
CONTINUATION_RE = re.compile(r"continued from previous page|\(continued\)|continued\b", re.I)


def starts_new_problem(chunk: str) -> bool:
    return bool(NEW_PROBLEM_RE.match(chunk)) and not CONTINUATION_RE.search(chunk[:120])


# Merging is chained: each merge grows the tail, which is then tested against the
# next paragraph. On books whose chunks often begin mid-sentence the model keeps
# answering "continues" and a single chunk can swallow the book (one CK-12 chunk
# reached 601KB and overflowed the context window). No real pedagogical unit is
# this long, so refuse merges past the cap.
MAX_MERGED_CHARS = 15000


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


def call_qwen(chunk_a: str, chunk_b: str, api_key: str, sleep: float,
              backend: str = "groq") -> bool:
    """Ask the model whether two consecutive chunks belong together.

    Returns a judgment only. The merged text is assembled in code so the output
    is the textbook's own wording, and so chained merges don't pay to have an
    ever-growing chunk retyped on every call.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"CHUNK A:\n{chunk_a}\n\nCHUNK B:\n{chunk_b}"},
    ]
    if backend == "ollama":
        url = OLLAMA_URL
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,   # adds ~1 point of agreement for a large slowdown
            "format": "json",
            "options": {"temperature": 0, "num_predict": 200},
        }
        timeout = 600
    else:
        url = GROQ_URL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "max_tokens": 200,
        }
        timeout = 90

    rate_attempts = 0
    fail_attempts = 0

    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                rate_attempts += 1
                wait = min(15 * (2 ** (rate_attempts - 1)), 120)
                print(f"    Rate limited (#{rate_attempts}) — waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            body = resp.json()
            raw = (body["message"]["content"] if backend == "ollama"
                   else body["choices"][0]["message"]["content"]).strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            time.sleep(sleep)
            try:
                data = json.loads(raw)
                if "merge" not in data:
                    raise ValueError(f"no 'merge' key in {raw[:80]!r}")
                return bool(data["merge"])
            except json.JSONDecodeError:
                # The answer is one boolean; don't discard it because the model
                # added a field or ran past the token limit mid-string.
                verdict = re.search(r'"merge"\s*:\s*(true|false)', raw, re.I)
                if verdict:
                    return verdict.group(1).lower() == "true"
                raise
        except Exception as e:
            fail_attempts += 1
            body = getattr(getattr(e, "response", None), "text", "")
            detail = f" | {body[:300]}" if body else ""
            print(f"    Error ({fail_attempts}): {str(e)[:120]}{detail}")
            if fail_attempts >= 5:
                return False
            time.sleep(10 * fail_attempts)


def stitch(paragraphs: List[str], api_key: str, sleep: float,
           progress: dict, progress_path: Path, backend: str = "groq") -> List[str]:
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

        if len(current_tail) + len(next_para) > MAX_MERGED_CHARS:
            print(f"  → SEPARATE (merge cap: tail is {len(current_tail)} chars)")
            chunks.append(next_para)
            i += 1
            progress["processed_up_to"] = i
            progress["chunks"] = chunks
            continue

        if starts_new_problem(next_para):
            print(f"  → SEPARATE (new problem number)")
            chunks.append(next_para)
            i += 1
            progress["processed_up_to"] = i
            progress["chunks"] = chunks
            continue

        should_merge = call_qwen(current_tail, next_para, api_key, sleep, backend)

        if should_merge:
            print(f"  → MERGED")
            chunks[-1] = f"{current_tail} {next_para}"
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
    parser.add_argument("--backend",     choices=["groq", "ollama"], default="groq",
                        help="groq (API key, fast) or ollama (local, free, slower)")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if args.backend == "groq" and not api_key:
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
        per_call = 8.0 if args.backend == "ollama" else args.sleep + 1.0
        eta = total * per_call / 60
        print(f"Estimated time: {eta:.0f} min ({eta/60:.1f} hrs)")
        print(f"Backend: {args.backend} "
              f"({OLLAMA_MODEL if args.backend == 'ollama' else MODEL_ID})")
        print("━" * 60)
        stitched = stitch(paragraphs, api_key, args.sleep, progress, progress_path,
                          args.backend)

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