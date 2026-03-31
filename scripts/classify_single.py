#!/usr/bin/env python3
"""
Single-model classifier for parallel runs.

Run qwen and gpt-oss independently in separate terminals, then merge with merge.py.

Usage:
  Terminal 1: python scripts/classify_single.py --name cpm_algebra2 --model qwen
  Terminal 2: python scripts/classify_single.py --name cpm_algebra2 --model gpt_oss

Output (under data/<name>/):
  <model>_results.csv    — all paragraphs with label + reasoning
  <model>_progress.json  — resume checkpoint

Options:
  --name         textbook identifier (required)
  --model        qwen | gpt_oss (required)
  --sleep        seconds between API calls (default: 1.5)
  --start-fresh  clear saved progress and restart
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_SLEEP = 1.5

VALID_CATEGORIES = {
    "INCORRECT_TO_CORRECT",
    "COMPARE_AND_CONTRAST",
    "EXPLICIT_ERROR_DETECTION",
    "COMMON_ERROR_ALERT",
    "NA",
}

MODEL_CONFIGS = {
    "qwen": {
        "model_id": "qwen/qwen3-32b",
        "response_format": {"type": "json_object"},
    },
    "gpt_oss": {
        "model_id": "openai/gpt-oss-120b",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "category":  {"type": "string"},
                    },
                    "required": ["reasoning", "category"],
                },
            },
        },
    },
}

SYSTEM_PROMPT = """You are an expert educational researcher. Your task is to classify
paragraphs from a high school math textbook into ONE of these categories based
on whether the paragraph uses error-focused pedagogy.

CATEGORIES:

INCORRECT_TO_CORRECT
  A named student's WRONG work is shown. The reader is asked to find the error and/or fix it.
  Signals: "find his/her error", "what did [name] do wrong", "correct the work".
  NOT this: asking students to solve a problem or check their own answer.

COMPARE_AND_CONTRAST
  Two or more named students/methods disagree or present different approaches.
  Signals: named characters (Raymond, Sarah, Katelyn, Janelle, etc.) have different answers;
  "who is correct?", "examine both approaches".
  NOT this: two unrelated example problems shown side by side without conflict.

EXPLICIT_ERROR_DETECTION
  An error is stated or shown and the reader must identify/locate it.
  Signals: "find the mistake", "which step is wrong", "identify the error in the following solution".
  NOT this: asking students to check their own answer or verify a result.

COMMON_ERROR_ALERT
  The text WARNS the reader about a mistake students frequently make.
  Signals: "students often...", "a common mistake is...", "be careful not to...",
  "do not confuse X with Y".
  NOT this: general tips or study advice without mentioning errors/mistakes.

NA
  Standard textbook content: problem sets, definitions, examples, instructions,
  vocabulary, reflections. When in doubt, use NA.

REAL EXAMPLES:

"Greg was working on his homework. He completed the square to change y = 2x2 + 24x + 34
into graphing form. Examine his work carefully and find his error. Then correct his work."
-> INCORRECT_TO_CORRECT

"Raymond thinks the answer is x = 4 but Sarah says x = -4. Hannah agrees with Raymond
while Aidan sides with Sarah. Who is correct?"
-> COMPARE_AND_CONTRAST

"Katelyn and Janelle have each started converting y = x2 + 8x + 7 into graphing form
but have gotten different answers. Examine their work and determine who is correct."
-> COMPARE_AND_CONTRAST

"A common mistake is writing C = 100 + 0.04t (linear) instead of C = 100(1.04)^t."
-> COMMON_ERROR_ALERT

"Be sure to test negative numbers. Students often forget to check negative inputs."
-> COMMON_ERROR_ALERT

"Find the equation of the line passing through (-2, 5) perpendicular to y = -5x + 2."
-> NA

"If you needed help solving these problems correctly, then you need more practice."
-> NA

"Are there any points that you think are NOT solutions? How could you check this?"
-> NA

INSTRUCTIONS:
Reason briefly first (1-2 sentences), then give your label.
Respond with valid JSON only:
{"reasoning": "...", "category": "CATEGORY_NAME"}

Use ONLY one of: INCORRECT_TO_CORRECT, COMPARE_AND_CONTRAST,
EXPLICIT_ERROR_DETECTION, COMMON_ERROR_ALERT, NA

When in doubt between an error category and NA, always choose NA.
"""


def load_paragraphs(csv_path: Path) -> List[str]:
    paragraphs = []
    with csv_path.open(encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("paragraph") or "").strip()
            if text:
                paragraphs.append(text)
    return paragraphs


def load_progress(path: Path) -> Dict:
    if path.exists():
        with path.open(encoding="utf8") as f:
            return json.load(f)
    return {"results": {}}


def save_progress(progress: Dict, path: Path) -> None:
    with path.open("w", encoding="utf8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def parse_response(raw: str) -> Tuple[str, str]:
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    try:
        data = json.loads(cleaned)
        category = str(data.get("category", "PARSE_ERROR")).strip().upper()
        reasoning = str(data.get("reasoning", "")).strip()
        label_map = {
            "INCORRECT-TO-CORRECT": "INCORRECT_TO_CORRECT",
            "INCORRECT TO CORRECT": "INCORRECT_TO_CORRECT",
            "COMPARE-AND-CONTRAST": "COMPARE_AND_CONTRAST",
            "COMPARE AND CONTRAST": "COMPARE_AND_CONTRAST",
            "EXPLICIT-ERROR-DETECTION": "EXPLICIT_ERROR_DETECTION",
            "EXPLICIT ERROR DETECTION": "EXPLICIT_ERROR_DETECTION",
            "COMMON-ERROR-ALERT": "COMMON_ERROR_ALERT",
            "COMMON ERROR ALERT": "COMMON_ERROR_ALERT",
            "N/A": "NA",
        }
        category = label_map.get(category, category)
        if category not in VALID_CATEGORIES:
            category = "PARSE_ERROR"
        return category, reasoning
    except json.JSONDecodeError:
        for cat in VALID_CATEGORIES:
            if cat in raw.upper():
                return cat, raw.strip()[:300]
        return "PARSE_ERROR", raw.strip()[:300]


def call_groq(
    model_id: str,
    paragraph: str,
    api_key: str,
    sleep: float,
    response_format: dict,
    max_retries: int = 8,
) -> Tuple[str, str]:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Paragraph to classify:\n"""{paragraph}"""'},
        ],
        "temperature": 0.1,
        "response_format": response_format,
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    rate_limit_attempts = 0
    non_429_attempts = 0

    while True:
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 429:
                rate_limit_attempts += 1
                wait = min(15 * (2 ** (rate_limit_attempts - 1)), 120)
                print(f"    Rate limited (#{rate_limit_attempts}) — waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            time.sleep(sleep)
            return parse_response(raw)
        except requests.exceptions.Timeout:
            non_429_attempts += 1
            print(f"    Timeout ({non_429_attempts}/{max_retries})")
            if non_429_attempts >= max_retries:
                return "ERROR", "timeout"
            time.sleep(10 * non_429_attempts)
        except requests.exceptions.HTTPError as e:
            non_429_attempts += 1
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text[:200]
            print(f"    HTTP {e.response.status_code} ({non_429_attempts}/{max_retries}): {body}")
            if e.response.status_code == 400:
                return "ERROR", f"400: {body}"
            if non_429_attempts >= max_retries:
                return "ERROR", str(e)[:200]
            time.sleep(10 * non_429_attempts)
        except Exception as e:
            non_429_attempts += 1
            print(f"    Error ({non_429_attempts}/{max_retries}): {str(e)[:100]}")
            if non_429_attempts >= max_retries:
                return "ERROR", str(e)[:200]
            time.sleep(10 * non_429_attempts)


def main():
    parser = argparse.ArgumentParser(description="Single-model classifier for parallel runs")
    parser.add_argument("--name",        required=True, help="Textbook identifier")
    parser.add_argument("--model",       required=True, choices=["qwen", "gpt_oss"],
                        help="Which model to run: qwen | gpt_oss")
    parser.add_argument("--sleep",       type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--start-fresh", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env")
        return 1

    cfg = MODEL_CONFIGS[args.model]
    model_id = cfg["model_id"]
    response_format = cfg["response_format"]

    data_dir      = Path("data") / args.name
    input_path    = data_dir / "paragraphs.csv"
    output_path   = data_dir / f"{args.model}_results.csv"
    progress_path = data_dir / f"{args.model}_progress.json"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run extract.py first.")
        return 1

    if args.start_fresh and progress_path.exists():
        progress_path.unlink()
        print("Starting fresh — cleared previous progress")

    print(f"Loading paragraphs from {input_path}...")
    all_paragraphs = load_paragraphs(input_path)
    total = len(all_paragraphs)
    print(f"Loaded {total} paragraphs")

    progress = load_progress(progress_path)
    already_done = set(progress["results"].keys())
    remaining = [p for p in all_paragraphs if p not in already_done]

    eta_mins = len(remaining) * (args.sleep + 1.0) / 60
    print(f"Already done: {len(already_done)} | Remaining: {len(remaining)}")
    print(f"Model: {model_id}")
    print(f"Estimated time: {eta_mins:.0f} min ({eta_mins/60:.1f} hrs)")
    print("━" * 60)

    if not output_path.exists():
        with output_path.open("w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "paragraph", f"{args.model}_label", f"{args.model}_reasoning"
            ])
            writer.writeheader()

    processed = 0
    errors = 0

    for i, paragraph in enumerate(remaining):
        idx = len(already_done) + i + 1
        short = paragraph[:80].replace("\n", " ")
        print(f"\n[{idx}/{total}] {short}...")

        cat, reason = call_groq(model_id, paragraph, api_key, args.sleep, response_format)
        status = f"✓ {cat}" if cat in VALID_CATEGORIES else f"✗ {cat}"
        print(f"  {args.model:<10} {status}")

        if cat == "ERROR":
            errors += 1

        with output_path.open("a", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "paragraph", f"{args.model}_label", f"{args.model}_reasoning"
            ])
            writer.writerow({
                "paragraph": paragraph,
                f"{args.model}_label": cat,
                f"{args.model}_reasoning": reason,
            })

        progress["results"][paragraph] = {"label": cat, "reasoning": reason}
        processed += 1

        if processed % 10 == 0:
            save_progress(progress, progress_path)
            pct = 100 * (len(already_done) + processed) / total
            print(f"\n  ── Saved: {len(already_done) + processed}/{total} ({pct:.1f}%) | errors: {errors} ──")

    save_progress(progress, progress_path)

    print("\n" + "━" * 60)
    print("COMPLETE")
    print(f"  Total:  {len(already_done) + processed}/{total}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())