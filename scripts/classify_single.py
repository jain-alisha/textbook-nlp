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
import hashlib
import json
import os
import re
import sys
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

# The two arms must come from different lineages: inter-model agreement is the
# reliability metric, and two models from one vendor correlate, inflating
# agreement without buying independence.
MODEL_CONFIGS = {
    "gemini": {
        "model_id": "gemini-2.5-flash",
        "provider": "gemini",
    },
    "qwen": {
        "model_id": "qwen/qwen3.8-27b",
        "provider": "groq",
        "response_format": {"type": "json_object"},
    },
    "gpt_oss": {
        "model_id": "openai/gpt-oss-120b",
        "provider": "groq",
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


def cache_key(model_id: str, paragraph: str) -> str:
    """Identify a result by what produced it, not by row position.

    Includes a fingerprint of the prompt so editing SYSTEM_PROMPT or switching
    models invalidates affected entries instead of serving stale labels.
    """
    prompt_fingerprint = hashlib.sha256(SYSTEM_PROMPT.encode("utf8")).hexdigest()[:12]
    payload = f"{model_id}\x00{prompt_fingerprint}\x00{paragraph}"
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def load_cache(path: Path, model_id: str) -> Dict[str, Tuple[str, str]]:
    """Read the append-only cache, keeping the last entry for each key."""
    if not path.exists():
        return {}
    cached: Dict[str, Tuple[str, str]] = {}
    with path.open(encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from an interrupted run
            if rec.get("model_id") == model_id:
                cached[rec["key"]] = (rec["label"], rec.get("reasoning", ""))
    return cached


def append_cache(path: Path, key: str, model_id: str,
                 paragraph: str, label: str, reasoning: str) -> None:
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps({
            "key": key,
            "model_id": model_id,
            "paragraph": paragraph,
            "label": label,
            "reasoning": reasoning,
        }, ensure_ascii=False) + "\n")


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


class DailyQuotaExhausted(Exception):
    """Every further call fails until the quota resets; stop rather than fill
    the cache with ERROR rows."""


_GEMINI_CLIENT = None


def _gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai
        from google.genai import types
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY not found in .env")
        _GEMINI_CLIENT = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=300_000))
    return _GEMINI_CLIENT


def call_gemini(model_id: str, paragraph: str, sleep: float,
                max_retries: int = 6) -> Tuple[str, str]:
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=0.1,
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {"reasoning": {"type": "string"},
                           "category": {"type": "string"}},
            "required": ["reasoning", "category"],
        },
    )
    attempts = 0
    while True:
        try:
            resp = _gemini_client().models.generate_content(
                model=model_id,
                contents=f'Paragraph to classify:\n"""{paragraph}"""',
                config=config,
            )
            time.sleep(sleep)
            return parse_response(resp.text or "")
        except Exception as e:
            if "PerDay" in str(e):
                raise DailyQuotaExhausted(str(e)) from e
            attempts += 1
            print(f"    Gemini error ({attempts}/{max_retries}): {str(e)[:120]}")
            if attempts >= max_retries:
                return "ERROR", str(e)[:200]
            time.sleep(10 * attempts)


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
    parser.add_argument("--model",       required=True,
                        choices=["gemini", "qwen", "gpt_oss"],
                        help="Which arm to run: gemini | qwen | gpt_oss")
    parser.add_argument("--paragraphs",  default=None,
                        help="Paragraphs CSV filename (default: paragraphs.csv)")
    parser.add_argument("--sleep",       type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--start-fresh", action="store_true")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    model_id = cfg["model_id"]
    provider = cfg["provider"]
    response_format = cfg.get("response_format")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if provider == "groq" and not api_key:
        print("ERROR: GROQ_API_KEY not found in .env")
        return 1
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY", "").strip():
        print("ERROR: GEMINI_API_KEY not found in .env")
        return 1

    data_dir      = Path("data") / args.name
    para_file  = args.paragraphs if args.paragraphs else "paragraphs.csv"
    input_path    = data_dir / para_file
    output_path   = data_dir / f"{args.model}_results.csv"
    cache_path    = data_dir / f"{args.model}_cache.jsonl"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run extract.py first.")
        return 1

    if args.start_fresh and cache_path.exists():
        cache_path.unlink()
        print("Starting fresh — cleared cached results")

    print(f"Loading paragraphs from {input_path}...")
    all_paragraphs = load_paragraphs(input_path)
    total = len(all_paragraphs)
    print(f"Loaded {total} paragraphs")

    cached = load_cache(cache_path, model_id)
    to_call = sum(1 for p in all_paragraphs if cache_key(model_id, p) not in cached)

    eta_mins = to_call * (args.sleep + 1.0) / 60
    print(f"Cached: {total - to_call} | Need API call: {to_call}")
    print(f"Model: {model_id}")
    print(f"Estimated time: {eta_mins:.0f} min ({eta_mins/60:.1f} hrs)")
    print("━" * 60)

    # The cache is the durable store; this CSV is a derived view rebuilt each
    # run, so every input row appears exactly once even when texts repeat.
    with output_path.open("w", encoding="utf8", newline="") as f:
        csv.DictWriter(f, fieldnames=[
            "paragraph", f"{args.model}_label", f"{args.model}_reasoning"
        ]).writeheader()

    processed = 0
    errors = 0

    for idx, paragraph in enumerate(all_paragraphs, start=1):
        key = cache_key(model_id, paragraph)
        hit = cached.get(key)

        if hit is not None:
            cat, reason = hit
        else:
            short = paragraph[:80].replace("\n", " ")
            print(f"\n[{idx}/{total}] {short}...")
            if provider == "gemini":
                try:
                    cat, reason = call_gemini(model_id, paragraph, args.sleep)
                except DailyQuotaExhausted:
                    print(f"\nERROR: Gemini daily quota exhausted after {processed} new "
                          f"paragraphs. Cached work is saved; re-run after it resets.",
                          file=sys.stderr)
                    return 3
            else:
                cat, reason = call_groq(model_id, paragraph, api_key, args.sleep,
                                        response_format)
            status = f"✓ {cat}" if cat in VALID_CATEGORIES else f"✗ {cat}"
            print(f"  {args.model:<10} {status}")

            if cat == "ERROR":
                # Caching a failure would make a re-run skip the paragraph and
                # bake the error into the results permanently.
                errors += 1
            else:
                append_cache(cache_path, key, model_id, paragraph, cat, reason)
                cached[key] = (cat, reason)
            processed += 1

            if processed % 10 == 0:
                pct = 100 * idx / total
                print(f"\n  ── {idx}/{total} ({pct:.1f}%) | new: {processed} | errors: {errors} ──")

        with output_path.open("a", encoding="utf8", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "paragraph", f"{args.model}_label", f"{args.model}_reasoning"
            ]).writerow({
                "paragraph": paragraph,
                f"{args.model}_label": cat,
                f"{args.model}_reasoning": reason,
            })

    print("\n" + "━" * 60)
    # Record which ensemble arm produced these labels. Groq decommissioned
    # qwen3-32b mid-project, and nothing in the results CSVs distinguishes labels
    # made by one model version from another — which silently splits the
    # inter-model agreement rate across ensembles.
    manifest_path = data_dir / f"{args.model}_manifest.json"
    manifest_path.write_text(json.dumps({
        "model_arm": args.model,
        "model_id": model_id,
        "prompt_sha256_12": hashlib.sha256(SYSTEM_PROMPT.encode("utf8")).hexdigest()[:12],
        "paragraphs_file": para_file,
        "rows": total,
        "new_api_calls": processed,
        "errors": errors,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n", encoding="utf8")

    print("COMPLETE")
    print(f"  Rows written: {total}")
    print(f"  New API calls: {processed}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())