#!/usr/bin/env python3
"""
Ensemble classifier: 2-model agreement via Groq.

Models:
  1. qwen/qwen3-32b       (primary validator)
  2. openai/gpt-oss-120b  (strongest, different architecture)

Voting logic:
  - 2/2 agree  → CONFIRMED   (highest confidence)
  - disagree   → written to uncertain_review.csv, skipped in main output

Output files (under data/<name>/):
  classified_results.csv   — confirmed paragraphs with labels
  uncertain_review.csv     — paragraphs where models disagreed
  progress.json            — resumable progress (safe to Ctrl+C and re-run)

Usage:
  python scripts/classify.py --name cpm_algebra2

Resume after interruption:
  python scripts/classify.py --name cpm_algebra2   (auto-resumes)

Options:
  --name           textbook identifier (required)
  --sleep          seconds between individual API calls (default: 2.0)
  --para-pause     extra seconds between paragraphs (default: 5.0)
  --start-fresh    ignore saved progress and restart from scratch

Rate limiting:
  Uses Groq flex service tier (10x higher limits, still free).
  Exponential backoff on 429s — will keep retrying until it succeeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────────────
# gpt-oss-120b: needs json_schema format
# qwen:         uses json_object format
MODELS = [
    ("qwen",    "qwen/qwen3-32b",       {"type": "json_object"}),
    ("gpt_oss", "openai/gpt-oss-120b",  {
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
    }),
]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_SLEEP      = 2.0   # seconds between each individual model call
DEFAULT_PARA_PAUSE = 5.0   # extra seconds between paragraphs (after all 3 models)

VALID_CATEGORIES = {
    "INCORRECT_TO_CORRECT",
    "COMPARE_AND_CONTRAST",
    "EXPLICIT_ERROR_DETECTION",
    "COMMON_ERROR_ALERT",
    "NA",
}

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert educational researcher. Your task is to classify
paragraphs from a high school math textbook into ONE of these six categories based
on whether the paragraph uses error-focused pedagogy.

━━━ CATEGORIES ━━━

INCORRECT_TO_CORRECT
  A named student's WRONG work is shown. The reader is asked to find the error
  and/or fix it.
  ✓ Signals: "find his/her error", "what did [name] do wrong", "correct the work",
    a student's solution steps are shown and are incorrect.
  ✗ NOT this: just asking students to solve a problem or check their own answer.

COMPARE_AND_CONTRAST
  Two or more named students/methods disagree or present different approaches.
  The reader evaluates which is correct or better.
  ✓ Signals: named characters (Raymond, Sarah, Katelyn, Janelle, etc.) have
    different answers or methods; "who is correct?", "examine both approaches".
  ✗ NOT this: two unrelated example problems shown side by side without conflict.

EXPLICIT_ERROR_DETECTION
  An error is stated or shown in the text, and the reader must identify/locate it
  — but NOT necessarily from a named student's work.
  ✓ Signals: "find the mistake", "which step is wrong", "identify the error in
    the following solution".
  ✗ NOT this: asking students to check their own answer or verify a result.

COMMON_ERROR_ALERT
  The text WARNS the reader about a mistake that students frequently make.
  ✓ Signals: "students often...", "a common mistake is...", "be careful not to...",
    "do not confuse X with Y", caution boxes about typical misconceptions.
  ✗ NOT this: general tips or study advice without mentioning errors/mistakes.

NA
  Standard textbook content: problem sets, definitions, examples, instructions,
  vocabulary, reflections, or anything else that does NOT involve error pedagogy.
  When in doubt and nothing above fits clearly, use NA.

━━━ REAL EXAMPLES FROM THIS TEXTBOOK ━━━

Paragraph: "Greg was working on his homework. He completed the square to change
y = 2x² + 24x + 34 into graphing form. His work is shown below. Examine his work
carefully and find his error. Then correct his work and find the vertex."
→ INCORRECT_TO_CORRECT  (Greg's specific wrong work is shown; reader fixes it)

Paragraph: "Raymond thinks the answer is x = 4 but Sarah says x = −4. Hannah
agrees with Raymond while Aidan sides with Sarah. Who is correct?"
→ COMPARE_AND_CONTRAST  (named students disagree; reader adjudicates)

Paragraph: "Katelyn and Janelle have each started converting y = x² + 8x + 7 into
graphing form but have gotten different answers. Examine their work and determine
who is correct, or if both are wrong."
→ COMPARE_AND_CONTRAST  (named students, conflicting answers, reader evaluates)

Paragraph: "A common mistake is writing C = 100 + 0.04t (linear) instead of the
correct exponential model C = 100(1.04)^t."
→ COMMON_ERROR_ALERT  (explicit warning about a frequent mistake)

Paragraph: "Be sure to test negative numbers. Students often forget to check
negative inputs when testing functions, leading to incorrect conclusions."
→ COMMON_ERROR_ALERT  (caution about a typical student error)

Paragraph: "Find the equation of the line passing through (−2, 5) perpendicular
to y = −5x + 2."
→ NA  (standard problem, no error pedagogy)

Paragraph: "If you needed help solving these problems correctly, then you need more
practice. Review the Checkpoint materials at the back of your book."
→ NA  (study guidance, not error pedagogy)

Paragraph: "Are there any points that you think are NOT solutions? How could you
check this?"
→ NA  (exploratory question, not asking to find error in shown work)

Paragraph: "Show how to use her method to write a general expression for the sum."
→ NA  (following a method, no error involved)

━━━ INSTRUCTIONS ━━━

Reason briefly first (1–2 sentences), then give your label.
Respond with valid JSON only — no markdown, no extra text:
{"reasoning": "...", "category": "CATEGORY_NAME"}

Use ONLY one of: INCORRECT_TO_CORRECT, COMPARE_AND_CONTRAST,
EXPLICIT_ERROR_DETECTION, COMMON_ERROR_ALERT, NA

When in doubt between an error category and NA, always choose NA.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_paragraphs(csv_path: Path) -> List[str]:
    paragraphs = []
    with csv_path.open(encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("paragraph") or "").strip()
            if text:
                paragraphs.append(text)
    return paragraphs


def load_progress(progress_path: Path) -> Dict:
    if progress_path.exists():
        with progress_path.open(encoding="utf8") as f:
            return json.load(f)
    return {"classified": {}, "uncertain": []}


def save_progress(progress: Dict, progress_path: Path) -> None:
    with progress_path.open("w", encoding="utf8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def parse_response(raw: str) -> Tuple[str, str]:
    """Extract (category, reasoning) from model response."""
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    # qwen3 sometimes emits <think>...</think> blocks before JSON — strip them
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    try:
        data = json.loads(cleaned)
        category = str(data.get("category", "PARSE_ERROR")).strip().upper()
        reasoning = str(data.get("reasoning", "")).strip()
        # Normalize variants
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
    """Call a single Groq model. Returns (category, reasoning).
    Exponential backoff on 429s — retries until max_retries non-429 failures.
    """
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
    non_429_attempts    = 0

    while True:
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)

            if resp.status_code == 429:
                rate_limit_attempts += 1
                # Exponential backoff: 15s, 30s, 60s, 120s, cap at 120s
                wait = min(15 * (2 ** (rate_limit_attempts - 1)), 120)
                print(f"    Rate limited (attempt {rate_limit_attempts}) — waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            time.sleep(sleep)
            return parse_response(raw)

        except requests.exceptions.Timeout:
            non_429_attempts += 1
            print(f"    Timeout (attempt {non_429_attempts}/{max_retries})")
            if non_429_attempts >= max_retries:
                return "ERROR", "timeout"
            time.sleep(10 * non_429_attempts)

        except requests.exceptions.HTTPError as e:
            non_429_attempts += 1
            body = ""
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text[:300]
            print(f"    HTTP {e.response.status_code} error (attempt {non_429_attempts}/{max_retries}): {body}")
            if e.response.status_code == 400:
                # 400s won't be fixed by retrying — bail immediately
                return "ERROR", f"400: {body}"
            if non_429_attempts >= max_retries:
                return "ERROR", str(e)[:200]
            time.sleep(10 * non_429_attempts)
        except Exception as e:
            non_429_attempts += 1
            print(f"    Error (attempt {non_429_attempts}/{max_retries}): {str(e)[:100]}")
            if non_429_attempts >= max_retries:
                return "ERROR", str(e)[:200]
            time.sleep(10 * non_429_attempts)


def majority_vote(votes: List[str]) -> Tuple[str, str]:
    """
    Given 2 category votes, return (final_label, confidence).
    CONFIRMED = both agree, UNCERTAIN = disagree or both errored.
    """
    valid = [v for v in votes if v in VALID_CATEGORIES]

    if len(valid) == 0:
        return "PARSE_ERROR", "ERROR"

    if len(valid) == 2 and valid[0] == valid[1]:
        return valid[0], "CONFIRMED"

    if len(valid) == 1:
        # Only one model got through — not enough to confirm, flag for review
        return "UNCERTAIN", "UNCERTAIN"

    # Both valid but disagree
    return "UNCERTAIN", "UNCERTAIN"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="2-model Groq ensemble classifier for textbook paragraphs")
    parser.add_argument("--name", required=True,
                        help="Textbook identifier (e.g. cpm_algebra2). Reads data/<name>/paragraphs.csv.")
    parser.add_argument("--sleep",      type=float, default=DEFAULT_SLEEP,
                        help="Seconds between individual model calls (default: 2.0)")
    parser.add_argument("--para-pause", type=float, default=DEFAULT_PARA_PAUSE,
                        help="Extra pause between paragraphs after all models (default: 5.0)")
    parser.add_argument("--start-fresh", action="store_true",
                        help="Ignore saved progress and restart")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment / .env")
        return 1

    data_dir = Path("data") / args.name
    input_path    = data_dir / "paragraphs.csv"
    output_path   = data_dir / "classified_results.csv"
    uncertain_path = data_dir / "uncertain_review.csv"
    progress_path = data_dir / "progress.json"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run extract.py first.")
        return 1

    # Load paragraphs
    print(f"Loading paragraphs from {input_path}...")
    all_paragraphs = load_paragraphs(input_path)
    total = len(all_paragraphs)
    print(f"Loaded {total} paragraphs")

    # Load or reset progress
    if args.start_fresh and progress_path.exists():
        progress_path.unlink()
        print("Starting fresh — cleared previous progress")

    progress = load_progress(progress_path)
    already_done = set(progress["classified"].keys())
    already_uncertain = {r["paragraph"] for r in progress["uncertain"]}
    already_processed = already_done | already_uncertain

    remaining = [p for p in all_paragraphs if p not in already_processed]
    print(f"Already processed: {len(already_processed)} | Remaining: {len(remaining)}")

    # Estimate time
    calls_per_para  = len(MODELS)
    secs_per_para   = calls_per_para * args.sleep + args.para_pause + 0.5
    eta_mins = (len(remaining) * secs_per_para) / 60
    print(f"Estimated time: {eta_mins:.0f} minutes ({eta_mins/60:.1f} hours)")
    print(f"Sleep: {args.sleep}s between calls, {args.para_pause}s between paragraphs")
    print(f"Models: {', '.join(m[1] for m in MODELS)}")
    print("━" * 60)

    # Write headers if files don't exist yet
    if not output_path.exists():
        with output_path.open("w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "paragraph", "final_label", "confidence",
                "qwen_label",  "qwen_reasoning",
                "gpt_oss_label", "gpt_oss_reasoning",
            ])
            writer.writeheader()

    if not uncertain_path.exists():
        with uncertain_path.open("w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "paragraph",
                "qwen_label",  "qwen_reasoning",
                "gpt_oss_label", "gpt_oss_reasoning",
            ])
            writer.writeheader()

    # Process
    processed_this_run = 0
    errors_this_run    = 0

    for i, paragraph in enumerate(remaining):
        idx = len(already_processed) + i + 1
        short = paragraph[:75].replace("\n", " ")
        print(f"\n[{idx}/{total}] {short}...")

        results: Dict[str, Tuple[str, str]] = {}

        for model_name, model_id, response_format in MODELS:
            cat, reason = call_groq(
                model_id, paragraph, api_key, args.sleep, response_format,
            )
            results[model_name] = (cat, reason)
            status = f"✓ {cat}" if cat in VALID_CATEGORIES else f"✗ {cat}"
            print(f"  {model_name:<10} {status}")

        votes = [results[name][0] for name, _, __ in MODELS]
        final_label, confidence = majority_vote(votes)

        row = {
            "paragraph":         paragraph,
            "qwen_label":        results["qwen"][0],
            "qwen_reasoning":    results["qwen"][1],
            "gpt_oss_label":     results["gpt_oss"][0],
            "gpt_oss_reasoning": results["gpt_oss"][1],
        }

        if confidence == "UNCERTAIN":
            print(f"  → UNCERTAIN (models disagree) — flagged for review")
            with uncertain_path.open("a", encoding="utf8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "paragraph",
                    "qwen_label",  "qwen_reasoning",
                    "gpt_oss_label", "gpt_oss_reasoning",
                ])
                writer.writerow(row)
            progress["uncertain"].append({"paragraph": paragraph, **{k: v[0] for k, v in results.items()}})
        else:
            print(f"  → {final_label} ({confidence})")
            out_row = {**row, "final_label": final_label, "confidence": confidence}
            with output_path.open("a", encoding="utf8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "paragraph", "final_label", "confidence",
                    "qwen_label",  "qwen_reasoning",
                    "gpt_oss_label", "gpt_oss_reasoning",
                ])
                writer.writerow(out_row)
            progress["classified"][paragraph] = {"final_label": final_label, "confidence": confidence}

        # Count errors
        error_count = sum(1 for v in votes if v == "ERROR")
        if error_count >= 2:
            errors_this_run += 1

        processed_this_run += 1

        # Inter-paragraph pause (lets token bucket refill)
        time.sleep(args.para_pause)

        # Save progress every 10 paragraphs
        if processed_this_run % 10 == 0:
            save_progress(progress, progress_path)
            done_total = len(progress["classified"]) + len(progress["uncertain"])
            pct = 100 * done_total / total
            print(f"\n  ── Progress saved: {done_total}/{total} ({pct:.1f}%) ──")

    # Final save
    save_progress(progress, progress_path)

    # Summary
    n_classified = len(progress["classified"])
    n_uncertain  = len(progress["uncertain"])
    print("\n" + "━" * 60)
    print("COMPLETE")
    print(f"  Classified:  {n_classified}")
    print(f"  Uncertain:   {n_uncertain} → {uncertain_path}")
    print(f"  Total:       {n_classified + n_uncertain}/{total}")

    # Category breakdown
    label_counts: Dict[str, int] = {}
    conf_counts:  Dict[str, int] = {}
    for item in progress["classified"].values():
        lbl  = item["final_label"]
        conf = item["confidence"]
        label_counts[lbl]  = label_counts.get(lbl, 0) + 1
        conf_counts[conf]  = conf_counts.get(conf, 0) + 1

    print("\nCategory breakdown:")
    for lbl, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(n_classified, 1)
        print(f"  {lbl:<30} {count:>5}  ({pct:.1f}%)")

    print("\nConfidence breakdown:")
    for conf, count in sorted(conf_counts.items(), key=lambda x: -x[1]):
        print(f"  {conf:<15} {count:>5}")

    print(f"\nOutputs:")
    print(f"  {output_path}")
    print(f"  {uncertain_path}")
    print(f"  {progress_path}  (delete this to start fresh next time)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())