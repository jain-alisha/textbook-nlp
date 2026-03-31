"""
Run the 3-model Groq ensemble classifier on a textbook's paragraphs.

Usage:
    python scripts/classify.py --name saxon_course1
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODELS = [
    {"id": "llama-3.3-70b-versatile",  "key": "llama",   "response_format": {"type": "json_object"}},
    {"id": "qwen/qwen3-32b",            "key": "qwen",    "response_format": {"type": "json_object"}},
    {"id": "openai/gpt-oss-120b",       "key": "gpt_oss", "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "INCORRECT_TO_CORRECT",
                            "COMPARE_AND_CONTRAST",
                            "EXPLICIT_ERROR_DETECTION",
                            "COMMON_ERROR_ALERT",
                            "NA",
                        ],
                    },
                },
                "required": ["reasoning", "category"],
                "additionalProperties": False,
            },
        },
    }},
]

SYSTEM_PROMPT = """
You are an expert educational researcher. Your task is to classify
paragraphs from a high school math textbook into ONE of these categories
based on whether the paragraph uses error-focused pedagogy.

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
  Signals: "students often...", "a common mistake is...", "be careful not to...", "do not confuse X with Y".
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

VALID_CATEGORIES = {
    "INCORRECT_TO_CORRECT",
    "COMPARE_AND_CONTRAST",
    "EXPLICIT_ERROR_DETECTION",
    "COMMON_ERROR_ALERT",
    "NA",
}

BACKOFF_STEPS = [15, 30, 60, 120]
INTER_MODEL_SLEEP = 1.2


def call_model(model: dict, paragraph: str) -> dict:
    """Call a single Groq model and return {"label": ..., "reasoning": ...}."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model["id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": paragraph},
        ],
        "max_tokens": 500,
        "temperature": 0.1,
        "response_format": model["response_format"],
    }

    backoff_steps = list(BACKOFF_STEPS)
    attempt = 0
    while True:
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = backoff_steps[min(attempt, len(backoff_steps) - 1)]
                print(f"    429 rate limit ({model['key']}), sleeping {wait}s...")
                time.sleep(wait)
                attempt += 1
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            label = parsed.get("category", "").strip().upper()
            reasoning = parsed.get("reasoning", "").strip()
            if label not in VALID_CATEGORIES:
                label = "NA"
            return {"label": label, "reasoning": reasoning}
        except requests.exceptions.HTTPError as e:
            wait = backoff_steps[min(attempt, len(backoff_steps) - 1)]
            print(f"    HTTP error ({model['key']}): {e}, sleeping {wait}s...")
            time.sleep(wait)
            attempt += 1
        except Exception as e:
            return {"label": "ERROR", "reasoning": str(e)}


def majority_vote(results: dict) -> tuple[str, str]:
    """
    Returns (final_label, confidence) where confidence is one of:
    CONFIRMED (3/3), MAJORITY (2/3), UNCERTAIN (all disagree).
    """
    labels = [results[m["key"]]["label"] for m in MODELS]
    counts: dict[str, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1

    best_label = max(counts, key=lambda k: counts[k])
    best_count = counts[best_label]

    if best_count == 3:
        return best_label, "CONFIRMED"
    if best_count == 2:
        return best_label, "MAJORITY"
    return labels[0], "UNCERTAIN"


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"classified": {}}


def save_progress(progress_path: Path, progress: dict) -> None:
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify textbook paragraphs with 3-model Groq ensemble.")
    parser.add_argument("--name", required=True, help="Textbook identifier (e.g. saxon_course1).")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set in environment / .env", file=sys.stderr)
        sys.exit(1)

    data_dir = Path("data") / args.name
    paragraphs_path = data_dir / "paragraphs.csv"
    classified_path = data_dir / "classified_results.csv"
    uncertain_path = data_dir / "uncertain_review.csv"
    progress_path = data_dir / "progress.json"

    if not paragraphs_path.exists():
        print(f"Error: {paragraphs_path} not found. Run extract.py first.", file=sys.stderr)
        sys.exit(1)

    # Load paragraphs.
    with paragraphs_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        paragraphs = [row["paragraph"] for row in reader]

    # Load progress (for resumability).
    progress = load_progress(progress_path)
    already_done = set(progress["classified"].keys())

    # Load existing classified rows so we can append correctly.
    classified_rows: list[dict] = []
    if classified_path.exists():
        with classified_path.open("r", encoding="utf-8") as f:
            classified_rows = list(csv.DictReader(f))

    uncertain_rows: list[dict] = []
    if uncertain_path.exists():
        with uncertain_path.open("r", encoding="utf-8") as f:
            uncertain_rows = list(csv.DictReader(f))

    classified_fieldnames = [
        "paragraph", "final_label", "confidence",
        "llama_label", "llama_reasoning",
        "qwen_label", "qwen_reasoning",
        "gpt_oss_label", "gpt_oss_reasoning",
    ]
    uncertain_fieldnames = [
        "paragraph",
        "llama_label", "llama_reasoning",
        "qwen_label", "qwen_reasoning",
        "gpt_oss_label", "gpt_oss_reasoning",
    ]

    pending = [p for p in paragraphs if p not in already_done]
    print(f"{len(paragraphs)} total paragraphs, {len(already_done)} already done, {len(pending)} to classify.")

    for i, para in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] classifying: {para[:80]!r}...")

        model_results = {}
        for model in MODELS:
            result = call_model(model, para)
            model_results[model["key"]] = result
            print(f"    {model['key']}: {result['label']}")
            time.sleep(INTER_MODEL_SLEEP)

        final_label, confidence = majority_vote(model_results)

        # Record in progress.
        progress["classified"][para] = {
            "final_label": final_label,
            "confidence": confidence,
        }
        save_progress(progress_path, progress)

        row = {
            "paragraph": para,
            "final_label": final_label,
            "confidence": confidence,
            "llama_label": model_results["llama"]["label"],
            "llama_reasoning": model_results["llama"]["reasoning"],
            "qwen_label": model_results["qwen"]["label"],
            "qwen_reasoning": model_results["qwen"]["reasoning"],
            "gpt_oss_label": model_results["gpt_oss"]["label"],
            "gpt_oss_reasoning": model_results["gpt_oss"]["reasoning"],
        }

        if confidence == "UNCERTAIN":
            uncertain_rows.append({k: row[k] for k in uncertain_fieldnames})
            print(f"    -> UNCERTAIN (written to uncertain_review.csv)")
        else:
            classified_rows.append(row)
            print(f"    -> {final_label} ({confidence})")

    # Write outputs.
    with classified_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=classified_fieldnames)
        writer.writeheader()
        writer.writerows(classified_rows)

    with uncertain_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=uncertain_fieldnames)
        writer.writeheader()
        writer.writerows(uncertain_rows)

    total = len(classified_rows) + len(uncertain_rows)
    print(f"\nDone. {len(classified_rows)} classified, {len(uncertain_rows)} uncertain.")
    print(f"  -> {classified_path}")
    print(f"  -> {uncertain_path}")
    print(f"  -> {progress_path}")


if __name__ == "__main__":
    main()
