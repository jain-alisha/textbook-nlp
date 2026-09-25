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
from typing import NamedTuple, Dict, List, Tuple

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
        # Pinned to 3.8 rather than 2.5-flash, which is closed to new API projects
        # and so cannot be re-run reproducibly. Changed while zero Gemini labels
        # existed; once this arm has written labels, changing it splits the corpus
        # the way the retired qwen3-32b did.
        "model_id": "gemini-3.8-flash",
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
                        "reasoning":  {"type": "string"},
                        "category":   {"type": "string"},
                        "confidence": {"type": "string"},
                        "considered": {"type": "string"},
                    },
                    "required": ["reasoning", "category"],
                },
            },
        },
    },
    # Stage-2 verification arm. Local, free, and from a different vendor and model
    # family than the Gemini screener, which is what the independence requirement
    # actually asks for. It is only viable because stage 2 sees a small subset:
    # measured at 11.7s/paragraph, a full census would take ~80 hours.
    "qwen_local": {
        "model_id": "qwen3:14b",
        "provider": "ollama",
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
  vocabulary, reflections. When in doubt, use NA — but record the doubt in the
  "confidence" and "considered" fields described below rather than discarding it.

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
{"reasoning": "...", "category": "CATEGORY_NAME",
 "confidence": "high" | "medium" | "low",
 "considered": "CATEGORY_NAME" | null}

Use ONLY one of: INCORRECT_TO_CORRECT, COMPARE_AND_CONTRAST,
EXPLICIT_ERROR_DETECTION, COMMON_ERROR_ALERT, NA

Still prefer NA when the paragraph merely asks a student to solve, check or
verify their own work. But DO NOT hide your uncertainty: if you weighed an error
category before settling on NA, name that category in "considered" and set
"confidence" to reflect how close the call was. A borderline case recorded as
low-confidence NA with "considered" filled in is handled correctly downstream; a
borderline case reported as confident NA is lost.
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


def load_cache(path: Path, model_id: str) -> Dict[str, Verdict]:
    """Read the append-only cache, keeping the last entry for each key."""
    if not path.exists():
        return {}
    cached: Dict[str, Verdict] = {}
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
                cached[rec["key"]] = Verdict(
                    rec["label"], rec.get("reasoning", ""),
                    # Entries written before the uncertainty fields existed are
                    # "unknown", which routes them to the second arm rather than
                    # letting a missing field read as confident.
                    rec.get("confidence", "unknown"), rec.get("considered", ""))
    return cached


def append_cache(path: Path, key: str, model_id: str,
                 paragraph: str, verdict: Verdict) -> None:
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps({
            "key": key,
            "model_id": model_id,
            "paragraph": paragraph,
            "label": verdict.category,
            "reasoning": verdict.reasoning,
            "confidence": verdict.confidence,
            "considered": verdict.considered,
        }, ensure_ascii=False) + "\n")


class Verdict(NamedTuple):
    """A label plus the uncertainty that produced it.

    `confidence` and `considered` are what make two-stage routing possible: a
    low-confidence NA, or an NA that weighed an error category, is a boundary case
    the second arm must see. Collapsing those into a bare "NA" — which the prompt
    used to instruct — deletes precisely the cases where the screener is weakest.
    """
    category: str
    reasoning: str
    confidence: str = "unknown"
    considered: str = ""


def parse_response(raw: str) -> Verdict:
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    try:
        data = json.loads(cleaned)
        category = str(data.get("category", "PARSE_ERROR")).strip().upper()
        reasoning = str(data.get("reasoning", "")).strip()
        confidence = str(data.get("confidence") or "unknown").strip().lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "unknown"
        considered = str(data.get("considered") or "").strip().upper()
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
        considered = label_map.get(considered, considered)
        if considered not in VALID_CATEGORIES or considered == category:
            considered = ""
        return Verdict(category, reasoning, confidence, considered)
    except json.JSONDecodeError:
        for cat in VALID_CATEGORIES:
            if cat in raw.upper():
                # Recovered from malformed JSON, so the uncertainty fields were
                # never parsed: treat it as unknown confidence, which routes the
                # paragraph to the second arm rather than trusting it.
                return Verdict(cat, raw.strip()[:300], "unknown", "")
        return Verdict("PARSE_ERROR", raw.strip()[:300], "unknown", "")


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
                max_retries: int = 6) -> Verdict:
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=0.1,
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {"reasoning": {"type": "string"},
                           "category": {"type": "string"},
                           "confidence": {"type": "string"},
                           "considered": {"type": "string"}},
            # confidence/considered are not required: the schema would otherwise
            # force the model to invent a "considered" category for every clear-cut
            # paragraph, which is exactly the signal we are trying to keep honest.
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
                return Verdict("ERROR", str(e)[:200], "unknown", "")
            time.sleep(10 * attempts)


OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/chat"


def call_ollama(model_id: str, paragraph: str, sleep: float,
                max_retries: int = 4) -> Verdict:
    """Local inference via Ollama. No key, no quota, no cost.

    `think: false` matters: qwen3 emits long reasoning traces by default, which
    triples latency for a task whose output is one label.
    """
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Paragraph to classify:\n"""{paragraph[:8000]}"""'},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    attempts = 0
    while True:
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            time.sleep(sleep)
            return parse_response(raw)
        except requests.exceptions.ConnectionError as e:
            # A stopped Ollama server would otherwise look like thousands of
            # classification failures rather than one fixable problem.
            raise RuntimeError(
                f"Cannot reach Ollama at {OLLAMA_URL}. Is `ollama serve` running?"
            ) from e
        except Exception as e:
            attempts += 1
            print(f"    Ollama error ({attempts}/{max_retries}): {str(e)[:120]}")
            if attempts >= max_retries:
                return Verdict("ERROR", str(e)[:200], "unknown", "")
            time.sleep(5 * attempts)


def call_groq(
    model_id: str,
    paragraph: str,
    api_key: str,
    sleep: float,
    response_format: dict,
    max_retries: int = 8,
) -> Verdict:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Paragraph to classify:\n"""{paragraph}"""'},
        ],
        "temperature": 0.1,
        "response_format": response_format,
        # 500 was too small once the schema grew to four fields: gpt-oss-120b emits
        # its own reasoning trace *before* the JSON document, so a long trace leaves
        # too little budget to close the object and Groq rejects the whole call with
        # 400 json_validate_failed ("max completion tokens reached before generating
        # a valid document"). That is a hard 400, not a transient error, so it is not
        # retried -- the paragraph just fails. Same failure mode that capped
        # stitch.py's merge replies earlier in this project. Headroom is nearly free
        # here: unused completion tokens are not billed.
        "max_tokens": 3000,
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
                return Verdict("ERROR", "timeout", "unknown", "")
            time.sleep(10 * non_429_attempts)
        except requests.exceptions.HTTPError as e:
            non_429_attempts += 1
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text[:200]
            print(f"    HTTP {e.response.status_code} ({non_429_attempts}/{max_retries}): {body}")
            if e.response.status_code == 400:
                return Verdict("ERROR", f"400: {body}", "unknown", "")
            if non_429_attempts >= max_retries:
                return Verdict("ERROR", str(e)[:200], "unknown", "")
            time.sleep(10 * non_429_attempts)
        except Exception as e:
            non_429_attempts += 1
            print(f"    Error ({non_429_attempts}/{max_retries}): {str(e)[:100]}")
            if non_429_attempts >= max_retries:
                return Verdict("ERROR", str(e)[:200], "unknown", "")
            time.sleep(10 * non_429_attempts)


def main():
    parser = argparse.ArgumentParser(description="Single-model classifier for parallel runs")
    parser.add_argument("--name",        required=True, help="Textbook identifier")
    parser.add_argument("--model",       required=True,
                        choices=sorted(MODEL_CONFIGS),
                        help="Which arm to run: " + " | ".join(sorted(MODEL_CONFIGS)))
    parser.add_argument("--paragraphs",  default=None,
                        help="Paragraphs CSV filename (default: paragraphs.csv)")
    parser.add_argument("--out",         default=None,
                        help="Output basename, without .csv (default: <model>_results). "
                             "Set this for a stage-2 run so it cannot overwrite a "
                             "census run by the same arm.")
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
    output_path   = data_dir / f"{args.out or args.model + '_results'}.csv"
    # The cache is keyed on model + prompt + paragraph, so it is shared across
    # runs of the same arm regardless of --out: a stage-2 run reuses any census
    # label for the same paragraph instead of paying for it twice.
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

    # A stage-2 worklist from route.py carries a `tier` column saying why each
    # paragraph was routed here. Carrying it through to the output is what lets
    # stage2_report.py compute per-stratum agreement and the recall bound.
    tier_of = {}
    with input_path.open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r.get("tier") and (r.get("paragraph") or "").strip():
                tier_of[r["paragraph"].strip()] = r["tier"]
    if tier_of:
        from collections import Counter
        print(f"Stage-2 worklist detected: {dict(Counter(tier_of.values()))}")

    cached = load_cache(cache_path, model_id)
    to_call = sum(1 for p in all_paragraphs if cache_key(model_id, p) not in cached)

    eta_mins = to_call * (args.sleep + 1.0) / 60
    print(f"Cached: {total - to_call} | Need API call: {to_call}")
    print(f"Model: {model_id}")
    print(f"Estimated time: {eta_mins:.0f} min ({eta_mins/60:.1f} hrs)")
    print("━" * 60)

    # The cache is the durable store; this CSV is a derived view rebuilt each
    # run, so every input row appears exactly once even when texts repeat.
    fieldnames = ["paragraph", f"{args.model}_label", f"{args.model}_reasoning",
                  f"{args.model}_confidence", f"{args.model}_considered"]
    if tier_of:
        fieldnames.append("tier")
    with output_path.open("w", encoding="utf8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    processed = 0
    errors = 0

    for idx, paragraph in enumerate(all_paragraphs, start=1):
        key = cache_key(model_id, paragraph)
        hit = cached.get(key)

        if hit is not None:
            v = hit
        else:
            short = paragraph[:80].replace("\n", " ")
            print(f"\n[{idx}/{total}] {short}...")
            if provider == "gemini":
                try:
                    v = call_gemini(model_id, paragraph, args.sleep)
                except DailyQuotaExhausted as e:
                    from extract import _quota_diagnosis
                    print(f"\nERROR: Gemini daily quota exhausted after {processed} new "
                          f"paragraphs. Cached work is saved.", file=sys.stderr)
                    print(_quota_diagnosis(str(e)), file=sys.stderr)
                    return 3
            elif provider == "ollama":
                v = call_ollama(model_id, paragraph, args.sleep)
            else:
                v = call_groq(model_id, paragraph, api_key, args.sleep,
                              response_format)
            status = f"✓ {v.category}" if v.category in VALID_CATEGORIES else f"✗ {v.category}"
            extra = f" [{v.confidence}{'/' + v.considered if v.considered else ''}]"
            print(f"  {args.model:<10} {status}{extra}")

            if v.category == "ERROR":
                # Caching a failure would make a re-run skip the paragraph and
                # bake the error into the results permanently.
                errors += 1
            else:
                append_cache(cache_path, key, model_id, paragraph, v)
                cached[key] = v
            processed += 1

            if processed % 10 == 0:
                pct = 100 * idx / total
                print(f"\n  ── {idx}/{total} ({pct:.1f}%) | new: {processed} | errors: {errors} ──")

        row = {
            "paragraph": paragraph,
            f"{args.model}_label": v.category,
            f"{args.model}_reasoning": v.reasoning,
            f"{args.model}_confidence": v.confidence,
            f"{args.model}_considered": v.considered,
        }
        if tier_of:
            row["tier"] = tier_of.get(paragraph, "")
        with output_path.open("a", encoding="utf8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

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
        "output_file": output_path.name,
        "rows": total,
        "tiers": dict(__import__("collections").Counter(tier_of.values())) or None,
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