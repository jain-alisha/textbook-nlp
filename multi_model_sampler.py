#!/usr/bin/env python3
"""Sample textbook paragraphs and probe multiple inference providers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import requests
from dotenv import load_dotenv

PROMPT_TEMPLATE = (
    "You are evaluating math textbook paragraphs for error-focused pedagogy. "
    "Classify the paragraph into one of these labels and explain briefly: "
    "[Incorrect-to-correct revision tasks, Compare-and-contrast examples, "
    "Explicit error detection, Common error alerts, N/A]. Respond with a short JSON "
    "object like {\"category\": \"...\", \"rationale\": \"...\"}."
)


def load_paragraphs(csv_path: Path) -> List[str]:
    paragraphs: List[str] = []
    with csv_path.open(encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get("paragraph") or "").strip()
            if text:
                paragraphs.append(text)
    if not paragraphs:
        raise RuntimeError(f"No paragraphs found in {csv_path}")
    return paragraphs


def middle_block(paragraphs: List[str], size: int) -> tuple[List[str], int]:
    total = len(paragraphs)
    if size >= total:
        return paragraphs, 0
    start = max(0, total // 2 - size // 2)
    end = min(total, start + size)
    return paragraphs[start:end], start


def run_groq(paragraph: str, timeout: int = 45) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment")
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"),
        "messages": [
            {"role": "system", "content": PROMPT_TEMPLATE},
            {"role": "user", "content": paragraph},
        ],
        "temperature": 0.1,
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def run_cloudflare(paragraph: str, timeout: int = 45) -> str:
    api_key = os.getenv("CLOUDFLARE_AI_API_KEY")
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if not api_key or not account_id:
        raise RuntimeError("Cloudflare API key/account id missing")
    model = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-2-7b-chat-int8")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    payload = {
        "messages": [
            {"role": "system", "content": PROMPT_TEMPLATE},
            {"role": "user", "content": paragraph},
        ]
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"Cloudflare error: {data.get('errors')}")
    result = data.get("result") or {}
    # Responses may appear under different keys depending on the model
    if isinstance(result, dict):
        if "response" in result:
            return result["response"].strip()
        if "output" in result:
            return result["output"].strip()
    return json.dumps(result)


def run_huggingface(paragraph: str, timeout: int = 45) -> str:
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing HUGGINGFACE_API_KEY")
    model = os.getenv("HUGGINGFACE_MODEL", "facebook/bart-large-mnli")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": {
            "premise": paragraph,
            "hypothesis": "This paragraph is about identifying or correcting math errors.",
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code == 503:
        # Model is loading; let caller retry
        raise RuntimeError("Hugging Face model loading, retry later")
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        best = max(data, key=lambda item: item.get("score", 0))
        return f"{best.get('label')} (score={best.get('score'):.3f})"
    return json.dumps(data)


def run_openrouter(paragraph: str, timeout: int = 45) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
        "messages": [
            {"role": "system", "content": PROMPT_TEMPLATE},
            {"role": "user", "content": paragraph},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE", "https://example.com"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "Textbook Analyzer"),
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def run_ollama(paragraph: str, timeout: int = 45) -> str:
    base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3")
    url = f"{base}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_TEMPLATE},
            {"role": "user", "content": paragraph},
        ],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()
    return json.dumps(data)


PROVIDERS: Dict[str, Callable[[str], str]] = {
    "groq": run_groq,
    "cloudflare": run_cloudflare,
    "huggingface": run_huggingface,
    "openrouter": run_openrouter,
    "ollama": run_ollama,
}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe multiple inference providers")
    parser.add_argument("--input", type=Path, default=Path("paragraph_clusters.csv"))
    parser.add_argument("--output", type=Path, default=Path("multi_model_results.jsonl"))
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between provider calls")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls and just dump sample")
    args = parser.parse_args(argv)

    load_dotenv()

    paragraphs = load_paragraphs(args.input)
    sample, start_index = middle_block(paragraphs, args.sample_size)
    print(f"Total paragraphs: {len(paragraphs)}")
    print(f"Sampling indices {start_index}–{start_index + len(sample) - 1}")

    results: List[Dict[str, Any]] = []

    for offset, paragraph in enumerate(sample):
        global_index = start_index + offset
        record: Dict[str, Any] = {"index": global_index, "paragraph": paragraph, "responses": {}}
        print(f"\nParagraph #{global_index}")
        if args.dry_run:
            results.append(record)
            continue

        for name, func in PROVIDERS.items():
            try:
                print(f"  -> querying {name}…", end="")
                response = func(paragraph)
                print(" done")
            except Exception as exc:  # noqa: BLE001
                response = f"ERROR: {exc}"
                print(f" failed ({exc})")
            record["responses"][name] = response
            if args.sleep:
                time.sleep(args.sleep)
        results.append(record)

    with args.output.open("w", encoding="utf8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} rows to {args.output}")
    if args.dry_run:
        print("(Dry run: no API calls were issued.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
