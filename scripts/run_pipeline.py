#!/usr/bin/env python3
"""
Full classification pipeline: qwen → wait for quota reset → gpt_oss → merge.

Usage:
  python scripts/run_pipeline.py --name cpm_algebra2

This runs everything hands-off:
  1. qwen on all paragraphs
  2. Waits --quota-wait hours (default 24) for Groq quota to reset
  3. gpt_oss on all paragraphs
  4. Merges both into classified_results.csv + uncertain_review.csv

Options:
  --name          textbook identifier (required)
  --quota-wait    hours to wait between qwen and gpt_oss (default: 24)
  --sleep         seconds between API calls per model (default: 1.5)
  --start-fresh   clear all saved progress and restart from scratch
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'━' * 60}")
    print(f"  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'━' * 60}\n")
    result = subprocess.run(cmd)
    return result.returncode


def wait_with_countdown(hours: float) -> None:
    total_seconds = int(hours * 3600)
    resume_at = datetime.now() + timedelta(seconds=total_seconds)
    print(f"\n{'━' * 60}")
    print(f"  Quota cooldown: waiting {hours:.1f} hours")
    print(f"  Will resume at: {resume_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  (Safe to leave running — Ctrl+C to abort)")
    print(f"{'━' * 60}")

    interval = 300  # print update every 5 minutes
    elapsed = 0
    while elapsed < total_seconds:
        time.sleep(min(interval, total_seconds - elapsed))
        elapsed += interval
        remaining = max(0, total_seconds - elapsed)
        hrs_left = remaining / 3600
        print(f"  [{datetime.now().strftime('%H:%M')}] "
              f"Quota wait: {hrs_left:.1f}h remaining...")

    print(f"\n  Quota wait complete — starting gpt_oss now")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full pipeline: qwen → quota wait → gpt_oss → merge"
    )
    parser.add_argument("--name",        required=True,
                        help="Textbook identifier (e.g. cpm_algebra2)")
    parser.add_argument("--quota-wait",  type=float, default=24.0,
                        help="Hours to wait between qwen and gpt_oss (default: 24)")
    parser.add_argument("--sleep",       type=float, default=1.5,
                        help="Seconds between API calls (default: 1.5)")
    parser.add_argument("--start-fresh", action="store_true",
                        help="Clear all saved progress and restart")
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    classify  = [sys.executable, str(scripts_dir / "classify_single.py")]
    merge     = [sys.executable, str(scripts_dir / "merge.py")]

    # ── Step 1: qwen ──────────────────────────────────────────────────────
    qwen_cmd = classify + ["--name", args.name, "--model", "qwen",
                           "--sleep", str(args.sleep)]
    if args.start_fresh:
        qwen_cmd.append("--start-fresh")

    rc = run(qwen_cmd, "STEP 1/3 — Running qwen classifier")
    if rc != 0:
        print(f"\nERROR: qwen classifier exited with code {rc}. Aborting.")
        return rc

    print(f"\n✓ qwen complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Step 2: quota wait ─────────────────────────────────────────────────
    if args.quota_wait > 0:
        wait_with_countdown(args.quota_wait)
    else:
        print("\nSkipping quota wait (--quota-wait 0)")

    # ── Step 3: gpt_oss ───────────────────────────────────────────────────
    gpt_cmd = classify + ["--name", args.name, "--model", "gpt_oss",
                          "--sleep", str(args.sleep)]
    if args.start_fresh:
        gpt_cmd.append("--start-fresh")

    rc = run(gpt_cmd, "STEP 2/3 — Running gpt_oss classifier")
    if rc != 0:
        print(f"\nERROR: gpt_oss classifier exited with code {rc}. Aborting.")
        return rc

    print(f"\n✓ gpt_oss complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Step 4: merge ──────────────────────────────────────────────────────
    merge_cmd = merge + ["--name", args.name]
    rc = run(merge_cmd, "STEP 3/3 — Merging results")
    if rc != 0:
        print(f"\nERROR: merge exited with code {rc}.")
        return rc

    print(f"\n{'━' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results:  data/{args.name}/classified_results.csv")
    print(f"  Review:   data/{args.name}/uncertain_review.csv")
    print(f"{'━' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
