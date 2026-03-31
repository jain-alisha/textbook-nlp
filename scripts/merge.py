#!/usr/bin/env python3
"""
Merge single-model results into a final classified dataset.

Run AFTER both classify_single.py runs are complete:
  python scripts/merge.py --name cpm_algebra2

Reads:
  data/<n>/qwen_results.csv
  data/<n>/gpt_oss_results.csv

Writes:
  data/<n>/classified_results.csv   — paragraphs where both models agree
  data/<n>/uncertain_review.csv     — paragraphs where models disagree or either errored
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

VALID_CATEGORIES = {
    "INCORRECT_TO_CORRECT",
    "COMPARE_AND_CONTRAST",
    "EXPLICIT_ERROR_DETECTION",
    "COMMON_ERROR_ALERT",
    "NA",
}


def load_results(path: Path) -> Dict[str, Tuple[str, str]]:
    """Load model results CSV into {paragraph: (label, reasoning)}."""
    results = {}
    with path.open(encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            para = (row.get("paragraph") or "").strip()
            if not para:
                continue
            # Column names are dynamic: qwen_label / gpt_oss_label etc.
            label = ""
            reasoning = ""
            for key, val in row.items():
                if key.endswith("_label"):
                    label = (val or "").strip()
                elif key.endswith("_reasoning"):
                    reasoning = (val or "").strip()
            results[para] = (label, reasoning)
    return results


def main():
    parser = argparse.ArgumentParser(description="Merge qwen + gpt_oss results into final dataset")
    parser.add_argument("--name", required=True, help="Textbook identifier (e.g. cpm_algebra2)")
    args = parser.parse_args()

    data_dir     = Path("data") / args.name
    qwen_path    = data_dir / "qwen_results.csv"
    gpt_oss_path = data_dir / "gpt_oss_results.csv"
    out_path     = data_dir / "classified_results.csv"
    uncertain_path = data_dir / "uncertain_review.csv"

    if not qwen_path.exists():
        print(f"ERROR: {qwen_path} not found. Run classify_single.py --model qwen first.")
        return 1
    if not gpt_oss_path.exists():
        print(f"ERROR: {gpt_oss_path} not found. Run classify_single.py --model gpt_oss first.")
        return 1

    print(f"Loading qwen results from {qwen_path}...")
    qwen = load_results(qwen_path)
    print(f"Loading gpt_oss results from {gpt_oss_path}...")
    gpt_oss = load_results(gpt_oss_path)

    # Use qwen's paragraph order as canonical (it should have all paragraphs)
    all_paragraphs = list(qwen.keys())
    print(f"Paragraphs in qwen: {len(qwen)}")
    print(f"Paragraphs in gpt_oss: {len(gpt_oss)}")

    # Paragraphs only in one model
    only_qwen    = set(qwen.keys()) - set(gpt_oss.keys())
    only_gpt_oss = set(gpt_oss.keys()) - set(qwen.keys())
    if only_qwen:
        print(f"  ⚠ {len(only_qwen)} paragraphs in qwen only — will be marked UNCERTAIN")
    if only_gpt_oss:
        print(f"  ⚠ {len(only_gpt_oss)} paragraphs in gpt_oss only — skipped")

    confirmed_rows = []
    uncertain_rows = []

    for para in all_paragraphs:
        q_label,  q_reason  = qwen.get(para, ("MISSING", ""))
        g_label,  g_reason  = gpt_oss.get(para, ("MISSING", ""))

        q_valid = q_label in VALID_CATEGORIES
        g_valid = g_label in VALID_CATEGORIES

        if q_valid and g_valid and q_label == g_label:
            confirmed_rows.append({
                "paragraph":       para,
                "final_label":     q_label,
                "confidence":      "CONFIRMED",
                "qwen_label":      q_label,
                "qwen_reasoning":  q_reason,
                "gpt_oss_label":   g_label,
                "gpt_oss_reasoning": g_reason,
            })
        else:
            uncertain_rows.append({
                "paragraph":       para,
                "qwen_label":      q_label,
                "qwen_reasoning":  q_reason,
                "gpt_oss_label":   g_label,
                "gpt_oss_reasoning": g_reason,
            })

    # Write classified
    with out_path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paragraph", "final_label", "confidence",
            "qwen_label", "qwen_reasoning",
            "gpt_oss_label", "gpt_oss_reasoning",
        ])
        writer.writeheader()
        writer.writerows(confirmed_rows)

    # Write uncertain
    with uncertain_path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paragraph",
            "qwen_label", "qwen_reasoning",
            "gpt_oss_label", "gpt_oss_reasoning",
        ])
        writer.writeheader()
        writer.writerows(uncertain_rows)

    total = len(confirmed_rows) + len(uncertain_rows)
    print("\n" + "━" * 60)
    print("MERGE COMPLETE")
    print(f"  Confirmed:  {len(confirmed_rows)}/{total} ({100*len(confirmed_rows)//max(total,1)}%)")
    print(f"  Uncertain:  {len(uncertain_rows)}/{total}")
    print(f"\nOutputs:")
    print(f"  {out_path}")
    print(f"  {uncertain_path}")

    # Category breakdown on confirmed
    counts: Dict[str, int] = {}
    for row in confirmed_rows:
        lbl = row["final_label"]
        counts[lbl] = counts.get(lbl, 0) + 1
    print("\nCategory breakdown (confirmed only):")
    for lbl, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(len(confirmed_rows), 1)
        print(f"  {lbl:<30} {count:>5}  ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())