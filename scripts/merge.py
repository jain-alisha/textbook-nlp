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
    parser = argparse.ArgumentParser(description="Merge two classifier arms into final dataset")
    parser.add_argument("--name", required=True, help="Textbook identifier (e.g. cpm_algebra2_hs)")
    # The arms must be from different lineages — agreement between two models of
    # one family measures shared bias, not reliability.
    parser.add_argument("--arms", nargs=2, default=["gemini", "gpt_oss"],
                        metavar=("ARM_A", "ARM_B"),
                        help="Which two arms to merge (default: gemini gpt_oss)")
    args = parser.parse_args()

    arm_a, arm_b = args.arms
    data_dir     = Path("data") / args.name
    path_a       = data_dir / f"{arm_a}_results.csv"
    path_b       = data_dir / f"{arm_b}_results.csv"
    out_path     = data_dir / "classified_results.csv"
    uncertain_path = data_dir / "uncertain_review.csv"

    for arm, path in ((arm_a, path_a), (arm_b, path_b)):
        if not path.exists():
            print(f"ERROR: {path} not found. Run classify_single.py --model {arm} first.")
            return 1

    print(f"Loading {arm_a} results from {path_a}...")
    res_a = load_results(path_a)
    print(f"Loading {arm_b} results from {path_b}...")
    res_b = load_results(path_b)

    # Arm A's paragraph order is canonical (it should have all paragraphs)
    all_paragraphs = list(res_a.keys())
    print(f"Paragraphs in {arm_a}: {len(res_a)}")
    print(f"Paragraphs in {arm_b}: {len(res_b)}")

    only_a = set(res_a.keys()) - set(res_b.keys())
    only_b = set(res_b.keys()) - set(res_a.keys())
    if only_a:
        print(f"  ⚠ {len(only_a)} paragraphs in {arm_a} only — will be marked UNCERTAIN")
    if only_b:
        print(f"  ⚠ {len(only_b)} paragraphs in {arm_b} only — skipped")

    confirmed_rows = []
    uncertain_rows = []

    for para in all_paragraphs:
        a_label, a_reason = res_a.get(para, ("MISSING", ""))
        b_label, b_reason = res_b.get(para, ("MISSING", ""))

        arm_cols = {
            f"{arm_a}_label":     a_label,
            f"{arm_a}_reasoning": a_reason,
            f"{arm_b}_label":     b_label,
            f"{arm_b}_reasoning": b_reason,
        }

        if (a_label in VALID_CATEGORIES and b_label in VALID_CATEGORIES
                and a_label == b_label):
            confirmed_rows.append({
                "paragraph":   para,
                "final_label": a_label,
                "confidence":  "CONFIRMED",
                **arm_cols,
            })
        else:
            uncertain_rows.append({"paragraph": para, **arm_cols})

    arm_fields = [f"{arm_a}_label", f"{arm_a}_reasoning",
                  f"{arm_b}_label", f"{arm_b}_reasoning"]

    # Write classified
    with out_path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paragraph", "final_label", "confidence", *arm_fields])
        writer.writeheader()
        writer.writerows(confirmed_rows)

    # Write uncertain
    with uncertain_path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["paragraph", *arm_fields])
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