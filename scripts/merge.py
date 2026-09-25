#!/usr/bin/env python3
"""
Build the final per-book classified dataset.

Two modes:

  two-stage (default) — Gemini screened every paragraph; a second arm verified a
      routed subset (see route.py). Every paragraph gets a final label, and a
      `verification` column says whether two arms saw it or only one. The majority
      of the corpus is confident-NA that was never sampled into the control, so it
      carries Gemini's label unverified — that is a real property of the design and
      it is recorded per row rather than left implicit in the routing arithmetic.

  census — the older design, two arms over every paragraph. Kept for corpus slices
      that were labelled that way.

Reads (two-stage):
  data/<n>/gemini_results.csv        stage 1, every paragraph
  data/<n>/qwen_local_stage2.csv     stage 2, routed subset, carries `tier`
  data/<n>/stage2_strata.json        pool sizes written by route.py

Writes:
  data/<n>/classified_results.csv    every paragraph, with label + verification
  data/<n>/uncertain_review.csv      dual-arm disagreements only
  data/<n>/dataset_manifest.json     row counts by status/verification/tier

Usage:
  python scripts/merge.py --name ck12_algebra1_hs
  python scripts/merge.py --name cpm_algebra2_hs --mode census --arms gemini gpt_oss
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

VALID_CATEGORIES = {
    "INCORRECT_TO_CORRECT",
    "COMPARE_AND_CONTRAST",
    "EXPLICIT_ERROR_DETECTION",
    "COMMON_ERROR_ALERT",
    "NA",
}

BAD = {"", "ERROR", "PARSE_ERROR", "MISSING"}


def load_results(path: Path, arm: str | None = None) -> Dict[str, dict]:
    """Load a results CSV into {paragraph: {label, reasoning, confidence, considered, tier}}."""
    out: Dict[str, dict] = {}
    with path.open(encoding="utf8") as f:
        for row in csv.DictReader(f):
            para = (row.get("paragraph") or "").strip()
            if not para:
                continue
            rec = {"label": "", "reasoning": "", "confidence": "", "considered": "",
                   "tier": (row.get("tier") or "").strip()}
            for key, val in row.items():
                v = (val or "").strip()
                if arm and not key.startswith(f"{arm}_"):
                    continue
                if key.endswith("_label"):
                    rec["label"] = v
                elif key.endswith("_reasoning"):
                    rec["reasoning"] = v
                elif key.endswith("_confidence"):
                    rec["confidence"] = v
                elif key.endswith("_considered"):
                    rec["considered"] = v
            out[para] = rec
    return out


def source_index(data_dir: Path, arm: str) -> tuple[str, Dict[str, int]]:
    """Map each paragraph to its 1-based row number in the extraction it came from.

    Row number in paragraphs.csv is the only stable per-paragraph identifier the
    pipeline has — it is what lets a finding be located in the book again.
    """
    manifest = data_dir / f"{arm}_manifest.json"
    src_name = "paragraphs.csv"
    if manifest.exists():
        try:
            src_name = json.loads(manifest.read_text()).get("paragraphs_file") or src_name
        except json.JSONDecodeError:
            pass
    src = data_dir / src_name
    idx: Dict[str, int] = {}
    if src.exists():
        for i, r in enumerate(csv.DictReader(src.open(encoding="utf8")), start=1):
            para = (r.get("paragraph") or "").strip()
            if para and para not in idx:   # first occurrence wins for repeated text
                idx[para] = i
    return src_name, idx


def write_findings(data_dir: Path, book: str, rows: list[dict], src_name: str) -> Path:
    """Every non-NA paragraph, identified by book and row number.

    Written as its own file because this is the output the research question is
    actually about; classified_results.csv is ~99% NA and is the audit trail.
    """
    findings = [r for r in rows if r["final_label"] not in ("NA", *BAD)]
    findings.sort(key=lambda r: (r["para_num"] if isinstance(r["para_num"], int) else 1 << 30))
    cols = ["book", "para_num", "final_label", "status", "verification", "tier",
            "source_file", "paragraph"]
    extra = [c for c in rows[0] if c.endswith(("_label", "_reasoning", "_confidence",
                                              "_considered"))] if rows else []
    path = data_dir / "findings.csv"
    with path.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + extra, extrasaction="ignore")
        w.writeheader()
        for r in findings:
            w.writerow({**r, "book": book, "source_file": src_name})
    return path


def check_staleness(data_dir: Path, arm: str, labelled: set[str]) -> list[str]:
    """Compare a results file's paragraph set against the extraction it claims.

    Re-extracting a book replaces its paragraphs wholesale, which leaves old label
    files describing text that no longer exists. Nothing in the CSVs themselves
    reveals that, so it is checked here rather than trusted to memory.
    """
    manifest = data_dir / f"{arm}_manifest.json"
    src_name = "paragraphs.csv"
    if manifest.exists():
        try:
            src_name = json.loads(manifest.read_text()).get("paragraphs_file") or src_name
        except json.JSONDecodeError:
            pass
    src = data_dir / src_name
    if not src.exists():
        return [f"{arm}: source {src_name} is missing; cannot verify freshness"]

    current = {(r.get("paragraph") or "").strip()
               for r in csv.DictReader(src.open(encoding="utf8"))}
    current.discard("")
    missing = current - labelled          # extracted but never labelled
    orphan = labelled - current           # labelled but no longer in the book
    problems = []
    if orphan:
        problems.append(
            f"{arm}: {len(orphan)} labelled paragraphs are absent from {src_name} "
            f"— these labels describe discarded text")
    if missing:
        problems.append(
            f"{arm}: {len(missing)} paragraphs in {src_name} have no {arm} label "
            f"— stage 1 is incomplete")
    return problems


def merge_two_stage(data_dir: Path, args) -> int:
    s1_path = data_dir / f"{args.stage1_arm}_results.csv"
    s2_path = data_dir / (args.stage2 or f"{args.stage2_arm}_stage2.csv")
    strata_path = data_dir / "stage2_strata.json"

    if not s1_path.exists():
        print(f"ERROR: {s1_path} not found. Run stage 1:\n"
              f"  python scripts/classify_single.py --name {args.name} "
              f"--model {args.stage1_arm}")
        return 1
    if not s2_path.exists():
        print(f"ERROR: {s2_path} not found. Run route.py then stage 2.")
        return 1

    s1 = load_results(s1_path, args.stage1_arm)
    s2 = load_results(s2_path, args.stage2_arm)
    strata = json.loads(strata_path.read_text()) if strata_path.exists() else {}

    problems = check_staleness(data_dir, args.stage1_arm, set(s1))
    orphan2 = set(s2) - set(s1)
    if orphan2:
        problems.append(f"{args.stage2_arm}: {len(orphan2)} stage-2 rows are not in "
                        f"stage 1 — the worklist predates the current stage-1 run")
    if problems:
        print("STALENESS CHECK FAILED:")
        for p in problems:
            print(f"  ✗ {p}")
        if not args.allow_stale:
            print("\nRefusing to build a dataset from mismatched inputs. Re-run the "
                  "affected stage, or pass --allow-stale if you know why this is fine.")
            return 2
        print("\n  --allow-stale given; continuing anyway.")

    src_name, para_idx = source_index(data_dir, args.stage1_arm)

    rows, uncertain = [], []
    for para, a in s1.items():
        b = s2.get(para)
        tier = (b or {}).get("tier") or "C_unsampled"
        a_lab = a["label"]

        if b is None:
            # Never routed to stage 2: confident NA outside the control sample.
            status = "UNVERIFIED" if a_lab in VALID_CATEGORIES else "ERROR"
            verification, final, b_lab, b_reason = "single_arm", a_lab, "", ""
        else:
            b_lab, b_reason = b["label"], b["reasoning"]
            verification = "dual_arm"
            if a_lab in VALID_CATEGORIES and b_lab in VALID_CATEGORIES:
                if a_lab == b_lab:
                    status, final = "CONFIRMED", a_lab
                else:
                    # Keep the screener's label so the dataset stays complete, but
                    # flag it: these rows should not be treated as findings.
                    status, final = "UNCERTAIN", a_lab
            else:
                status, final = "ERROR", a_lab

        row = {
            "book": args.name,
            "para_num": para_idx.get(para, ""),
            "paragraph": para,
            "final_label": final,
            "status": status,
            "verification": verification,
            "tier": tier,
            f"{args.stage1_arm}_label": a_lab,
            f"{args.stage1_arm}_confidence": a["confidence"],
            f"{args.stage1_arm}_considered": a["considered"],
            f"{args.stage1_arm}_reasoning": a["reasoning"],
            f"{args.stage2_arm}_label": b_lab,
            f"{args.stage2_arm}_reasoning": b_reason,
        }
        rows.append(row)
        if status == "UNCERTAIN":
            uncertain.append(row)

    fields = list(rows[0].keys()) if rows else ["paragraph"]
    out_path = data_dir / "classified_results.csv"
    with out_path.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    unc_path = data_dir / "uncertain_review.csv"
    with unc_path.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(uncertain)

    findings_path = write_findings(data_dir, args.name, rows, src_name)

    by_status = Counter(r["status"] for r in rows)
    by_verif = Counter(r["verification"] for r in rows)
    by_tier = Counter(r["tier"] for r in rows)
    positives = [r for r in rows if r["final_label"] not in ("NA", *BAD)]
    unnumbered = sum(1 for r in positives if not isinstance(r["para_num"], int))

    manifest = {
        "book": args.name,
        "mode": "two-stage",
        "stage1_arm": args.stage1_arm,
        "stage2_arm": args.stage2_arm,
        "rows": len(rows),
        "by_status": dict(by_status),
        "by_verification": dict(by_verif),
        "by_tier": dict(by_tier),
        "positives": len(positives),
        "positive_rate": round(len(positives) / max(len(rows), 1), 5),
        "strata": strata or None,
        "staleness_problems": problems or None,
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8")

    print("━" * 62)
    print(f"DATASET BUILT — {args.name}: {len(rows)} paragraphs")
    print(f"\n  by status:        {dict(by_status)}")
    print(f"  by verification:  {dict(by_verif)}")
    print(f"  by tier:          {dict(by_tier)}")
    n_single = by_verif.get("single_arm", 0)
    if n_single:
        print(f"\n  {n_single} rows ({n_single/len(rows):.1%}) carry a single unverified "
              f"Gemini label.\n  They are confident-NA paragraphs outside the control "
              f"sample; verification='single_arm'\n  marks them. Any count restricted to "
              f"verified labels must exclude them.")
    print(f"\n  positives: {len(positives)} ({len(positives)/max(len(rows),1):.2%})")
    for lbl, n in Counter(r["final_label"] for r in positives).most_common():
        print(f"    {lbl:<28} {n:>5}")
    if unnumbered:
        print(f"  !! {unnumbered} findings could not be matched to a row number in "
              f"{src_name}")
    print(f"\n  {out_path}\n  {unc_path}\n  {findings_path}  <- the non-NA paragraphs"
          f"\n  {data_dir/'dataset_manifest.json'}")
    return 0


def merge_census(data_dir: Path, args) -> int:
    arm_a, arm_b = args.arms
    path_a, path_b = data_dir / f"{arm_a}_results.csv", data_dir / f"{arm_b}_results.csv"
    for arm, path in ((arm_a, path_a), (arm_b, path_b)):
        if not path.exists():
            print(f"ERROR: {path} not found. Run classify_single.py --model {arm} first.")
            return 1
    res_a, res_b = load_results(path_a, arm_a), load_results(path_b, arm_b)

    problems = check_staleness(data_dir, arm_a, set(res_a))
    problems += check_staleness(data_dir, arm_b, set(res_b))
    if problems:
        print("STALENESS CHECK FAILED:")
        for p in problems:
            print(f"  ✗ {p}")
        if not args.allow_stale:
            print("\nRefusing to merge mismatched inputs. Pass --allow-stale to override.")
            return 2

    src_name, para_idx = source_index(data_dir, arm_a)
    rows, uncertain = [], []
    for para, a in res_a.items():
        b = res_b.get(para, {"label": "MISSING", "reasoning": ""})
        agree = (a["label"] in VALID_CATEGORIES and b["label"] in VALID_CATEGORIES
                 and a["label"] == b["label"])
        row = {"book": args.name,
               "para_num": para_idx.get(para, ""),
               "paragraph": para,
               "final_label": a["label"],
               "status": "CONFIRMED" if agree else "UNCERTAIN",
               "verification": "dual_arm",
               "tier": "census",
               f"{arm_a}_label": a["label"], f"{arm_a}_reasoning": a["reasoning"],
               f"{arm_b}_label": b["label"], f"{arm_b}_reasoning": b["reasoning"]}
        rows.append(row)
        if not agree:
            uncertain.append(row)

    fields = list(rows[0].keys()) if rows else ["paragraph"]
    for path, data in ((data_dir / "classified_results.csv", rows),
                       (data_dir / "uncertain_review.csv", uncertain)):
        with path.open("w", encoding="utf8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    findings_path = write_findings(data_dir, args.name, rows, src_name)
    conf = sum(1 for r in rows if r["status"] == "CONFIRMED")
    pos = sum(1 for r in rows if r["final_label"] not in ("NA", *BAD))
    print(f"MERGE COMPLETE — {conf}/{len(rows)} confirmed, {len(uncertain)} uncertain")
    print(f"  {pos} non-NA paragraphs -> {findings_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Build the final classified dataset")
    p.add_argument("--name", required=True, help="Textbook identifier")
    p.add_argument("--mode", choices=["two-stage", "census"], default="two-stage")
    p.add_argument("--stage1-arm", default="gemini")
    p.add_argument("--stage2-arm", default="qwen_local")
    p.add_argument("--stage2", default=None,
                   help="Stage-2 results CSV (default: <stage2-arm>_stage2.csv)")
    # The arms must be from different lineages — agreement between two models of
    # one family measures shared bias, not reliability.
    p.add_argument("--arms", nargs=2, default=["gemini", "gpt_oss"],
                   metavar=("ARM_A", "ARM_B"), help="census mode only")
    p.add_argument("--allow-stale", action="store_true",
                   help="Merge even if label files do not match the current extraction")
    args = p.parse_args()

    data_dir = Path("data") / args.name
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found.")
        return 1
    return (merge_two_stage if args.mode == "two-stage" else merge_census)(data_dir, args)


if __name__ == "__main__":
    raise SystemExit(main())
