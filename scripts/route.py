#!/usr/bin/env python3
"""Stage 1 -> Stage 2 routing for the two-stage classifier.

Stage 1 runs the Gemini arm over every paragraph. Stage 2 runs a local, free arm
over a *subset*, because a second census would cost ~80 hours of local inference.
Which subset is the whole methodological question, so the rule is explicit here
rather than left to default:

  TIER A  positive        stage-1 label != NA
                          -> verify every one. These are the findings.

  TIER B  boundary        stage-1 label == NA, but confidence is not "high", or a
                          non-NA category was recorded in `considered`.
                          -> the cases where the screener itself was unsure. Under
                             the old prompt these were silently indistinguishable
                             from confident NA, which is why the prompt now records
                             confidence and considered.

  TIER C  control         confident NA, no alternative considered.
                          -> a random sample, sized by --control-n. This stratum is
                             the only thing that can bound what the whole design
                             misses, because stage 2 never sees the rest of it.

Why the control sample must be large: error pedagogy runs at roughly 0.5-1.7% of
paragraphs. With 0 misses found in n samples, the 95% upper bound on the miss rate
is about 3/n, so a small control bounds recall uselessly loosely — 80 samples
against a 1,572-paragraph pool permits a recall as low as 17%. Stage 2 is free,
so the control should be spent generously; --control-n 2000 on a 25,000-paragraph
corpus buys a recall floor around 95% instead.

A prefilter was tried first and abandoned: see the changelog entry for 2026-09-24.

Usage:
    python scripts/route.py --name ck12_algebra1_hs --control-n 600
    python scripts/classify_single.py --name ck12_algebra1_hs --model qwen_local \
        --paragraphs stage2_worklist.csv --out qwen_local_stage2 --sleep 0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

HIGH = "high"
DEFAULT_PLAN = Path("data") / "control_plan.json"


def tier_rows(rows: list[dict], arm: str) -> dict[str, list[dict]]:
    """Assign every stage-1 row to tier A, B or the tier-C pool.

    Shared with plan_control.py so the planner counts pools exactly the way the
    router later samples them; two copies of this rule would drift.
    """
    lab, conf, cons = f"{arm}_label", f"{arm}_confidence", f"{arm}_considered"
    out: dict[str, list[dict]] = {"A_positive": [], "B_boundary": [],
                                  "C_control_pool": [], "errors": []}
    for r in rows:
        label = (r.get(lab) or "").strip()
        if label in ("ERROR", "PARSE_ERROR", ""):
            out["errors"].append(r)
        elif label != "NA":
            out["A_positive"].append(r)
        elif (r.get(conf) or "").strip().lower() != HIGH or (r.get(cons) or "").strip():
            out["B_boundary"].append(r)
        else:
            out["C_control_pool"].append(r)
    return out


def required_columns(rows: list[dict], arm: str) -> list[str]:
    need = [f"{arm}_label", f"{arm}_confidence", f"{arm}_considered"]
    return [c for c in need if rows and c not in rows[0]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the stage-2 worklist from stage-1 labels.")
    ap.add_argument("--name", required=True, help="Textbook identifier")
    ap.add_argument("--stage1", default="gemini_results.csv",
                    help="Stage-1 results CSV (default: gemini_results.csv)")
    ap.add_argument("--arm", default="gemini", help="Stage-1 arm name, for column prefixes")
    # No default: the control size comes from the corpus-wide proportional-allocation
    # plan, because a per-book default is either a trap (too small to bound anything)
    # or a census in disguise (2000 exceeds most books' entire confident-NA pool).
    ap.add_argument("--control-n", type=int, default=None,
                    help="Override this book's control size. Breaks proportional "
                         "allocation; normally let --plan decide.")
    ap.add_argument("--plan", default=str(DEFAULT_PLAN),
                    help=f"Proportional-allocation plan (default: {DEFAULT_PLAN})")
    ap.add_argument("--min-recall-floor", type=float, default=0.90,
                    help="Warn below this per-book projected recall floor (default: 0.90). "
                         "Advisory only: the binding bound is computed per series by "
                         "stage2_corpus_report.py, since no per-book recall claim is made.")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", default="stage2_worklist.csv")
    args = ap.parse_args()

    data_dir = Path("data") / args.name
    s1 = data_dir / args.stage1
    if not s1.exists():
        print(f"ERROR: {s1} not found. Run stage 1 first:\n"
              f"  python scripts/classify_single.py --name {args.name} --model {args.arm}",
              file=sys.stderr)
        return 1

    lab, conf = f"{args.arm}_label", f"{args.arm}_confidence"
    rows = list(csv.DictReader(s1.open(encoding="utf8")))
    if not rows:
        print(f"ERROR: {s1} is empty.", file=sys.stderr)
        return 1
    missing = required_columns(rows, args.arm)
    if missing:
        print(f"ERROR: {s1} lacks {missing}. It predates the uncertainty fields; "
              f"re-run stage 1 so boundary cases can be identified.", file=sys.stderr)
        return 1

    tiers = tier_rows(rows, args.arm)
    errors = tiers["errors"]
    pool = tiers["C_control_pool"]

    # Control size: a proportional-allocation plan takes precedence over --control-n,
    # because the corpus-level allocation is only proportional if every book honours
    # its share. An explicit --control-n on the command line overrides the plan.
    plan_path = Path(args.plan)
    allocation = None
    if args.control_n is None and plan_path.exists():
        plan = json.loads(plan_path.read_text())
        entry = plan.get("allocation", {}).get(args.name)
        if entry is None:
            print(f"ERROR: {plan_path} has no allocation for {args.name}. Re-run "
                  f"plan_control.py after stage 1 finished for this book.",
                  file=sys.stderr)
            return 1
        allocation = entry
        control_n = entry["control_n"]
        print(f"Using proportional allocation from {plan_path}: {control_n} "
              f"({entry['pool_share']:.2%} of the corpus confident-NA pool)")
    elif args.control_n is None:
        print(f"ERROR: no --control-n and no plan at {plan_path}.\n"
              f"  Run: python scripts/plan_control.py --target N\n"
              f"  (or pass --control-n to sample this book in isolation, which breaks "
              f"proportional allocation)", file=sys.stderr)
        return 1
    else:
        control_n = args.control_n

    random.seed(args.seed)
    control = random.sample(pool, min(control_n, len(pool)))

    # Advisory only. A per-book recall floor is not a guarantee this study uses:
    # every claim is corpus-level or cross-series, so the binding bound is computed
    # per series across books. Blocking here would refuse a book's correct
    # proportional share merely for being a small share.
    tp = len(tiers["A_positive"])
    if tp and control and len(control) < len(pool):
        floor = tp / (tp + len(pool) * 3 / len(control))
        if floor < args.min_recall_floor:
            print(f"  note: this book alone bounds recall at {floor:.1%} — expected for a "
                  f"proportional share.\n        The binding bound is per series; run "
                  f"stage2_corpus_report.py after stage 2.")

    worklist = ([(r, "A_positive") for r in tiers["A_positive"]]
                + [(r, "B_boundary") for r in tiers["B_boundary"]]
                + [(r, "C_control") for r in control])

    out_path = data_dir / args.out
    with out_path.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["paragraph", "tier", f"stage1_{args.arm}_label",
                                          f"stage1_{args.arm}_confidence"])
        w.writeheader()
        for r, tier in worklist:
            w.writerow({"paragraph": r["paragraph"], "tier": tier,
                        f"stage1_{args.arm}_label": r[lab],
                        f"stage1_{args.arm}_confidence": r[conf]})

    strata = {
        "book": args.name,
        "stage1_file": args.stage1,
        "stage1_arm": args.arm,
        "n_paragraphs": len(rows),
        "n_stage1_errors": len(errors),
        "tier_A_positive": len(tiers["A_positive"]),
        "tier_B_boundary": len(tiers["B_boundary"]),
        "tier_C_pool": len(pool),
        "tier_C_sampled": len(control),
        "seed": args.seed,
        "positive_rate": round(len(tiers["A_positive"]) / len(rows), 5),
        # Recorded so stage2_corpus_report.py can verify that what was sampled
        # matches the plan; a book routed with an explicit --control-n breaks the
        # proportional allocation and the report has to be able to say so.
        "allocation": allocation,
        "control_source": "plan" if allocation else "explicit --control-n",
    }
    (data_dir / "stage2_strata.json").write_text(json.dumps(strata, indent=2) + "\n",
                                                 encoding="utf8")

    print(f"{args.name}: {len(rows)} paragraphs")
    print(f"  A positive      {len(tiers['A_positive']):6}  ({len(tiers['A_positive'])/len(rows):.2%})")
    print(f"  B boundary      {len(tiers['B_boundary']):6}")
    print(f"  C control pool  {len(pool):6}  -> sampling {len(control)}")
    if errors:
        print(f"  !! {len(errors)} stage-1 errors excluded; re-run stage 1 to fill them")
    print(f"\n  stage-2 worklist: {len(worklist)} paragraphs -> {out_path}")
    if pool and control:
        # Rule of three: 0 misses in n gives a 95% upper bound of ~3/n.
        fn_hi = len(pool) * 3 / len(control)
        tp = len(tiers["A_positive"])
        if tp:
            print(f"  if stage 2 finds 0 misses in tier C, recall floor "
                  f"~{tp/(tp+fn_hi):.1%} (FN <= {fn_hi:.0f})")
            print(f"  confidence breakdown: "
                  f"{dict(Counter((r.get(conf) or '?') for r in rows))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
