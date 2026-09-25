#!/usr/bin/env python3
"""Per-stratum agreement and the recall bound for a two-stage run.

Reports, per tier:
  A_positive  agreement between the arms on stage-1's findings. Disagreement here
              is the interesting case: the screener claimed error pedagogy and the
              independent arm did not.
  B_boundary  what the screener's own hesitation was worth.
  C_control   misses. Every non-NA label here is a paragraph stage 1 called a
              confident NA and the second arm did not — i.e. a false negative.

Kappa is reported on A+B+C combined and NOT corpus-wide, deliberately. Over a
corpus that is ~99% NA, agreement is dominated by both arms trivially agreeing on
obvious non-cases, which inflates kappa without evidencing reliability on the
judgement that matters. Restricting it to the routed strata is the more
informative statistic, and is a design choice, not a limitation.

Usage: python scripts/stage2_report.py --name ck12_algebra1_hs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa. Returns nan when undefined (single category, or n == 0)."""
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--stage1-arm", default="gemini")
    ap.add_argument("--stage2-arm", default="qwen_local")
    ap.add_argument("--stage2", default=None,
                    help="Stage-2 results CSV (default: <stage2-arm>_stage2.csv)")
    args = ap.parse_args()

    d = Path("data") / args.name
    s2_path = d / (args.stage2 or f"{args.stage2_arm}_stage2.csv")
    strata_path = d / "stage2_strata.json"
    for p in (s2_path, strata_path):
        if not p.exists():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            return 1
    strata = json.loads(strata_path.read_text())

    s1_lab = f"{args.stage1_arm}_label"
    s1 = {}
    s1_path = d / strata["stage1_file"]
    for r in csv.DictReader(s1_path.open(encoding="utf8")):
        s1[(r["paragraph"] or "").strip()] = (r.get(s1_lab) or "").strip()

    s2_lab = f"{args.stage2_arm}_label"
    by_tier: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    unmatched = 0
    for r in csv.DictReader(s2_path.open(encoding="utf8")):
        para = (r.get("paragraph") or "").strip()
        tier = (r.get("tier") or "?").strip()
        l2 = (r.get(s2_lab) or "").strip()
        l1 = s1.get(para)
        if l1 is None:
            unmatched += 1
            continue
        by_tier[tier].append((para, l1, l2))

    print(f"{args.name}: stage1={args.stage1_arm} stage2={args.stage2_arm}")
    print(f"  {strata['n_paragraphs']} paragraphs, "
          f"stage-1 positive rate {strata['positive_rate']:.2%}")
    if unmatched:
        print(f"  !! {unmatched} stage-2 rows had no stage-1 match "
              f"(stale worklist? re-run route.py)")
    print()
    print(f"{'tier':12} {'n':>6} {'agree':>7} {'s2 non-NA':>10}")
    print("-" * 40)
    for tier in ("A_positive", "B_boundary", "C_control"):
        rows = by_tier.get(tier, [])
        if not rows:
            print(f"{tier:12} {0:6}")
            continue
        agree = sum(1 for _, a, b in rows if a == b)
        nonna = sum(1 for _, _, b in rows if b not in ("NA", "ERROR", "PARSE_ERROR", ""))
        print(f"{tier:12} {len(rows):6} {agree/len(rows):7.1%} {nonna:10}")

    allrows = [r for t in ("A_positive", "B_boundary", "C_control") for r in by_tier.get(t, [])]
    if allrows:
        k = kappa([a for _, a, _ in allrows], [b for _, _, b in allrows])
        print(f"\nkappa over routed strata (n={len(allrows)}): {k:.3f}")
        print("  (deliberately not corpus-wide; see module docstring)")

    # The bound the control stratum exists to produce.
    ctrl = by_tier.get("C_control", [])
    tp = strata["tier_A_positive"]
    if ctrl and tp:
        misses = [(p, b) for p, a, b in ctrl if b not in ("NA", "ERROR", "PARSE_ERROR", "")]
        pool, n = strata["tier_C_pool"], len(ctrl)
        if misses:
            rate = len(misses) / n
            fn = pool * rate
            print(f"\nRECALL ESTIMATE: {len(misses)}/{n} control paragraphs were non-NA "
                  f"for stage 2\n  -> est. {fn:.0f} missed positives in the "
                  f"{pool}-paragraph confident-NA pool\n  -> recall ~{tp/(tp+fn):.1%}")
        else:
            fn_hi = pool * 3 / n   # rule of three, 95%
            print(f"\nRECALL BOUND: 0/{n} control paragraphs were non-NA for stage 2.")
            print(f"  95% upper bound on the miss rate is 3/{n} = {3/n:.3%}")
            print(f"  -> at most ~{fn_hi:.0f} missed positives in the {pool}-paragraph pool")
            print(f"  -> recall >= {tp/(tp+fn_hi):.1%}")
            if tp / (tp + fn_hi) < 0.9:
                print(f"  !! Loose. Raise --control-n and re-run stage 2; the arm is free.")
        print("\n  Note: stage 2 is a second model, not ground truth. This bounds "
              "agreement-\n  based recall only; a human-labelled probe set remains "
              "the validity check.")

    dis = [(p, a, b) for p, a, b in by_tier.get("A_positive", []) if a != b]
    if dis:
        print(f"\nDISAGREEMENTS ON STAGE-1 POSITIVES ({len(dis)}) — review these:")
        for p, a, b in dis[:15]:
            print(f"  s1={a:24} s2={b:24} {p[:70]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
