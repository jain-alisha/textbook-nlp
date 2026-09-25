#!/usr/bin/env python3
"""Per-series and corpus recall bounds from the stratified tier-C control.

Aggregates every book's stage-2 control results. Because the control is allocated
proportionally to each book's confident-NA pool, the pooled miss rate is an unbiased
estimate of the corpus miss rate and each series' control share equals its pool
share — so series-level bounds need no reweighting.

Bounds are reported both ways on purpose:

  absolute   "at most N missed paragraphs in a pool of P"
  relative   "recall >= R%"

The relative form is unstable where positives are rare: with ~36 positives in a
series, a bound of 19 missed paragraphs reads as "recall >= 65%" purely because the
denominator is small. The absolute count is the more honest statement, and the
series comparison is what the study's claims rest on.

Usage: python scripts/stage2_corpus_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_control import series_of          # noqa: E402
from stage2_report import wilson            # noqa: E402

BAD = {"", "NA", "ERROR", "PARSE_ERROR", "MISSING"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-arm", default="gemini")
    ap.add_argument("--stage2-arm", default="qwen_local")
    ap.add_argument("--stage2", default=None)
    args = ap.parse_args()

    s2_name = args.stage2 or f"{args.stage2_arm}_stage2.csv"
    s2_lab = f"{args.stage2_arm}_label"

    books = sorted(p.parent.name for p in Path("data").glob("*/stage2_strata.json"))
    if not books:
        print("No stage2_strata.json found. Run route.py first.", file=sys.stderr)
        return 1

    rows = []
    explicit = []
    for book in books:
        d = Path("data") / book
        strata = json.loads((d / "stage2_strata.json").read_text())
        if strata.get("control_source") != "plan":
            explicit.append(book)
        s2p = d / s2_name
        if not s2p.exists():
            print(f"  ✗ {book}: {s2_name} missing — stage 2 not run")
            continue
        ctrl = [r for r in csv.DictReader(s2p.open(encoding="utf8"))
                if (r.get("tier") or "") == "C_control"]
        misses = [r for r in ctrl if (r.get(s2_lab) or "").strip() not in BAD]
        rows.append({
            "book": book, "series": series_of(book),
            "pool": strata["tier_C_pool"], "control": len(ctrl),
            "misses": len(misses), "tp": strata["tier_A_positive"],
            "miss_rows": misses,
        })

    if not rows:
        print("No stage-2 control results found.", file=sys.stderr)
        return 1
    if explicit:
        print(f"WARNING: {len(explicit)} book(s) were routed with an explicit "
              f"--control-n rather than the plan:\n  {', '.join(explicit)}")
        print("  Allocation is no longer strictly proportional; pooled estimates are "
              "biased toward\n  over-sampled books. Re-run plan_control.py and route.py "
              "to restore it.\n")

    def block(label: str, subset: list[dict]) -> None:
        pool = sum(r["pool"] for r in subset)
        ctrl = sum(r["control"] for r in subset)
        miss = sum(r["misses"] for r in subset)
        tp = sum(r["tp"] for r in subset)
        if not ctrl:
            print(f"{label:10} no control sampled")
            return
        lo, hi = wilson(miss, ctrl)
        fn_hi = pool * hi
        rec = tp / (tp + fn_hi) if tp else float("nan")
        share = ctrl / pool if pool else 0
        print(f"{label:10} {pool:7} {ctrl:7} {share:7.1%} {miss:6} "
              f"{hi:8.3%} {fn_hi:9.0f} {tp:6} {rec:8.1%}")

    print(f"{'':10} {'pool':>7} {'control':>7} {'sampled':>7} {'misses':>6} "
          f"{'rate hi':>8} {'FN <=':>9} {'TP':>6} {'recall>=':>8}")
    print("-" * 76)
    by_series = defaultdict(list)
    for r in rows:
        by_series[r["series"]].append(r)
    for s in sorted(by_series):
        block(s, by_series[s])
    print("-" * 76)
    block("CORPUS", rows)

    print("\nper-book detail:")
    for r in sorted(rows, key=lambda r: (r["series"], r["book"])):
        print(f"  {r['book']:24} pool {r['pool']:6} control {r['control']:5} "
              f"({r['control']/max(r['pool'],1):5.1%})  misses {r['misses']:3}")

    # Comparability across series is the thing the study's claims need.
    rates = {s: sum(r["misses"] for r in v) / max(sum(r["control"] for r in v), 1)
             for s, v in by_series.items()}
    if len(rates) > 1:
        print("\nCOMPARABILITY ACROSS SERIES (what cross-series claims rest on):")
        for s in sorted(rates):
            ctrl = sum(r["control"] for r in by_series[s])
            lo, hi = wilson(sum(r["misses"] for r in by_series[s]), ctrl)
            print(f"  {s:8} miss rate {rates[s]:7.3%}  95% CI [{lo:.3%}, {hi:.3%}]  n={ctrl}")
        los = {s: wilson(sum(r['misses'] for r in by_series[s]),
                         sum(r['control'] for r in by_series[s]))[0] for s in rates}
        his = {s: wilson(sum(r['misses'] for r in by_series[s]),
                         sum(r['control'] for r in by_series[s]))[1] for s in rates}
        overlap = all(not (los[a] > his[b] or los[b] > his[a])
                      for a in rates for b in rates)
        if overlap:
            print("  -> all intervals overlap: no evidence of differential recall "
                  "across series.")
        else:
            print("  -> intervals DO NOT all overlap: recall may differ by series, which "
                  "would\n     confound cross-series comparisons. Investigate before "
                  "reporting rates.")

    misses = [(r["book"], m) for r in rows for m in r["miss_rows"]]
    if misses:
        print(f"\nMISSED POSITIVES FOUND IN CONTROL ({len(misses)}) — stage 1 called these "
              f"confident NA:")
        for book, m in misses[:20]:
            print(f"  {book:22} {m.get(s2_lab,''):24} {(m.get('paragraph') or '')[:60]!r}")
    print("\nNote: stage 2 is a second model, not ground truth. These bounds are "
          "agreement-based;\na human-labelled stratified probe remains the validity check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
