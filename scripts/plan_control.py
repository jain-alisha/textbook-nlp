#!/usr/bin/env python3
"""Allocate the tier-C control sample across books by proportional allocation.

Stratified sampling, strata = books, allocation strictly proportional to each
book's confident-NA pool:

    n_b = N * P_b / sum(P)

Proportional allocation is what makes the aggregate estimates unbiased without
reweighting: every confident-NA paragraph in the corpus has the same probability
N/sum(P) of entering the control, whichever book it is in. It also makes each
*series* share of the control equal its share of the pool automatically, which is
what the per-series recall bounds need — the study's claims are cross-series, so
that is the level the bound has to hold at.

Integer shares are assigned by the largest-remainder method so they sum to exactly
N rather than drifting by a paragraph per book. A book whose proportional share
exceeds its pool is capped at its pool, and the freed quota is redistributed over
the remaining books, so capping never silently shrinks the total.

Run this after stage 1 has finished for every book, and before route.py.

Usage:
    python scripts/plan_control.py --target 3000
    python scripts/plan_control.py --target 3000 --books ck12_algebra1_hs cpm_course2_ms
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route import required_columns, tier_rows   # noqa: E402  (shared tier rule)


def series_of(book: str) -> str:
    for s in ("ck12", "cpm", "saxon", "bigideas"):
        if book.startswith(s):
            return s
    return "other"


def allocate(pools: dict[str, int], target: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, capped at each pool."""
    alloc = {b: 0 for b in pools}
    remaining = {b: n for b, n in pools.items() if n > 0}
    quota = target
    while quota > 0 and remaining:
        total = sum(remaining.values())
        exact = {b: quota * n / total for b, n in remaining.items()}
        base = {b: min(int(v), remaining[b]) for b, v in exact.items()}
        assigned = sum(base.values())
        # Hand out the leftover one at a time, largest fractional part first.
        leftover = quota - assigned
        order = sorted(remaining, key=lambda b: exact[b] - int(exact[b]), reverse=True)
        for b in order:
            if leftover <= 0:
                break
            if base[b] < remaining[b]:
                base[b] += 1
                leftover -= 1
        for b, n in base.items():
            alloc[b] += n
            remaining[b] -= n
        quota = leftover
        remaining = {b: n for b, n in remaining.items() if n > 0}
        if not any(base.values()):
            break     # nothing more can be placed; pools are exhausted
    return alloc


def main() -> int:
    ap = argparse.ArgumentParser(description="Proportional control allocation across books")
    ap.add_argument("--target", type=int, required=True,
                    help="Total tier-C control paragraphs across the whole corpus")
    ap.add_argument("--arm", default="gemini", help="Stage-1 arm name")
    ap.add_argument("--stage1", default="gemini_results.csv")
    ap.add_argument("--books", nargs="*", default=None,
                    help="Restrict to these books (default: every book with stage-1 results)")
    ap.add_argument("--out", default="data/control_plan.json")
    args = ap.parse_args()

    candidates = args.books or sorted(
        p.parent.name for p in Path("data").glob(f"*/{args.stage1}"))
    if not candidates:
        print(f"No */{args.stage1} found. Run stage 1 first.", file=sys.stderr)
        return 1

    pools, tiers_by_book, skipped = {}, {}, []
    for book in candidates:
        path = Path("data") / book / args.stage1
        if not path.exists():
            skipped.append((book, "no stage-1 results"))
            continue
        rows = list(csv.DictReader(path.open(encoding="utf8")))
        if not rows:
            skipped.append((book, "empty"))
            continue
        missing = required_columns(rows, args.arm)
        if missing:
            skipped.append((book, f"missing {missing[0]} — stage 1 predates uncertainty fields"))
            continue
        t = tier_rows(rows, args.arm)
        tiers_by_book[book] = {k: len(v) for k, v in t.items()} | {"n_rows": len(rows)}
        pools[book] = len(t["C_control_pool"])

    if skipped:
        print("Skipped:")
        for b, why in skipped:
            print(f"  ✗ {b}: {why}")
        print()
    if not pools:
        print("No usable books.", file=sys.stderr)
        return 1

    total_pool = sum(pools.values())
    alloc = allocate(pools, min(args.target, total_pool))

    plan = {
        "target": args.target,
        "allocated": sum(alloc.values()),
        "total_confident_na_pool": total_pool,
        "method": "stratified, strata=books, proportional allocation, largest remainder",
        "arm": args.arm,
        "allocation": {
            b: {"control_n": alloc[b],
                "pool": pools[b],
                "pool_share": round(pools[b] / total_pool, 6),
                "sampled_share_of_pool": round(alloc[b] / pools[b], 6) if pools[b] else 0.0,
                "series": series_of(b),
                "tiers": tiers_by_book[b]}
            for b in sorted(pools)
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")

    print(f"{'book':24} {'series':7} {'pool':>7} {'share':>7} {'control':>8} {'of pool':>8}")
    print("-" * 66)
    for b in sorted(pools):
        a = plan["allocation"][b]
        print(f"{b:24} {a['series']:7} {a['pool']:7} {a['pool_share']:6.2%} "
              f"{a['control_n']:8} {a['sampled_share_of_pool']:7.1%}")
    print("-" * 66)
    print(f"{'TOTAL':24} {'':7} {total_pool:7} {1.0:6.2%} {sum(alloc.values()):8} "
          f"{sum(alloc.values())/total_pool:7.1%}")

    by_series: dict[str, list[int]] = {}
    for b in sorted(pools):
        a = plan["allocation"][b]
        s = by_series.setdefault(a["series"], [0, 0])
        s[0] += a["pool"]; s[1] += a["control_n"]
    print("\nper-series (the level the recall bound is reported at):")
    for s, (p, c) in sorted(by_series.items()):
        print(f"  {s:8} pool {p:6}  control {c:5}  "
              f"pool share {p/total_pool:5.1%} vs control share {c/max(sum(alloc.values()),1):5.1%}")
    est_hours = sum(alloc.values()) * 11.7 / 3600
    print(f"\n  {out}")
    print(f"  control alone is ~{est_hours:.1f} h of local stage-2 inference")
    print(f"  next: route.py per book (it reads this plan), then stage 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
