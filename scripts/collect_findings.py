#!/usr/bin/env python3
"""Concatenate every book's findings.csv into one corpus-wide table.

Each book's merge.py run writes data/<book>/findings.csv. This gathers them into
data/findings_all.csv so the corpus can be read in one place, and prints the
per-book and per-category breakdown that the research question asks for.

Rates are reported per 1,000 paragraphs as well as as percentages, because the
books differ in extraction granularity and a raw count favours whichever book was
split into more paragraphs.

Usage: python scripts/collect_findings.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

COLS = ["book", "para_num", "final_label", "status", "verification", "tier",
        "source_file", "paragraph"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/findings_all.csv")
    ap.add_argument("--verified-only", action="store_true",
                    help="Exclude rows whose label was never checked by a second arm")
    args = ap.parse_args()

    books = sorted(p.parent.name for p in Path("data").glob("*/findings.csv"))
    if not books:
        print("No findings.csv found. Run merge.py for at least one book first.")
        return 1

    rows, totals = [], {}
    for book in books:
        d = Path("data") / book
        found = list(csv.DictReader((d / "findings.csv").open(encoding="utf8")))
        if args.verified_only:
            found = [r for r in found if r.get("verification") == "dual_arm"]
        rows += found
        mf = d / "dataset_manifest.json"
        totals[book] = json.loads(mf.read_text()).get("rows") if mf.exists() else None

    out = Path(args.out)
    with out.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_book = defaultdict(Counter)
    for r in rows:
        by_book[r["book"]][r["final_label"]] += 1

    cats = sorted({r["final_label"] for r in rows})
    print(f"{'book':24} {'paras':>7} {'found':>6} {'per 1k':>7}  " +
          " ".join(f"{c[:11]:>11}" for c in cats))
    print("-" * (48 + 12 * len(cats)))
    for book in books:
        n = sum(by_book[book].values())
        tot = totals.get(book)
        per1k = f"{n/tot*1000:7.1f}" if tot else "      ?"
        print(f"{book:24} {tot if tot else '?':>7} {n:6} {per1k}  " +
              " ".join(f"{by_book[book][c]:>11}" for c in cats))

    print("-" * (48 + 12 * len(cats)))
    grand = Counter(r["final_label"] for r in rows)
    known = [t for t in totals.values() if t]
    print(f"{'TOTAL':24} {sum(known) if known else '?':>7} {len(rows):6} "
          f"{'':>7}  " + " ".join(f"{grand[c]:>11}" for c in cats))
    if args.verified_only:
        print("\n  --verified-only: single_arm rows excluded.")
    else:
        n_single = sum(1 for r in rows if r.get("verification") == "single_arm")
        if n_single:
            print(f"\n  {n_single} of these findings carry an unverified single-arm label; "
                  f"re-run with --verified-only to exclude them.")
    print(f"\n  {out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
