#!/bin/bash
# Stitch paragraphs for all 8 newly-extracted books (5 CPM + 3 CK-12).
# Usage: bash scripts/run_stitch_new.sh

set -e

NAMES=("cpm_course1_ms" "cpm_course2_ms" "cpm_course3_ms" "cpm_algebra1_hs" "cpm_geometry_hs" "ck12_algebra1_hs" "ck12_algebra2_hs" "ck12_geometry_hs")

for NAME in "${NAMES[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STITCHING: $NAME"
    echo "  $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python -u scripts/stitch.py --name "$NAME" --start-fresh

    echo "  ✓ $NAME done at $(date)"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ALL 8 BOOKS STITCHED"
echo "  $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
