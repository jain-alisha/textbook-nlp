#!/bin/bash
# Extract paragraphs for the remaining CPM Core Connections PDFs
# (cpm_algebra2_hs already has a full processed run from the old PDF, so it's excluded here).
# Usage: bash scripts/run_cpm_extract.sh

set -e

NAMES=("cpm_course1_ms" "cpm_course2_ms" "cpm_course3_ms" "cpm_algebra1_hs" "cpm_geometry_hs")
PDFS=("pdfs/cpm_course1_ms.pdf" "pdfs/cpm_course2_ms.pdf" "pdfs/cpm_course3_ms.pdf" "pdfs/cpm_algebra1_hs.pdf" "pdfs/cpm_geometry_hs.pdf")

for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"
    PDF="${PDFS[$i]}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  EXTRACTING: $NAME  <-  $PDF"
    echo "  $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python -u scripts/extract.py "$PDF" --name "$NAME"

    echo "  ✓ $NAME done at $(date)"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ALL 5 REMAINING CPM BOOKS EXTRACTED"
echo "  $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
