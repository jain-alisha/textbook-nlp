#!/bin/bash
# Extract paragraphs for the 3 new CK-12 PDFs.
# Usage: bash scripts/run_ck12_extract.sh

set -e

NAMES=("ck12_algebra1_hs" "ck12_algebra2_hs" "ck12_geometry_hs")
PDFS=("pdfs/ck12_algebra1_hs.pdf" "pdfs/ck12_algebra2_hs.pdf" "pdfs/ck12_geometry_hs.pdf")

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
echo "  ALL 3 CK-12 BOOKS EXTRACTED"
echo "  $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
