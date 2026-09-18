#!/bin/bash
# Run the full pipeline for all 3 Saxon books in succession.
# Each book: extract → stitch → classify (qwen + gpt-oss) → merge
#
# Usage: bash scripts/run_all_saxon.sh
# Leave running overnight — takes ~6-8 hours total.

set -e  # stop on any error

BOOKS=("saxon_course1_ms" "saxon_course2_ms" "saxon_course3_ms")
PDFS=("pdfs/saxon_course1_ms.pdf" "pdfs/saxon_course2_ms.pdf" "pdfs/saxon_course3_ms.pdf")

for i in "${!BOOKS[@]}"; do
    NAME="${BOOKS[$i]}"
    PDF="${PDFS[$i]}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STARTING: $NAME"
    echo "  $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Step 1: Extract
    echo ""
    echo "  [1/3] Extracting paragraphs from $PDF..."
    python scripts/extract.py "$PDF" --name "$NAME"

    # Step 2: Stitch
    echo ""
    echo "  [2/3] Stitching paragraphs..."
    python scripts/stitch.py --name "$NAME" --start-fresh

    # Step 3: Classify + merge
    echo ""
    echo "  [3/3] Classifying..."
    python scripts/run_pipeline.py \
        --name "$NAME" \
        --paragraphs paragraphs_stitched.csv \
        --quota-wait 0 \
        --start-fresh

    echo ""
    echo "  ✓ $NAME COMPLETE at $(date)"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ALL 3 SAXON BOOKS COMPLETE"
echo "  $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
