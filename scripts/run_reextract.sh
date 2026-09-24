#!/bin/bash
# Re-extract every book with an identified defect, under the current extractor.
#
#   CK-12 x3        — Era 2 runs that silently fell back to PyMuPDF after 504s
#   CPM Algebra 2   — from v2 (the 531pp file is an OCR'd partial volume)
#   Saxon x3        — Era 1, unchunked gemini-2.0-flash, unreproducible
#
# Deliberately NOT re-extracted: cpm_course1/2/3_ms, cpm_algebra1_hs,
# cpm_geometry_hs. Those are clean Era 2 runs; re-running them would spend money
# to replace working output and risk introducing a new inconsistency.
#
# 25 pages/chunk: CK-12 yields ~7 paragraphs/page, so 50-page chunks ran ~250s
# and hit the old 300s deadline. At 25 pages a chunk completes in ~90s.
#
# Usage: caffeinate -i bash scripts/run_reextract.sh

set -u

NAMES=(ck12_algebra1_hs ck12_geometry_hs ck12_algebra2_hs cpm_algebra2_hs
       saxon_course1_ms saxon_course2_ms saxon_course3_ms)
PDFS=(pdfs/ck12_algebra1_hs.pdf pdfs/ck12_geometry_hs.pdf pdfs/ck12_algebra2_hs.pdf
      pdfs/cpm_algebra2_hs_v2.pdf
      pdfs/saxon_course1_ms.pdf pdfs/saxon_course2_ms.pdf pdfs/saxon_course3_ms.pdf)

failed=()
for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"; PDF="${PDFS[$i]}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $NAME  <-  $PDF"
    echo "  $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Keep going on failure: one book aborting (e.g. the >20% fallback guard)
    # must not strand the rest of the queue.
    if python3 -u scripts/extract.py "$PDF" --name "$NAME" --chunk-size 25; then
        echo "  ✓ $NAME done at $(date)"
    else
        echo "  ✗ $NAME FAILED (exit $?) at $(date)"
        failed+=("$NAME")
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ${#failed[@]} -eq 0 ]; then
    echo "  ALL ${#NAMES[@]} BOOKS RE-EXTRACTED"
else
    echo "  ${#failed[@]} FAILED: ${failed[*]}"
fi
echo "  $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
