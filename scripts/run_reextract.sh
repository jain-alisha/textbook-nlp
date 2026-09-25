#!/bin/bash
# Re-extract every book with an identified defect, under the current extractor.
#
#   CK-12 x3        — Era 2 runs that silently fell back to PyMuPDF after 504s
#   CPM Algebra 2   — from v2 (the 531pp file is an OCR'd partial volume)
#   CPM Geometry    — structural audit (2026-09-25) found 24.8% letter-spaced
#                     text and 11.6% inline margin-note titles fused into problem
#                     text, e.g. "Core Connections Geometry ETHODS AND MEANINGS
#                     MATH NOTES The Perimeter..." — a PyMuPDF reading-order
#                     artifact wearing Era 2's paragraph-count profile, not an
#                     actual Gemini extraction. It had been on the leave-alone
#                     list on the strength of stats alone (paragraph count,
#                     median length); this is why that isn't sufficient — see
#                     the Saxon Course 3 case below and the README changelog.
#   Saxon x3        — Era 1, unchunked gemini-2.0-flash, unreproducible.
#                     Course 3 in particular looked clean by paragraph-count and
#                     median-length stats alone (1.00/page, median 1384 chars)
#                     but is 31% letter-spaced raw text underneath.
#
# Deliberately NOT re-extracted: cpm_course1/2/3_ms, cpm_algebra1_hs. Checked with
# the same structural audit (letter-spacing, inline margin notes, ligatures) that
# caught the books above, not just era-dated — these come back clean: 1.6-5.6%
# letter-spacing (likely genuine spaced headings, not systemic) and 0% inline
# margin notes. They do carry a running-header prefix on ~48% of paragraphs
# ("Core Connections Course 2 1-51. ..."), which is a clean_paragraphs.py job,
# not a re-extraction.
#
# 25 pages/chunk: CK-12 yields ~7 paragraphs/page, so 50-page chunks ran ~250s
# and hit the old 300s deadline. At 25 pages a chunk completes in ~90s.
#
# Usage: caffeinate -i bash scripts/run_reextract.sh

set -u

NAMES=(ck12_algebra1_hs ck12_geometry_hs ck12_algebra2_hs cpm_algebra2_hs
       cpm_geometry_hs saxon_course1_ms saxon_course2_ms saxon_course3_ms)
PDFS=(pdfs/ck12_algebra1_hs.pdf pdfs/ck12_geometry_hs.pdf pdfs/ck12_algebra2_hs.pdf
      pdfs/cpm_algebra2_hs_v2.pdf pdfs/cpm_geometry_hs.pdf
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
