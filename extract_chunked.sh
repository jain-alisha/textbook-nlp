#!/bin/bash
# Extract textbook paragraphs in chunks to avoid Gemini timeouts
# Usage: ./extract_chunked.sh <pdf_path> <chunk_size> [output_csv]
#
# Example: ./extract_chunked.sh ccalg2vol2.pdf 100 paragraph_clusters.csv
# This will extract 100 pages at a time and append to paragraph_clusters.csv

set -e

PDF_PATH="${1}"
CHUNK_SIZE="${2:-100}"  # Default: 100 pages per chunk
OUTPUT_CSV="${3:-paragraph_clusters.csv}"

if [ -z "$PDF_PATH" ]; then
    echo "Usage: $0 <pdf_path> [chunk_size] [output_csv]"
    exit 1
fi

if [ ! -f "$PDF_PATH" ]; then
    echo "Error: PDF file not found: $PDF_PATH"
    exit 1
fi

# Get total page count
TOTAL_PAGES=$(python -c "import fitz; doc = fitz.open('$PDF_PATH'); print(len(doc))")
echo "Total pages: $TOTAL_PAGES"
echo "Chunk size: $CHUNK_SIZE pages"
echo "Output: $OUTPUT_CSV"
echo ""

# Remove existing output file if present
if [ -f "$OUTPUT_CSV" ]; then
    echo "Removing existing $OUTPUT_CSV..."
    rm -f "$OUTPUT_CSV"
fi

# Extract in chunks
START_PAGE=0
while [ $START_PAGE -lt $TOTAL_PAGES ]; do
    END_PAGE=$((START_PAGE + CHUNK_SIZE))
    if [ $END_PAGE -gt $TOTAL_PAGES ]; then
        END_PAGE=$TOTAL_PAGES
    fi
    
    echo "Extracting pages $START_PAGE-$END_PAGE..."
    
    if [ $START_PAGE -eq 0 ]; then
        # First chunk: create new file
        .venv/bin/python gemini_reader.py "$PDF_PATH" \
            --output "$OUTPUT_CSV" \
            --start-page $START_PAGE \
            --end-page $END_PAGE
    else
        # Subsequent chunks: append
        .venv/bin/python gemini_reader.py "$PDF_PATH" \
            --output "$OUTPUT_CSV" \
            --start-page $START_PAGE \
            --end-page $END_PAGE \
            --append
    fi
    
    START_PAGE=$END_PAGE
    echo ""
done

# Count final paragraphs
TOTAL_PARAS=$(tail -n +2 "$OUTPUT_CSV" | wc -l | tr -d ' ')
echo "Extraction complete! Total paragraphs: $TOTAL_PARAS"
