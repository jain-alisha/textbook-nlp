# Textbook Error Analysis Toolkit

Gemini-powered pipeline for extracting and classifying textbook paragraphs into pedagogical error categories.

## Project Structure

```
textbook_errors/
├── gemini_reader.py          # Extract paragraphs from PDF (Gemini or PyMuPDF)
├── gemini_analysis.py         # Classify paragraphs using Gemini
├── seed_phrases.csv           # Category definitions and examples
├── paragraph_clusters.csv     # Extracted paragraphs (generated)
├── analysis_results.csv       # Classification output (generated)
├── requirements.txt           # Python dependencies
├── .env                       # Local configuration (API keys, paths)
└── README.md                  # This file
```

## Setup

1. **Create virtual environment and install dependencies:**

```bash
cd /Users/alishajain/Downloads/textbook_errors
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Configure environment:**

Edit `.env` to set your Gemini API key and textbook path:

```bash
GEMINI_API_KEY=your_key_here
GEMINI_ENABLED=1
TEXTBOOK_PDF_PATH="/Users/alishajain/Downloads/textbook_errors/ccalg2vol2.pdf"
TEXTBOOK_SAMPLE_FRAC=0.4
```

> **Important:** Never commit `.env` to version control—it contains secrets.

## Usage

### Step 1: Extract Paragraphs from PDF

Extract paragraph chunks from the textbook PDF and save to CSV:

```bash
python gemini_reader.py ccalg2vol2.pdf --output paragraph_clusters.csv
```

This runs **once** to create `paragraph_clusters.csv` which can be reused for multiple classification runs.

**Options:**
- `--use-gemini`: Force Gemini extraction (otherwise controlled by `GEMINI_ENABLED` env var)
- `--page-fraction 0.5`: Process only 50% of pages (default: 1.0 = all pages)
- `--start-page 0`: Starting page (0-indexed)
- `--end-page 100`: Ending page (exclusive)
- `--append`: Append to existing CSV instead of overwriting

**Chunked Extraction (for large PDFs):**

For large textbooks, extract in chunks to avoid timeouts:

```bash
# Extract first 100 pages
python gemini_reader.py ccalg2vol2.pdf --output paragraph_clusters.csv --start-page 0 --end-page 100

# Extract next 100 pages and append
python gemini_reader.py ccalg2vol2.pdf --output paragraph_clusters.csv --start-page 100 --end-page 200 --append

# Or use the convenience script:
./extract_chunked.sh ccalg2vol2.pdf 100 paragraph_clusters.csv
```

The `extract_chunked.sh` script automatically divides the PDF into chunks and combines results.

### Step 2: Classify Paragraphs

Classify extracted paragraphs using Gemini and seed phrase categories:

```bash
python gemini_analysis.py paragraph_clusters.csv --output analysis_results.csv --sample-fraction 0.4
```

**Output format:** CSV with two columns: `[paragraph, category]`

**Categories:**
- `Incorrect-to-correct revision tasks`: Prompts asking students to find and fix errors
- `Compare-and-contrast examples`: Comparing multiple solution methods
- `Explicit error detection`: Identifying specific mistakes
- `Common error alerts`: Warnings about typical student mistakes
- `N/A`: Neutral content (standard problems, definitions, instructions)

**Options:**
- `--sample-fraction 0.4`: Classify 40% of paragraphs (reduce to avoid quota limits)
- `--seed-phrases seed_phrases.csv`: Category definitions (default: seed_phrases.csv)
- `--random-seed 42`: Reproducible sampling

**Direct PDF analysis (combines both steps):**

```bash
python gemini_analysis.py ccalg2vol2.pdf --output analysis_results.csv --sample-fraction 0.4
```

### Multi-model sampling harness

Use `multi_model_sampler.py` to pull 20 paragraphs from the middle of `paragraph_clusters.csv` and send them to multiple inference providers (Groq, Cloudflare Workers AI, Hugging Face Serverless, OpenRouter, and a local Ollama instance).

```bash
python multi_model_sampler.py \
	--input paragraph_clusters.csv \
	--output multi_model_results.jsonl
```

Flags:
- `--sample-size`: Number of contiguous paragraphs to probe (default: 20).
- `--dry-run`: Skip API calls to verify sampling logic.
- `--sleep`: Seconds to pause between provider calls (default: 1).

Configure credentials in `.env`:

```
GROQ_API_KEY=...
GROQ_MODEL=mixtral-8x7b-32768
CLOUDFLARE_AI_API_KEY=...
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_MODEL=@cf/meta/llama-2-7b-chat-int8
HUGGINGFACE_API_KEY=...
HUGGINGFACE_MODEL=facebook/bart-large-mnli
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openrouter/auto
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
```

Results are written as JSONL, one record per paragraph with provider-specific responses.

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Gemini API key (required)
- `GEMINI_ENABLED`: Set to `1` to enable Gemini extraction (default: fallback to PyMuPDF)
- `TEXTBOOK_PDF_PATH`: Default PDF path
- `TEXTBOOK_SAMPLE_FRAC`: Default sampling fraction (0-1)
- `GEMINI_CLASSIFIER_MODEL`: Model for classification (default: `models/gemini-flash-latest`)
- `GEMINI_READER_MODEL`: Model for PDF extraction (default: `models/gemini-flash-latest`)

### Seed Phrases

`seed_phrases.csv` defines categories with example phrases. Format:

```csv
phrase,category
"Review the steps in Priya's solution above...",Incorrect-to-correct revision tasks
"Compare Method A and Method B...",Comparison of multiple methods
```

## Workflow

1. **Extract once:** `python gemini_reader.py ccalg2vol2.pdf` → saves `paragraph_clusters.csv`
2. **Classify multiple times:** Reuse `paragraph_clusters.csv` with different sampling fractions or category definitions without re-extracting from PDF
3. **Review results:** Open `analysis_results.csv` to see classified paragraphs

## API Quota Management

Gemini free tier has rate limits (e.g., 2 requests/minute). To manage:

- Start with `--sample-fraction 0.01` to test (classifies ~1-2 paragraphs)
- Gradually increase: `0.05`, `0.1`, `0.4` as quota allows
- If quota exhausted, script stops gracefully with message
- Wait or upgrade API plan for larger batch classification

## Troubleshooting

**"Gemini extraction failed"**: Large PDFs may timeout or hit size/token limits with Gemini API. The script automatically falls back to PyMuPDF extraction. For very large textbooks (500+ pages), consider:
- Using chunked extraction: `./extract_chunked.sh ccalg2vol2.pdf 100`
- Disabling Gemini: Set `GEMINI_ENABLED=0` in `.env` to use PyMuPDF directly
- Using `--start-page` and `--end-page` flags for manual chunking

**"ResourceExhausted" error**: Gemini API quota exceeded. Reduce `--sample-fraction` or wait for quota reset.

**No paragraphs extracted**: Check PDF path in `.env` and ensure file exists at specified location.

**"Other" category appearing**: The classification now uses `N/A` for content that doesn't fit any error category. "Other" was removed to avoid redundancy.

## Notes

- **PyMuPDF for large PDFs**: Gemini PDF extraction works best for smaller documents (<100 pages). For large textbooks, use PyMuPDF extraction (set `GEMINI_ENABLED=0`).
- **Two-column detection**: Script automatically handles multi-column textbook layouts with k-means clustering.
- **Incremental extraction**: Use `--append` flag to build up extraction results across multiple runs without re-processing earlier pages.
# textbook-nlp
