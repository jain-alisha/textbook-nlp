# Textbook Error Pedagogy Classifier

A pipeline to extract paragraphs from math textbook PDFs and classify them for error-based pedagogy using a 3-model Groq ensemble.

Each textbook gets its own isolated folder under `data/` so nothing is mixed across books.

---

## Project structure

```
textbook_errors/
  pdfs/                          # drop PDFs here before running extract.py
  data/
    <textbook_name>/
      paragraphs.csv             # output of extract.py
      classified_results.csv     # output of classify.py (confirmed + majority votes)
      uncertain_review.csv       # output of classify.py (all-disagree rows)
      progress.json              # checkpoint — classify.py resumes from here
  scripts/
    extract.py                   # PDF -> paragraphs.csv
    classify.py                  # paragraphs.csv -> classified_results.csv
    analyze.py                   # (stub) summary statistics
  .env                           # API keys — never commit this
  requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Add your Groq API key to `.env`:

```
GROQ_API_KEY=your_key_here
```

---

## Usage

### 1. Extract paragraphs from a PDF

```bash
python scripts/extract.py pdfs/cpm_algebra2.pdf --name cpm_algebra2
```

Output: `data/cpm_algebra2/paragraphs.csv`

### 2. Classify paragraphs

```bash
python scripts/classify.py --name cpm_algebra2
```

Output:
- `data/cpm_algebra2/classified_results.csv` — confirmed (2/2) votes
- `data/cpm_algebra2/uncertain_review.csv` — rows where models disagreed or errored
- `data/cpm_algebra2/progress.json` — checkpoint; re-running skips already-classified paragraphs

### 3. Analyze results (stub)

```bash
python scripts/analyze.py --name cpm_algebra2
```

---

## Classification categories

| Category | Description |
|---|---|
| `INCORRECT_TO_CORRECT` | A named student's wrong work is shown; reader finds/fixes the error |
| `COMPARE_AND_CONTRAST` | Two named students disagree; reader determines who is correct |
| `EXPLICIT_ERROR_DETECTION` | An error is shown; reader must identify or locate it |
| `COMMON_ERROR_ALERT` | Text warns about a mistake students frequently make |
| `NA` | Standard content: problem sets, definitions, examples, instructions |

Voting: 2/2 agreement = **CONFIRMED**, any disagreement or single model error = **UNCERTAIN** (written to `uncertain_review.csv`, excluded from `classified_results.csv`).

---

## Textbook naming convention

Use the format `publisherabbrev_coursename`, all lowercase with underscores:

| Textbook | Name |
|---|---|
| CPM Core Connections Algebra 2 | `cpm_algebra2` |
| Saxon Math Course 1 | `saxon_course1` |
| Saxon Math Course 2 | `saxon_course2` |
| Saxon Math Course 3 | `saxon_course3` |
| Saxon Algebra 1 | `saxon_algebra1` |
| Big Ideas Math Algebra 1 | `bigideas_algebra1` |

The name you pass to `--name` becomes the folder under `data/` and should match the PDF filename in `pdfs/` for clarity.
