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
python scripts/extract.py pdfs/cpm_algebra2_hs.pdf --name cpm_algebra2_hs
```

Output: `data/cpm_algebra2_hs/paragraphs.csv`

### 2. Classify paragraphs

```bash
python scripts/classify.py --name cpm_algebra2_hs
```

Output:
- `data/cpm_algebra2_hs/classified_results.csv` — confirmed (2/2) votes
- `data/cpm_algebra2_hs/uncertain_review.csv` — rows where models disagreed or errored
- `data/cpm_algebra2_hs/progress.json` — checkpoint; re-running skips already-classified paragraphs

### 3. Analyze results (stub)

```bash
python scripts/analyze.py --name cpm_algebra2_hs
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

Use the format `publisherabbrev_coursename_level`, all lowercase with underscores, where `level` is
`ms` (middle school) or `hs` (high school):

| Textbook | Level | Name |
|---|---|---|
| CPM Core Connections Course 1 | Middle school | `cpm_course1_ms` |
| CPM Core Connections Course 2 | Middle school | `cpm_course2_ms` |
| CPM Core Connections Course 3 | Middle school | `cpm_course3_ms` |
| CPM Core Connections Algebra 1 | High school | `cpm_algebra1_hs` |
| CPM Core Connections Algebra 2 | High school | `cpm_algebra2_hs` |
| CPM Core Connections Geometry | High school | `cpm_geometry_hs` |
| Saxon Math Course 1 | Middle school | `saxon_course1_ms` |
| Saxon Math Course 2 | Middle school | `saxon_course2_ms` |
| Saxon Math Course 3 | Middle school | `saxon_course3_ms` |
| Saxon Algebra 1 | High school | `saxon_algebra1_hs` |
| CK-12 Middle School Math Grade 6 | Middle school | `ck12_grade6_ms` |
| CK-12 Middle School Math Grade 7 | Middle school | `ck12_grade7_ms` |
| CK-12 Algebra I (2nd ed.) | High school | `ck12_algebra1_hs` |
| CK-12 Algebra II with Trigonometry | High school | `ck12_algebra2_hs` |
| CK-12 Geometry (2nd ed.) | High school | `ck12_geometry_hs` |
| Big Ideas Math Algebra 1 | High school | `bigideas_algebra1_hs` |

The name you pass to `--name` becomes the folder under `data/` and should match the PDF filename in `pdfs/` for clarity.
