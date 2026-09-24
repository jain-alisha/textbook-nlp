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

---

## Known limitation: the CPM corpus spans two editions

CPM is mid-rollout of its 3rd Edition, staggered title by title, so the CPM books here are not all
the same edition. Where a cover says "3RD EDITION" the file is unambiguous; the others carry no
edition statement at all, which is what a current-and-only edition looks like before a successor
exists to distinguish it from. At the time of collection CPM's own store listed 3rd Edition Geometry
and Algebra 2 as "coming soon, order now to receive 2nd Edition print materials", while 3rd Edition
Algebra 1 had a ship date.

| Book | Edition marker on the file | Edition |
|---|---|---|
| `cpm_course1_ms`, `cpm_course2_ms`, `cpm_course3_ms` | "3RD EDITION COURSE n" | 3rd |
| `cpm_algebra1_hs` | "3RD EDITION ALGEBRA 1" | 3rd |
| `cpm_geometry_hs` | none | 2nd (inferred) |
| `cpm_algebra2_hs` | none | 2nd (inferred) |

**Consequence for analysis:** any cross-title comparison within CPM (e.g. error-pedagogy rates in
Course 3 vs Geometry) also crosses an edition boundary, and an edition revision can change how much
error-based pedagogy a book contains. Treat CPM-internal differences as confounded with edition
unless the comparison is within one edition group.

A superseded scan, `pdfs/cpm_algebra2_hs.pdf` (531pp), is retained but **not used**: its cover reads
"Second Edition, Version 4.0, Volume 2" while its own contents page lists Volume 1, and its text
layer is OCR-garbled (`y = ax* +bx+c` for `ax²`, "How can! find"). Algebra 2 is extracted from
`pdfs/cpm_algebra2_hs_v2.pdf` (896pp), whose contents run Chapter 1 through Chapter 10 with normal
back matter (appendix p751, glossary p861, index p895) — a complete single volume, at a page count
in line with its siblings (Algebra 1 961pp, Geometry 891pp).

---

## Extraction provenance: three eras

`extract.py` was rewritten twice, and a book's quality depends on which version produced it. This is
recorded here because the extractor identity is **not** recoverable from the older data files — they
predate the `source` column, and no current code path reproduces them.

**Era 1 — unchunked `gemini-2.0-flash`** (through commit `fecec01`, Apr 2026). The entire PDF was
uploaded in a single request; any failure fell back to PyMuPDF for the whole book. Output is heavily
fragmented (median ~105–203 chars/paragraph, 4–8 paragraphs per page). `gemini-2.0-flash` is itself
now scheduled for retirement. Books: `cpm_algebra2_hs` (2026-04-02), `saxon_course1_ms`
(2026-04-03), `saxon_course2_ms` (2026-04-04), `saxon_course3_ms` (2026-04-06).

**Era 2 — chunked `gemini-2.5-flash`** (commit `b4e190f`, 2026-09-17). 50 pages per request, but a
single failed chunk still sent the *whole* book to PyMuPDF, with no guard and nothing recorded.
Books: `cpm_course1_ms`, `cpm_course2_ms`, `cpm_course3_ms`, `cpm_algebra1_hs`, `cpm_geometry_hs`
(all clean, ~1 paragraph/page, median ~1,300 chars) and the three `ck12_*` books, which all
silently fell back to PyMuPDF after 504 timeouts and are **not** usable as extracted.

**Era 3 — current** (commits `c1dc338` onward, 2026-09-18+). Per-chunk retry, then halve the chunk,
then PyMuPDF for only the pages that still fail; the run aborts rather than save if more than 20% of
pages fall back; every paragraph carries a `source` column; a daily-quota error stops the run
instead of filling the output with errors.

**Per-book status is therefore:** Era 2 CPM books are clean and left as-is; everything from Era 1,
plus the three CK-12 books, requires re-extraction under Era 3 before use.
