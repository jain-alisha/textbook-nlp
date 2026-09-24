# Textbook Error Pedagogy Classifier

A pipeline to extract paragraphs from math textbook PDFs and classify them for error-based pedagogy
using a two-arm cross-vendor ensemble: `gemini-3.8-flash` (Google) and `gpt-oss-120b` (Groq). The arms
are deliberately from different vendors and model families so they fail independently.

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

**Era 4 — `gemini-3.8-flash`** (2026-09-24+). Same Era 3 machinery, different model. `gemini-2.5-flash`
is closed to new API projects (a 404, not a quota error), i.e. soft-deprecated, and a head-to-head
showed 3.8 is also the better extractor for this task — see the changelog entry for 2026-09-24.

**Per-book status is therefore:** everything from Era 1, plus the three CK-12 books, requires
re-extraction. Era 2 CPM books are usable, but re-extracting them under Era 4 as well is the only way
to put the whole corpus on one extractor.

---

## Changelog

Newest first. This records *why* things changed, which git history alone doesn't preserve — several
entries below are reversals of earlier decisions, and the reasoning is what keeps them from being
re-litigated. Add an entry whenever a pipeline stage, model, or corpus decision changes.

### 2026-09-24 — Extraction moved to `gemini-3.8-flash`

`gemini-2.5-flash` now returns 404 for newly created API projects ("no longer available to new
users"), so it is on a deprecation path. A head-to-head on the same 25 pages of
`ck12_algebra1_hs`, same prompt and config, through the production code path:

| | 2.5-flash | 3.8-flash |
|---|---|---|
| paragraphs | 264 | 93 |
| median chars | 84 | 283 |
| paragraphs/page | 10.6 | 3.7 |
| wall clock | 110s | 54s |

The counts understate it; the difference is in what a paragraph *is*. 2.5 split a worked solution
across one row per line and gave each bullet of a properties list its own row. 3.8 returned
`Example 2` whole with its solution, and `Review Questions` problem 1 whole with parts a–h. **That
fragmentation is what `stitch.py` exists to repair**, so a better extractor shrinks the most
failure-prone stage in the pipeline rather than feeding it.

Cost, measured per page and extrapolated (±30%; the API returned `candidates_token_count: None`, so
payload tokens are derived from character counts):

| | extract | + Gemini classify arm | total | wall clock |
|---|---|---|---|---|
| 7 defective books, 2.5 | $12.90 | $26.52 | ~$39 | 2.4 h |
| 7 defective books, 3.8 | $21.28 | $9.34 | ~$31 | 1.2 h |
| all 12 books, 3.8 | $36.16 | $15.88 | ~$52 | 2.0 h |

3.8 costs 1.6x more per page but yields 2.8x fewer paragraphs, and classification is billed per
paragraph across two arms — so it is cheaper end to end.

**Known tradeoff:** coarser paragraphs change the unit of analysis. A 2,300-char chunk can contain
error pedagogy *and* ordinary content, so one label per paragraph is lossier than before. It does
move CK-12 closer to the CPM books' granularity, which helps cross-publisher comparison.

### 2026-09-23 — CPM edition split and extraction eras documented

Established that the CPM corpus straddles 2nd and 3rd editions, and that extractor identity is not
recoverable from the older data files. See the two sections above.

### 2026-09-22 — Classifier ensemble rebuilt around vendor independence

`qwen/qwen3-32b` was decommissioned, which silently split the ensemble: labels written before and
after the change came from different models. The Qwen arm was replaced with **Gemini** rather than a
second Groq model, keeping `gpt-oss-120b` as the stable anchor, so the two arms fail independently
instead of sharing a vendor and a model family. `merge.py` was generalised to any two named arms, and
each run now writes `<model>_manifest.json` recording the exact model string.

A 50-paragraph drift probe set was frozen — then **voided**, because every paragraph in it came from
a book queued for re-extraction. It must be rebuilt from books that have stopped moving.

### 2026-09-21 — Provenance recorded in the data, not the logs

Every paragraph now carries a `source` column (`gemini` / `pymupdf` / `mixed`). Previously a book
that fell back to PyMuPDF looked identical to one that didn't.

`stitch.py` stopped treating repeated failures as "these paragraphs are separate" — 197 Ollama JSON
parse failures had become 39 silent forced splits. Unresolved pairs now go to `stitch_unresolved.csv`.

### 2026-09-20 — Stitching moved to local Ollama (free)

`stitch.py` gained `--backend ollama` (`qwen3:14b`, local). Validated at 92% agreement with the paid
backend on real-content pairs. Stitching had been the pipeline's main running cost.

### 2026-09-18 — Extraction hardened; classification cached

- Per-chunk retry → halve the chunk → PyMuPDF **for those pages only**. Previously one bad chunk sent
  the entire book to PyMuPDF.
- Abort rather than save if >20% of pages fall back, instead of silently shipping degraded output.
- Daily-quota errors now stop the run (exit 3) instead of filling the output with error rows.
- Content-hash cache on classification keyed by model + prompt fingerprint + paragraph. **ERROR
  results are deliberately not cached** — caching a failure would bake it in permanently.
- The stitch prompt's completion was reduced to a boolean with joining done in code: ~875 completion
  tokens per pair down to ~6.
- `MAX_MERGED_CHARS = 15000` cap added after runaway merge chaining produced a single 601,552-char
  chunk and consumed 87% of a $5 Groq spend.

### 2026-09-17 — Corpus expanded and renamed

Six CPM and three CK-12 books added. All books renamed to `publisher_course_level` with explicit
`_ms` / `_hs` tags. Added `clean_paragraphs.py` to strip the CPM per-page watermark and bullet runs.

### 2026-03-28 → 2026-04-06 — Initial pipeline

Groq ensemble classifier, per-textbook folder isolation, unchunked Gemini extraction. Everything
extracted in this window is Era 1 and requires re-extraction.
