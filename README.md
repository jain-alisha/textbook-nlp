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
    stitch.py                    # rejoin paragraphs split across chunk boundaries
    clean_paragraphs.py          # strip watermarks / bullet runs
    classify_single.py           # one arm -> <arm>_results.csv          (stage 1)
    plan_control.py              # proportional control allocation -> control_plan.json
    route.py                     # stage-1 labels -> stage2_worklist.csv (tiers A/B/C)
    merge.py                     # -> classified_results.csv + findings.csv
    collect_findings.py          # all books -> findings_all.csv
    stage2_report.py             # per-book agreement + recall bound
    stage2_corpus_report.py      # per-series + corpus bounds, comparability check
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

### 2. Classify — the two-stage pipeline

Stage 1 screens every paragraph with Gemini. Run it for every book first, because the
control allocation in step 2b is computed across the whole corpus.

```bash
# 2a. stage 1, per book
python scripts/classify_single.py --name cpm_algebra2_hs --model gemini

# 2b. allocate the control sample proportionally across ALL books (once)
python scripts/plan_control.py --target 3000

# 2c. route each book into tiers A/B/C, then verify the subset locally (free)
python scripts/route.py --name cpm_algebra2_hs
python scripts/classify_single.py --name cpm_algebra2_hs --model qwen_local \
    --paragraphs stage2_worklist.csv --out qwen_local_stage2 --sleep 0

# 2d. build the dataset
python scripts/merge.py --name cpm_algebra2_hs
```

Per-book output:
- `classified_results.csv` — every paragraph, with `final_label`, `status`,
  `verification` (`dual_arm` / `single_arm`) and `tier`
- `findings.csv` — the non-NA paragraphs only, with `book` and `para_num`
- `uncertain_review.csv` — dual-arm disagreements
- `dataset_manifest.json` — counts by status, verification and tier

Then corpus-wide:

```bash
python scripts/collect_findings.py            # -> data/findings_all.csv
python scripts/stage2_corpus_report.py        # per-series + corpus recall bounds
```

Requires `ollama serve` running with `qwen3:14b` pulled for the stage-2 arm.

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

**First full run — `ck12_algebra1_hs`, 714 pages, 50 pages/chunk:** 1,811 paragraphs (2.54/page,
median 395 chars), of which 1,805 from Gemini and **6 from PyMuPDF — 1% of pages**. Contrast the
previous Era 2 run of the same book: 713 paragraphs whose first two rows were the copyright page.
The new run drops front matter entirely and opens on real content.

Two things the 25-page benchmark had not revealed:

1. **RECITATION blocks are common on copyrighted textbook content** — 4 of the 13 initial 50-page
   chunks (31%) came back with a non-STOP finish reason. This is the model declining to reproduce
   the text, not an error.
2. **Halving reliably resolves them, because the refusal is localised.** Every blocked range
   recovered at 25 or 12 pages except one 6-page stretch (363–368). The split ladder is therefore
   doing real work — it isolates the few pages the model won't touch and salvages the rest — and it
   is the reason a 31% block rate cost only 1% fallback. Expect ~1.7x the request count from splits
   when budgeting.

**Stitching is now nearly redundant for 3.8-extracted books.** Only 3 of 1,805 Gemini paragraphs
(0.17%) begin mid-sentence, about the number of chunk boundaries in the run; the other three
mid-sentence rows are PyMuPDF's. A full `stitch.py` pass over 1,811 paragraphs to fix 3 of them is a
poor trade given that stage's history. A boundary-only pass — testing just the last and first
paragraph of adjacent chunks — would cover the same defects in ~20 comparisons.

### 2026-09-24 — Two-stage classification; the regex prefilter failed instructively

**The negative result first, because it is the more useful half.**

To avoid paying for two LLM arms over ~25,000 paragraphs, a deliberately broad
regex prefilter was built to pass only plausible candidates. On `ck12_algebra1_hs`
it passed 12.7%, which looked like an 8x saving. It does not survive contact with
the rest of the corpus.

*It does not transfer across series.* Per-paragraph pass rate ranges from 2.6%
(`saxon_course2_ms`) to 44.5% (`cpm_geometry_hs`) — a 17-fold spread:

| series | paragraphs | pass % | hits/1k chars |
|---|---|---|---|
| cpm | 6,147 | 28.4% | 0.54 |
| ck12 | 3,463 | 17.3% | 0.33 |
| saxon | 13,200 | 4.4% | 0.26 |

Normalising per 1,000 characters (necessary, because the eras extract at very
different granularities) narrows but does not close the gap: CPM still shows ~2x
the trigger density. Worse, the subpatterns `who is correct` and `students often`
fire **only** in CPM — zero occurrences in CK-12 and Saxon. The filter encodes
CPM's vocabulary, so a CPM-vs-others finding would be partly an artifact of the
instrument.

*And it cannot be cheaply validated — this is the decisive part.* An audit labelled
80 filter-accepted and 80 filter-rejected paragraphs per series with the Gemini
arm. It found 0 false negatives, which sounds like perfect recall and is nearly
uninformative. By the rule of three, 0/80 puts the 95% upper bound on the miss rate
at 3.75%; against the real rejected-pool sizes that permits:

| series | rejected pool | FN upper bound | TP | recall lower bound |
|---|---|---|---|---|
| ck12 | 1,572 | 59 | 12 | **16.9%** |
| cpm | 501 | 19 | 13 | **41.1%** |
| saxon | 724 | 27 | 5 | **15.9%** |

The audit is equally consistent with the filter catching everything and with it
missing five positives in six. The reason is the base rate: error pedagogy runs at
**0.55%–1.72% of paragraphs**. Bounding false negatives tightly at that rarity
requires labelling essentially the whole rejected pool — which is the census the
filter existed to avoid.

**So the flaw is structural, not a matter of better trigger words.** Any cheap
prefilter for a ~1% phenomenon is unfalsifiable by construction. No iteration on
the regex escapes that, and the filter was dropped rather than tuned.

**Replacement: the LLM screens, and routing is explicit.** Stage 1 runs the Gemini
arm over every paragraph (~$16 corpus-wide, no vocabulary assumption to validate).
Stage 2 runs a local, free `qwen3:14b` arm over a routed subset — `scripts/route.py`
assigns every paragraph to one of:

| tier | rule | purpose |
|---|---|---|
| **A positive** | stage-1 label != NA | verify the findings |
| **B boundary** | NA, but confidence < high **or** a category was `considered` | the screener's own hesitation |
| **C control** | confident NA, nothing considered | random sample; bounds what the design misses |

Tier B required changing the prompt. It previously ended *"When in doubt between an
error category and NA, always choose NA"* — which collapsed boundary cases into
plain `NA` and recorded nothing. Merely conservative under a census; under routing
it deletes exactly the cases the second arm is for. The prompt now returns
`confidence` and `considered`, and a smoke test put **8 of 22 paragraphs in tier B**
— all previously indistinguishable from confident `NA`. Cache entries lacking the
new fields read as `unknown`, which routes them to stage 2 rather than letting a
missing field pass as confident.

**Tier C uses stratified sampling with proportional allocation.** Strata are books,
and each book's control is strictly proportional to its confident-NA pool:

    n_b = N × P_b / Σ P

`scripts/plan_control.py --target N` computes the allocation across every book with
stage-1 results and writes `data/control_plan.json`; `route.py` reads that plan
rather than taking a per-book number. Integer shares use the largest-remainder
method so they sum to exactly `N`, and a book whose share exceeds its pool is capped
with the freed quota redistributed, so capping never silently shrinks the total.

Proportional allocation is the point, not a convenience: every confident-NA
paragraph in the corpus gets the same inclusion probability `N/ΣP` regardless of
which book it sits in, so pooled and per-series estimates are unbiased without
reweighting — and **each series' share of the control automatically equals its share
of the pool**, which is what the per-series bounds need. `route.py` records
`control_source` in `stage2_strata.json`, and `stage2_corpus_report.py` warns if any
book was routed with an explicit `--control-n`, since that breaks proportionality and
biases pooled estimates toward the over-sampled book.

**The bound is reported per series, not per book.** An earlier per-book default of
2,000 was wrong in a way worth recording: correct at corpus level, it silently became
a *full census* per book, because a 1,811-paragraph book has only ~1,540
confident-NA paragraphs. That is 82 hours of local inference instead of ~21. The
per-book floor check in `route.py` is now advisory only — no claim in this study is
of the form "in Saxon specifically, recall was Y%", so a per-book guarantee costs
time for nothing.

Per series is nevertheless the right unit, and *not* a single pooled number. Every
headline claim here is cross-series ("CPM uses more error pedagogy than Saxon"), and
a pooled 90% floor is perfectly compatible with 96% recall on CPM and 65% on Saxon —
differential recall would masquerade as a finding. That is the same failure mode that
killed the prefilter, so `stage2_corpus_report.py` reports each series' miss rate
with a Wilson interval and states plainly whether the intervals overlap.

Bounds are given in both forms, because the relative one misleads at this base rate:

| form | example | note |
|---|---|---|
| absolute | "at most 19 missed paragraphs in a 5,400 pool" | the honest statement |
| relative | "recall ≥ 65%" | unstable: reads low only because ~36 positives is a small denominator |

Sizing note, from the measured base rates: a 90% *per-series* floor would need 4,420
control paragraphs for CK-12 (87% of its pool) and 5,310 for Saxon (97%) — it
degenerates into the census, because the bound is limited by how few positives exist
rather than by control size. Only CPM, with ~3x the positives, saves anything. A
target around 3,000 (~10 h) gives per-series absolute bounds that are worth stating;
tightening the relative floor further is not worth the hours, because a
human-labelled stratified probe has to underwrite that number anyway.

**Kappa is now reported over the routed strata, not corpus-wide.** This is a design
choice, not a limitation. Over a corpus that is ~99% `NA`, agreement is dominated
by both arms trivially concurring on obvious non-cases, which inflates kappa
without evidencing reliability on the judgement that matters.

**Groq is no longer required.** It is blocked by a spend *alert* threshold rather
than exhausted funds, but the second arm is now local and free either way. Also
checked: OpenRouter serves `qwen/qwen3.8-27b:free` — the exact model — but caps
free use at 50 requests/day, so it cannot serve a census. Local `qwen3:14b` was
measured at 11.7s/paragraph, which is ~80 hours for a census and a few hours for a
routed subset. The two-stage design is what makes a free arm viable.

`merge.py` produces the per-book dataset from the routed results. Every paragraph
appears in `classified_results.csv` with a **`verification`** column: `dual_arm`
where both arms saw it (tiers A, B and the sampled part of C), `single_arm` for the
confident-NA majority that only Gemini ever saw. That majority is real and is
recorded per row rather than left implicit in the routing arithmetic — **any count
restricted to verified labels must exclude `single_arm` rows.** `status` is
`CONFIRMED` / `UNCERTAIN` / `UNVERIFIED`, and `tier` distinguishes `C_control` from
`C_unsampled`.

Each book also gets **`findings.csv`** — every non-NA paragraph with `book`,
`para_num` (its 1-based row in the extraction it came from, the only stable
per-paragraph identifier the pipeline has), `final_label`, `status`, `verification`
and the full text. `scripts/collect_findings.py` concatenates these into
`data/findings_all.csv` and prints per-book rates per 1,000 paragraphs, since raw
counts favour whichever book was split into more paragraphs. `--verified-only`
excludes `single_arm` rows.

`merge.py` also refuses to build a dataset when a label file's paragraph set does
not match the extraction its manifest names — re-extracting a book replaces its
paragraphs wholesale, and nothing in the CSVs themselves reveals that the labels
now describe discarded text. `--allow-stale` overrides it.

**A retracted concern, kept because the mistake is instructive.** During the recall
audit `qwen3:14b` labelled a paragraph beginning *"Example 6: Find the opposite of
each of the following"* as `COMMON_ERROR_ALERT`, which was recorded here as a likely
misclassification and evidence that a 14B model might be too weak for the arm. It
was not a misclassification. The paragraph ends: *"A common mistake in this example
is to assume that the opposite of (x−3) is (x+3). Avoid this mistake!"* The label is
correct, and the judgement was wrong because it was made from a 70-character
preview rather than the paragraph. Both arms independently label it correctly and
cite that sentence.

This also happens to be direct evidence for the Era 4 extraction change: the
error-pedagogy signal sits in the last sentence of a 700-character unit that opens
as a routine worked example. Under 2.5-flash's fragmentation the example and the
warning would very likely have become separate paragraphs, leaving the example
labelled `NA` and the pedagogy attached to a stray fragment. Coarser units preserve
the context the label depends on.

**Still outstanding:** `qwen3:14b` has not been validated for the job it now has —
discriminating among candidates the screener already flagged, rather than scanning
raw prose cold. That needs a probe built for the new task, with human labels, not
the voided earlier one and not more model-vs-model agreement.

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
