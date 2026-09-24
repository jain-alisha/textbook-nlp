# pdfs/ — intentionally empty in this repo

`scripts/extract.py` reads source textbook PDFs from this directory, but **no PDF is committed here
and none ever should be.** `.gitignore` excludes `*.pdf` repo-wide.

## Why

These are copyrighted, commercially published textbooks. The CPM files additionally carry a per-page
license notice stating the materials are provided for research use only, may not be used for
commercial purposes, and may not be distributed on the internet. One Saxon file was obtained from a
shadow library.

The derived research artifacts — extracted paragraphs and classifications under `data/` — are
transformative outputs and are fine to publish. The source scans are not.

To reproduce the pipeline you must supply your own copies of the books, named as below.

## Naming convention

`publisherabbrev_coursename_level.pdf`, where level is `ms` (middle school) or `hs` (high school),
matching the `--name` identifier used in the pipeline — so `cpm_course1_ms.pdf` feeds
`data/cpm_course1_ms/`. See the table in the top-level README for the full list.

## Notes on specific files

- `cpm_algebra2_hs.pdf` (531pp) and `cpm_algebra2_hs_v2.pdf` (896pp) are different editions of the
  same title. **v2 is canonical**; the 531pp file is an OCR-garbled partial volume and is not used.
  See "Known limitation: the CPM corpus spans two editions" in the top-level README.
- `saxon_course2_ms.pdf` is an image-only scan with no extractable text layer, and postdates the
  extraction in `data/saxon_course2_ms/` — that older data cannot be reproduced from this file.
