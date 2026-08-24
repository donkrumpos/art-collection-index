# Handoff: OCR → research corpus over the Art Collection's scanned books

Paste the block below into a fresh Claude Code session started in
`~/Documents/Art Collection`. It's a self-contained brief for the OCR pass —
a different kind of work (text/RAG) than the image catalog, so it gets its own
session.

---

```
Build an on-device OCR → research corpus over the scanned book pages already
detected in this Art Collection. Everything must stay local (Apple Vision OCR),
no cloud. Do NOT move, rename, or delete any original image files.

WHAT ALREADY EXISTS (built in a prior session — read these first):
- ~/Documents/Art Collection/_index/index.db — SQLite. Table `facets` has a
  `kind` facet; rows where tag IN ('text page','title page','text with figure')
  are the scanned document pages (~6,400). tag='blank' = empty pages to skip.
- ~/Documents/Art Collection/_index/documents-manifest.json — those pages
  grouped by source folder (one folder ≈ one book), sets of >=5 pages, sorted
  by page count. Start from here.
- ~/Documents/Art Collection/_index/.venv — Python 3.12 venv (PIL, numpy).
- Originals live at ~/Documents/Art Collection/<folder>/... (paths in the DB are
  relative to that root).

THE TASK:
1. Add OCR to the venv:  _index/.venv/bin/pip install ocrmac
   (ocrmac wraps Apple's Vision framework — offline, high-quality, fast on M-series.)
2. For each source set in documents-manifest.json, OCR its pages IN FILENAME
   ORDER. Concatenate into one Markdown file per book at
   _index/research/<slug-of-folder>.md, with a `--- page: <filename> ---` marker
   before each page's text.
3. Skip pages whose OCR yields < ~40 characters — that's the definitive blank/
   non-text filter. Log how many were skipped per book.
4. Build a search surface over the corpus:
   - Minimum: a SQLite FTS5 table (book, page, text) + a small query script
     `research_search.py "phrase"` returning book + page + snippet.
   - Nice: also embed each page with a local text model for semantic search,
     mirroring how _index/serve.py does image search.
5. Feed accuracy back (optional but valuable): pages with lots of OCR text are
   confirmed documents; pages with near-zero are artwork/blank — write a small
   report of any misclassifications so the image catalog's `kind` facet can be
   corrected.

CONSTRAINTS:
- On-device only. Owned files. Originals are read-only to you.
- Batch by book; commit/write incrementally so it's resumable over ~6,400 pages.
- Priority order = the manifest's order (biggest books first), e.g.
  "De historia stirpium" (1542 botanical herbal), "modernalchemy",
  "Vintage medical and anatomy", "The Morbid Anatomy Gallery".

DELIVERABLE: _index/research/ with one .md per book + an FTS index + a query
script, plus a one-paragraph summary of what got OCR'd and what was skipped.
```

---

Where this fits: PLAN.md is about *image* findability. This corpus is the
*text* dividend from the same scans — a separate, parallel asset. Keeping it
under `_index/research/` means it rides along in the collection's backups.
