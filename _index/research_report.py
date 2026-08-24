#!/usr/bin/env python
"""Feed OCR results back to the image catalog: misclassification report.

Pages the CLIP pass tagged as documents (text page / title page / text with
figure) but whose OCR found near-zero text are probably artwork or blank —
candidates for correcting the `kind` facet in index.db. Pages with real text
are OCR-confirmed documents.

Writes _index/research/misclassification-report.md. Read-only: does NOT
modify index.db.

Usage:  _index/.venv/bin/python _index/research_report.py
"""
import glob
import json
import os
import sqlite3

INDEX = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.join(INDEX, 'research')
PAGES_DIR = os.path.join(RESEARCH, '.pages')
OUT = os.path.join(RESEARCH, 'misclassification-report.md')
NEAR_ZERO = 10   # < this many chars → almost certainly not a text page
LOW = 40         # the corpus skip threshold


def main():
    db = sqlite3.connect(os.path.join(INDEX, 'index.db'))
    kind = dict(db.execute(
        "SELECT path, tag FROM facets WHERE facet='kind' "
        "AND tag IN ('text page','title page','text with figure')"))
    db.close()

    by_path = {}  # last record wins (retries append corrected rows)
    for jsonl in glob.glob(os.path.join(PAGES_DIR, '*.jsonl')):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    by_path[rec['path']] = rec
                except json.JSONDecodeError:
                    pass
    recs = list(by_path.values())

    confirmed = [r for r in recs if r.get('chars', 0) >= LOW]
    low = [r for r in recs if NEAR_ZERO <= r.get('chars', 0) < LOW
           and not r.get('error')]
    zero = [r for r in recs if r.get('chars', 0) < NEAR_ZERO
            and not r.get('error')]
    errors = [r for r in recs if r.get('error')]

    by_folder = {}
    for r in zero:
        by_folder.setdefault(os.path.dirname(r['path']), []).append(r)

    lines = [
        '# OCR feedback: `kind` facet misclassification candidates', '',
        f'Of {len(recs)} pages tagged as documents by the CLIP pass:', '',
        f'- **{len(confirmed)}** OCR-confirmed documents (≥{LOW} chars of text)',
        f'- **{len(low)}** marginal ({NEAR_ZERO}–{LOW - 1} chars — captions, '
        'plate numbers; plausibly "text with figure" but not corpus material)',
        f'- **{len(zero)}** near-zero (<{NEAR_ZERO} chars) — **likely artwork or '
        'blank, candidates for facet correction**',
        f'- **{len(errors)}** OCR errors (unreadable files)', '',
        '## Near-zero pages by folder', '',
        'These carry a document `kind` tag in index.db but contain almost no '
        'machine-readable text. Suggested correction: retag as `artwork` or '
        '`blank` after spot-checking.', '',
    ]
    for folder in sorted(by_folder, key=lambda k: -len(by_folder[k])):
        rs = by_folder[folder]
        lines.append(f'### {folder} — {len(rs)} page(s)')
        lines.append('')
        for r in sorted(rs, key=lambda r: r['file']):
            tag = kind.get(r['path'], '?')
            lines.append(f'- `{r["path"]}` (tagged *{tag}*, {r["chars"]} chars)')
        lines.append('')
    if errors:
        lines.append('## OCR errors')
        lines.append('')
        for r in sorted(errors, key=lambda r: r['path']):
            lines.append(f'- `{r["path"]}` — {r["error"]}')
        lines.append('')

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines))
    print(f'report written: {OUT}')
    print(f'confirmed {len(confirmed)} · marginal {len(low)} · '
          f'near-zero {len(zero)} · errors {len(errors)}')


if __name__ == '__main__':
    main()
