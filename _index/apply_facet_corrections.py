#!/usr/bin/env python
"""Apply OCR-derived corrections to the `kind` facet in index.db.

Pages the CLIP pass tagged as documents (text page / title page / text with
figure) whose OCR found <= MAX_CHARS characters are retagged `artwork`.
Conservative by design: the 3-39 char pages (captions, plate numbers) are
left alone.

Every change is logged to research/facet-corrections.csv
(path, old_tag, new_tag, chars, score) so it can be reverted.

Usage:
  _index/.venv/bin/python _index/apply_facet_corrections.py          # dry run
  _index/.venv/bin/python _index/apply_facet_corrections.py --apply
"""
import csv
import glob
import json
import os
import sqlite3
import sys

INDEX = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(INDEX, 'research', '.pages')
LOG_CSV = os.path.join(INDEX, 'research', 'facet-corrections.csv')
DOC_TAGS = ('text page', 'title page', 'text with figure')
MAX_CHARS = 2


def main():
    apply = '--apply' in sys.argv

    by_path = {}  # last record wins
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

    near_zero = {p for p, r in by_path.items()
                 if not r.get('error') and r.get('chars', 0) <= MAX_CHARS}

    db = sqlite3.connect(os.path.join(INDEX, 'index.db'))
    rows = db.execute(
        "SELECT path, tag, score FROM facets WHERE facet='kind' "
        "AND tag IN (?,?,?)", DOC_TAGS).fetchall()
    targets = [(p, t, s) for p, t, s in rows if p in near_zero]
    print(f'{len(targets)} pages to retag as artwork '
          f'(OCR <= {MAX_CHARS} chars){"" if apply else "  [DRY RUN]"}')

    if apply and targets:
        with open(LOG_CSV, 'a', newline='') as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(['path', 'old_tag', 'new_tag', 'chars', 'score'])
            for path, tag, score in targets:
                db.execute("DELETE FROM facets WHERE path=? AND facet='kind' "
                           "AND tag=?", (path, tag))
                # source='ocr' so auto_tag.py's clip-scoped rerun DELETEs
                # don't wipe these corrections
                db.execute("INSERT OR IGNORE INTO facets(path, facet, tag, score, source) "
                           "VALUES (?, 'kind', 'artwork', ?, 'ocr')", (path, score))
                w.writerow([path, tag, 'artwork',
                            by_path[path].get('chars', 0), score])
        db.commit()
        print(f'applied; log appended to {LOG_CSV}')
    else:
        for path, tag, _ in targets[:10]:
            print(f'  {tag:16} -> artwork  {path}')
        if len(targets) > 10:
            print(f'  ... and {len(targets) - 10} more')
    db.close()


if __name__ == '__main__':
    main()
