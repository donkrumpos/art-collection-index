#!/usr/bin/env python
"""OCR the collection's scanned book pages into a research corpus.

Reads document pages from index.db (facet kind IN text page/title page/
text with figure), grouped by source folder per documents-manifest.json,
OCRs each page with Apple Vision (ocrmac, on-device), and writes one
Markdown file per book under _index/research/.

Resumable: per-page results append to research/.pages/<slug>.jsonl as they
finish; already-done pages are skipped on restart. The .md is (re)rendered
from the jsonl when a book completes.

Run from the collection root:  _index/.venv/bin/python _index/ocr_corpus.py
"""
import json
import os
import re
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, '_index')
RESEARCH = os.path.join(INDEX, 'research')
PAGES_DIR = os.path.join(RESEARCH, '.pages')
LOG = os.path.join(RESEARCH, 'ocr-log.txt')
MIN_CHARS = 40
DOC_TAGS = ('text page', 'title page', 'text with figure')


def slugify(folder):
    s = folder.lower()
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    return s[:120] or 'untitled'


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def load_sets():
    manifest = json.load(open(os.path.join(INDEX, 'documents-manifest.json')))
    db = sqlite3.connect(os.path.join(INDEX, 'index.db'))
    q = ("SELECT path FROM facets WHERE facet='kind' AND tag IN (?,?,?)")
    by_dir = {}
    for (path,) in db.execute(q, DOC_TAGS):
        by_dir.setdefault(os.path.dirname(path), []).append(path)
    db.close()
    sets = []
    for s in manifest['sets']:
        folder = s['source_folder']
        pages = sorted(by_dir.get(folder, []), key=os.path.basename)
        if pages:
            sets.append((folder, pages))
    return sets


def load_done(jsonl_path):
    done = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        done[rec['file']] = rec
                    except json.JSONDecodeError:
                        pass  # torn write from an interrupted run
    return done


def render_md(folder, slug, pages, records):
    kept = skipped = errors = 0
    out = [f'# {folder}', '',
           f'> OCR corpus (Apple Vision, on-device). Source folder: `{folder}`',
           '']
    for path in pages:
        rec = records.get(os.path.basename(path))
        if rec is None:
            continue
        if rec.get('error'):
            errors += 1
            continue
        if rec['chars'] < MIN_CHARS:
            skipped += 1
            continue
        kept += 1
        out.append(f'--- page: {rec["file"]} ---')
        out.append('')
        out.append(rec['text'].rstrip())
        out.append('')
    out.insert(3, f'> Pages kept: {kept} · skipped (<{MIN_CHARS} chars): '
                  f'{skipped} · errors: {errors}')
    out.insert(4, '')
    with open(os.path.join(RESEARCH, slug + '.md'), 'w') as f:
        f.write('\n'.join(out))
    return kept, skipped, errors


def main():
    os.makedirs(PAGES_DIR, exist_ok=True)
    from ocrmac import ocrmac
    sets = load_sets()
    if len(sys.argv) > 1:  # optional substring filter for testing
        sets = [s for s in sets if sys.argv[1].lower() in s[0].lower()]
    total_pages = sum(len(p) for _, p in sets)
    log(f'START run: {len(sets)} books, {total_pages} pages')
    t_run = time.time()
    for i, (folder, pages) in enumerate(sets, 1):
        slug = slugify(folder)
        jsonl_path = os.path.join(PAGES_DIR, slug + '.jsonl')
        done = load_done(jsonl_path)
        todo = [p for p in pages if os.path.basename(p) not in done]
        if todo:
            t0 = time.time()
            with open(jsonl_path, 'a') as jf:
                for path in todo:
                    rec = {'file': os.path.basename(path), 'path': path}
                    try:
                        ann = ocrmac.OCR(os.path.join(ROOT, path),
                                         recognition_level='accurate').recognize()
                        text = '\n'.join(t for t, conf, bbox in ann)
                        rec['chars'] = len(text)
                        rec['text'] = text
                    except Exception as e:
                        rec['chars'] = 0
                        rec['text'] = ''
                        rec['error'] = f'{type(e).__name__}: {e}'
                    jf.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    jf.flush()
                    done[rec['file']] = rec
            dt = time.time() - t0
        else:
            dt = 0.0
        kept, skipped, errors = render_md(folder, slug, pages, done)
        log(f'[{i}/{len(sets)}] {folder}: {len(pages)} pages -> '
            f'kept {kept}, skipped {skipped}, errors {errors} '
            f'({len(todo)} new, {dt:.0f}s)')
    log(f'DONE in {(time.time() - t_run) / 60:.1f} min')


if __name__ == '__main__':
    sys.exit(main())
