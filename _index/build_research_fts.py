#!/usr/bin/env python
"""Build the FTS5 search index over the OCR research corpus.

Reads _index/research/.pages/*.jsonl (written by ocr_corpus.py) and
(re)builds _index/research/corpus.db with one row per kept page.
Idempotent: drops and rebuilds every run.

Usage:  _index/.venv/bin/python _index/build_research_fts.py
"""
import glob
import json
import os
import sqlite3

INDEX = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.join(INDEX, 'research')
PAGES_DIR = os.path.join(RESEARCH, '.pages')
DB = os.path.join(RESEARCH, 'corpus.db')
MIN_CHARS = 40


def main():
    db = sqlite3.connect(DB)
    db.executescript("""
        DROP TABLE IF EXISTS pages;
        CREATE VIRTUAL TABLE pages USING fts5(
            book,           -- source folder (the book)
            page,           -- page image filename
            path UNINDEXED, -- path relative to collection root
            text            -- OCR'd page text
        );
    """)
    books = pages = 0
    for jsonl in sorted(glob.glob(os.path.join(PAGES_DIR, '*.jsonl'))):
        by_path = {}  # last record wins (retries append corrected rows)
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    by_path[rec['path']] = rec
                except json.JSONDecodeError:
                    continue
        n = 0
        for rec in by_path.values():
            if rec.get('error') or rec.get('chars', 0) < MIN_CHARS:
                continue
            book = os.path.dirname(rec['path'])
            db.execute('INSERT INTO pages(book, page, path, text) VALUES (?,?,?,?)',
                       (book, rec['file'], rec['path'], rec['text']))
            n += 1
        if n:
            books += 1
            pages += n
    db.commit()
    db.close()
    print(f'corpus.db rebuilt: {pages} pages across {books} books')


if __name__ == '__main__':
    main()
