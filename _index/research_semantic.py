#!/usr/bin/env python
"""Semantic search over the OCR research corpus.

Embeds the query with the same multilingual model as embed_research.py and
ranks page chunks by cosine similarity — finds pages by meaning, across the
corpus's languages (an English query can land on a Latin or German page).

Usage:
  _index/.venv/bin/python _index/research_semantic.py "plants believed to scream when uprooted"
  _index/.venv/bin/python _index/research_semantic.py --book anatomy "diseases of the skull"
  _index/.venv/bin/python _index/research_semantic.py -n 20 "turning base metals into gold"

For exact words/phrases use research_search.py (FTS) instead — it's instant;
this loads a model (~5s).
"""
import argparse
import os
import sqlite3
import sys

import numpy as np

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  'research', 'corpus.db')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('query')
    ap.add_argument('-n', type=int, default=10, help='max pages (default 10)')
    ap.add_argument('--book', help='filter: substring of the book/folder name')
    args = ap.parse_args()

    if not os.path.exists(DB):
        sys.exit('corpus.db not found — run build_research_fts.py + embed_research.py')
    db = sqlite3.connect(DB)
    try:
        model_name = db.execute(
            "SELECT value FROM chunks_meta WHERE key='model'").fetchone()[0]
    except sqlite3.OperationalError:
        sys.exit('no chunk vectors — run embed_research.py first')

    sql = 'SELECT book, page, path, text, embedding FROM chunks'
    params = []
    if args.book:
        sql += ' WHERE book LIKE ?'
        params.append(f'%{args.book}%')
    rows = db.execute(sql, params).fetchall()
    if not rows:
        sys.exit('no chunks match the --book filter')
    mat = np.frombuffer(b''.join(r[4] for r in rows),
                        dtype=np.float32).reshape(len(rows), -1)

    from sentence_transformers import SentenceTransformer
    q = SentenceTransformer(model_name).encode([args.query],
                                               normalize_embeddings=True)[0]
    sims = mat @ q

    best = {}  # one hit per page: keep its best chunk
    for i, (book, page, path, text, _) in enumerate(rows):
        key = path
        if key not in best or sims[i] > best[key][0]:
            best[key] = (sims[i], book, page, path, text)
    hits = sorted(best.values(), reverse=True)[:args.n]

    for sim, book, page, path, text in hits:
        excerpt = text if len(text) <= 220 else text[:220] + '…'
        print(f'\033[1m{book}\033[0m · {page}  ({sim:.3f})')
        print(f'  {excerpt}')
        print(f'  ./{path}')
        print()
    print(f'{len(hits)} page(s)')


if __name__ == '__main__':
    main()
