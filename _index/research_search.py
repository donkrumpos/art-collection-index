#!/usr/bin/env python
"""Full-text search over the OCR research corpus.

Usage:
  _index/.venv/bin/python _index/research_search.py "mandrake root"
  _index/.venv/bin/python _index/research_search.py --book alchemy "mercur*"
  _index/.venv/bin/python _index/research_search.py -n 30 "krampus"

Query syntax is SQLite FTS5: phrases in quotes, AND/OR/NOT, prefix*.
Results: book · page filename · snippet. The path shown is relative to
the collection root, so the original scan is easy to open.
"""
import argparse
import os
import sqlite3
import sys

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  'research', 'corpus.db')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('query', help='FTS5 query (phrases in quotes, AND/OR/NOT, prefix*)')
    ap.add_argument('-n', type=int, default=15, help='max results (default 15)')
    ap.add_argument('--book', help='filter: substring of the book/folder name')
    args = ap.parse_args()

    if not os.path.exists(DB):
        sys.exit('corpus.db not found — run build_research_fts.py first')
    db = sqlite3.connect(DB)
    sql = ("SELECT book, page, path, snippet(pages, 3, '>>', '<<', ' … ', 18) "
           "FROM pages WHERE pages MATCH ?")
    params = [args.query]
    if args.book:
        sql += " AND book LIKE ?"
        params.append(f'%{args.book}%')
    sql += " ORDER BY rank LIMIT ?"
    params.append(args.n)
    try:
        rows = db.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        sys.exit(f'query error: {e}')
    if not rows:
        print('no matches')
        return
    for book, page, path, snip in rows:
        print(f'\033[1m{book}\033[0m · {page}')
        print(f'  {" ".join(snip.split())}')
        print(f'  ./{path}')
        print()
    print(f'{len(rows)} result(s)')


if __name__ == '__main__':
    main()
