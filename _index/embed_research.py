#!/usr/bin/env python
"""Embed the OCR research corpus for semantic search.

Chunks each kept page (>=40 chars) into ~600-char pieces and embeds them with
paraphrase-multilingual-MiniLM-L12-v2 (local, MPS) — multilingual so English
queries can land on the corpus's Latin / German / French pages. Vectors go in
research/corpus.db alongside the FTS table.

Idempotent: drops and rebuilds the vectors table. ~4,600 pages -> a few
minutes on M-series.

Usage:  _index/.venv/bin/python _index/embed_research.py
"""
import glob
import json
import os
import sqlite3

import numpy as np

INDEX = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.join(INDEX, 'research')
PAGES_DIR = os.path.join(RESEARCH, '.pages')
DB = os.path.join(RESEARCH, 'corpus.db')
MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
MIN_CHARS = 40
CHUNK = 600
OVERLAP = 100


def chunk_text(text):
    text = ' '.join(text.split())
    if len(text) <= CHUNK:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK
        if end < len(text):  # break at a word boundary
            sp = text.rfind(' ', start + CHUNK - 80, end)
            if sp > start:
                end = sp
        chunks.append(text[start:end])
        start = end - OVERLAP if end < len(text) else len(text)
    return chunks


def main():
    from sentence_transformers import SentenceTransformer
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(MODEL, device=device)

    rows = []   # (book, page, path, chunk_idx, chunk_text)
    for jsonl in sorted(glob.glob(os.path.join(PAGES_DIR, '*.jsonl'))):
        by_path = {}
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
        for rec in by_path.values():
            if rec.get('error') or rec.get('chars', 0) < MIN_CHARS:
                continue
            for i, ch in enumerate(chunk_text(rec['text'])):
                rows.append((os.path.dirname(rec['path']), rec['file'],
                             rec['path'], i, ch))
    print(f'{len(rows)} chunks to embed on {device}')

    vecs = model.encode([r[4] for r in rows], batch_size=128,
                        normalize_embeddings=True,
                        show_progress_bar=True).astype(np.float32)

    db = sqlite3.connect(DB)
    db.executescript("""
        DROP TABLE IF EXISTS chunks;
        CREATE TABLE chunks (
            id        INTEGER PRIMARY KEY,
            book      TEXT NOT NULL,
            page      TEXT NOT NULL,
            path      TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            text      TEXT NOT NULL,
            embedding BLOB NOT NULL      -- float32, L2-normalized
        );
        DROP TABLE IF EXISTS chunks_meta;
        CREATE TABLE chunks_meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    for r, v in zip(rows, vecs):
        db.execute('INSERT INTO chunks(book, page, path, chunk_idx, text, embedding) '
                   'VALUES (?,?,?,?,?,?)', (*r, v.tobytes()))
    db.execute("INSERT INTO chunks_meta VALUES ('model', ?)", (MODEL,))
    db.execute("INSERT INTO chunks_meta VALUES ('dim', ?)", (str(vecs.shape[1]),))
    db.commit()
    db.close()
    print(f'stored {len(rows)} chunk vectors ({vecs.shape[1]}-dim) in corpus.db')


if __name__ == '__main__':
    main()
