#!/usr/bin/env python3
"""
blank_scan.py — flag near-blank / empty scan pages.

Many book scans include empty white verso pages. They aren't artwork and they
pollute both browse and search (a flat white image weakly matches everything).
This pass reads each image's pixels, and flags it as kind='blank' when it's
overwhelmingly one flat light tone.

Writes:
  - facet rows ('kind','blank') into index.db  (the gallery hides these)
  - blank-pages.txt : a plain list of the blank files, for review/deletion

Usage:
    python blank_scan.py
    python blank_scan.py --stats

Tune BLANK_MEAN / BLANK_STD below if it's too aggressive or too lax.
"""

import argparse
import os
import sqlite3
import time

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "index.db")
LIST_PATH = os.path.join(HERE, "blank-pages.txt")

# A page is "blank" when it's very light AND very uniform (little variation).
BLANK_MEAN = 232   # mean gray (0-255); higher = lighter
BLANK_STD = 14     # std of gray; lower = flatter/more uniform


def is_blank(full):
    with Image.open(full) as im:
        g = im.convert("L").resize((48, 48))
        a = np.asarray(g, dtype=np.float32)
    return a.mean() >= BLANK_MEAN and a.std() <= BLANK_STD, a.mean(), a.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    root = os.path.dirname(HERE)  # portable: derive from script location, not stored absolute path

    if args.stats:
        n = conn.execute(
            "SELECT COUNT(*) FROM facets WHERE facet='kind' AND tag='blank'"
        ).fetchone()[0]
        print(f"{n} pages flagged blank. List: {LIST_PATH}")
        return

    paths = [r[0] for r in conn.execute("SELECT path FROM images ORDER BY path")]
    print(f"Scanning {len(paths)} images for blank pages...")

    conn.execute("DELETE FROM facets WHERE facet='kind' AND tag='blank'")
    blanks, rows = [], []
    t0 = time.time()
    for i, rel in enumerate(paths):
        try:
            blank, _, _ = is_blank(os.path.join(root, rel))
        except Exception:
            continue
        if blank:
            blanks.append(rel)
            rows.append((rel, "kind", "blank", 1.0))
        if i % 2000 == 0 and i:
            print(f"\r  {i}/{len(paths)}  ({len(blanks)} blank)", end="", flush=True)

    conn.executemany(
        "INSERT OR REPLACE INTO facets(path, facet, tag, score) VALUES(?,?,?,?)", rows
    )
    conn.commit()
    with open(LIST_PATH, "w") as f:
        f.write("\n".join(blanks) + ("\n" if blanks else ""))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"{len(blanks)} blank pages flagged.\nList written: {LIST_PATH}")


if __name__ == "__main__":
    main()
