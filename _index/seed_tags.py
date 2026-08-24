#!/usr/bin/env python3
"""
seed_tags.py — Phase 2 of the Art Collection index.

Bank each image's CURRENT folder location into the index as its first tags,
BEFORE any flatten/reorg can throw that information away (PLAN.md Rule 2).

Every path component becomes a 'folder'-source tag. Example:
    fantasy art/alan lee/90 Artworks by Alan Lee/89.jpg
  ->  tags: "fantasy art", "alan lee", "90 artworks by alan lee"

This is free, needs no decisions, and every later phase (rename, gallery,
LLM tags) reads these. Idempotent: rerun any time.

Usage:
    python seed_tags.py            # populate folder tags for all indexed images
    python seed_tags.py --stats    # show tag counts
"""

import argparse
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "index.db")


def ensure_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tags (
            path   TEXT NOT NULL,
            tag    TEXT NOT NULL,
            source TEXT NOT NULL,          -- 'folder' now; 'llm' etc. later
            PRIMARY KEY (path, tag, source)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_path ON tags(path)")
    conn.commit()


def norm(tag):
    return tag.strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(DB_PATH):
        raise SystemExit(f"No index at {DB_PATH}. Run embed.py first.")
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    if args.stats:
        n = conn.execute(
            "SELECT COUNT(*) FROM tags WHERE source='folder'"
        ).fetchone()[0]
        print(f"{n} folder tags across "
              f"{conn.execute('SELECT COUNT(DISTINCT path) FROM tags').fetchone()[0]} images")
        print("\nTop 25 folder tags:")
        for tag, c in conn.execute(
            "SELECT tag, COUNT(*) c FROM tags WHERE source='folder' "
            "GROUP BY tag ORDER BY c DESC LIMIT 25"
        ):
            print(f"  {c:>6}  {tag}")
        return

    paths = [r[0] for r in conn.execute("SELECT path FROM images")]
    print(f"Banking folder tags for {len(paths)} images...")

    rows = []
    for p in paths:
        folder = os.path.dirname(p)
        if not folder:
            continue
        for comp in folder.split(os.sep):
            t = norm(comp)
            if t and t not in ("_mine", "_review-unsorted"):
                rows.append((p, t, "folder"))
        # also keep the top-level folder explicitly flagged for provenance
        top = norm(folder.split(os.sep)[0])
        rows.append((p, top, "folder-top"))

    conn.executemany(
        "INSERT OR IGNORE INTO tags(path, tag, source) VALUES(?,?,?)", rows
    )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
    print(f"Done. {n} tag rows banked into {DB_PATH}")
    print("Run with --stats to see the vocabulary.")


if __name__ == "__main__":
    main()
