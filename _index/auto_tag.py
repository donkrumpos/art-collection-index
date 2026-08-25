#!/usr/bin/env python3
"""
auto_tag.py — Layer 2: free, on-device auto-tagging via CLIP zero-shot.

Scores every image's existing embedding against the concept vocabulary in
vocab.py and writes the matches into a `facets` table. No per-image AI cost —
it's matrix math over embeddings we already have. Runs in a couple of minutes.

Also (re)populates the 'folder' facet from each image's top-level directory, so
the gallery has one clean faceted store to read.

Usage:
    python auto_tag.py            # (re)compute all auto facets
    python auto_tag.py --stats    # show the tag vocabulary + counts

Idempotent: rerun any time (e.g. after editing vocab.py).
"""

import argparse
import os
import sqlite3

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "index.db")

import sys
sys.path.insert(0, HERE)
import vocab as V


def ensure_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS facets (
            path  TEXT NOT NULL,
            facet TEXT NOT NULL,
            tag   TEXT NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (path, facet, tag)
        )
        """
    )
    # 'source' distinguishes these zero-shot CLIP tags ('clip') from the richer
    # LLM dossier tags ('llm', written by describe.py). Reruns here must not wipe
    # the LLM tags — see the source-scoped DELETEs below.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(facets)")]
    if "source" not in cols:
        conn.execute("ALTER TABLE facets ADD COLUMN source TEXT DEFAULT 'clip'")
        conn.execute("UPDATE facets SET source='clip' WHERE source IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facets_ft ON facets(facet, tag)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facets_path ON facets(path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facets_source ON facets(source)")
    conn.commit()


def load_embeddings(conn):
    paths, vecs = [], []
    for p, e in conn.execute("SELECT path, embedding FROM images ORDER BY path"):
        paths.append(p)
        vecs.append(np.frombuffer(e, dtype=np.float32))
    return paths, np.vstack(vecs)


def encode_prompts(model_id, prompts):
    import torch
    import open_clip

    name, pretrained = model_id.split("/", 1)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer(name)
    with torch.no_grad():
        t = tok(prompts).to(device)
        f = model.encode_text(t)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy().astype(np.float32)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=60000")  # wait out a concurrent writer (e.g. describe.py sweep)
    ensure_schema(conn)

    if args.stats:
        print("Facet vocabulary in the index:\n")
        for facet, tag, c in conn.execute(
            "SELECT facet, tag, COUNT(*) c FROM facets GROUP BY facet, tag "
            "ORDER BY facet, c DESC"
        ):
            print(f"  {facet:9} {tag:22} {c:>6}")
        return

    model_id = conn.execute(
        "SELECT value FROM meta WHERE key='model'"
    ).fetchone()[0]

    paths, MAT = load_embeddings(conn)
    n = len(paths)
    print(f"Auto-tagging {n} images against vocab.py...")

    # --- folder facet (rebuild from paths) ---  source-scoped so LLM tags survive
    conn.execute("DELETE FROM facets WHERE facet='folder' AND source='clip'")
    conn.executemany(
        "INSERT OR REPLACE INTO facets(path, facet, tag, score, source) VALUES(?,?,?,1.0,'clip')",
        [(p, "folder", p.split(os.sep)[0]) for p in paths],
    )

    # OCR corrections (apply_facet_corrections.py, source='ocr') outrank
    # zero-shot CLIP on `kind` — OCR read the actual page. Don't re-insert
    # CLIP's opinion for those paths.
    ocr_kind = {p for (p,) in conn.execute(
        "SELECT path FROM facets WHERE facet='kind' AND source='ocr'")}

    # --- CLIP zero-shot facets ---
    for facet, mapping in V.FACETS.items():
        labels = list(mapping.keys())
        prompts = [mapping[l] for l in labels]
        print(f"  {facet}: scoring {len(labels)} concepts...")
        tvecs = encode_prompts(model_id, prompts)     # (T, D)
        scores = MAT @ tvecs.T                         # (N, T) raw cosine

        conn.execute("DELETE FROM facets WHERE facet=? AND source='clip'", (facet,))
        rows = []
        if facet == "kind":
            # 'artwork' wins unless a document-kind beats it by KIND_MARGIN.
            art_j = labels.index("artwork")
            doc_js = [k for k, lab in enumerate(labels) if lab in V.DOCUMENT_KINDS]
            art = scores[:, art_j]
            for i in range(n):
                if paths[i] in ocr_kind:
                    continue
                best_doc = max(doc_js, key=lambda k: scores[i, k])
                if scores[i, best_doc] > art[i] + V.KIND_MARGIN:
                    rows.append((paths[i], facet, labels[best_doc],
                                 float(scores[i, best_doc])))
                else:
                    rows.append((paths[i], facet, "artwork", float(art[i])))
        elif facet in V.SINGLE_LABEL:
            best = scores.argmax(axis=1)
            for i in range(n):
                j = best[i]
                rows.append((paths[i], facet, labels[j], float(scores[i, j])))
        else:
            order = np.argsort(-scores, axis=1)[:, : V.MULTI_MAX]
            for i in range(n):
                for rank_pos, j in enumerate(order[i]):
                    s = float(scores[i, j])
                    # always keep the single best; keep the rest above the floor
                    if rank_pos == 0 or s >= V.MULTI_FLOOR:
                        rows.append((paths[i], facet, labels[j], s))
        conn.executemany(
            "INSERT OR REPLACE INTO facets(path, facet, tag, score, source) "
            "VALUES(?,?,?,?,'clip')",
            rows,
        )
        conn.commit()
        print(f"    -> {len(rows)} tags assigned")

    total = conn.execute("SELECT COUNT(*) FROM facets").fetchone()[0]
    print(f"\nDone. {total} facet tags in {DB_PATH}")
    print("Run  python auto_tag.py --stats  to see the vocabulary counts.")


if __name__ == "__main__":
    main()
