#!/usr/bin/env python3
"""
embed.py — Phase 1 of the Art Collection index.

Walk the collection, compute a local CLIP embedding for every image, and store
it in a SQLite index. Fully on-device (Apple Silicon / MPS), no cloud, no
per-image API cost. Incremental + resumable: rerun any time and it only embeds
files that are new or changed.

Usage:
    python embed.py                # embed everything under the collection root
    python embed.py --root PATH    # override collection root
    python embed.py --limit 500    # only embed the first N pending (for testing)
    python embed.py --stats        # print index stats and exit

The index lives at _index/index.db next to this script.
"""

import argparse
import os
import sqlite3
import sys
import time

import numpy as np
import torch
import open_clip
from PIL import Image, ImageFile

# Some collection files are truncated/corrupt; let PIL load what it can.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional HEIC support (7 files in the collection).
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass

# ---- Config -----------------------------------------------------------------

# Default collection root = parent of this _index/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)
DB_PATH = os.path.join(HERE, "index.db")

# CLIP model. ViT-B-32 is fast and 512-dim — a good default for ~40k images.
# For higher search quality (slower, 768-dim) swap to:
#   MODEL_NAME, PRETRAINED = "ViT-L-14", "laion2b_s32b_b82k"
# then delete index.db (or it'll refuse to mix models) and rerun.
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".tif", ".tiff", ".bmp", ".heic",
}

# Directories to skip entirely (relative path prefixes from root).
SKIP_DIRS = {"_index"}

BATCH_SIZE = 32
COMMIT_EVERY = 20  # commit to sqlite every N batches (resumability granularity)


# ---- Device -----------------------------------------------------------------

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---- DB ---------------------------------------------------------------------

def open_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            path      TEXT PRIMARY KEY,   -- relative to collection root
            mtime     REAL NOT NULL,
            size      INTEGER NOT NULL,
            model     TEXT NOT NULL,
            dim       INTEGER NOT NULL,
            embedding BLOB NOT NULL        -- float32, L2-normalized
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY, value TEXT
        )
        """
    )
    conn.commit()
    return conn


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn, key, value):
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


# ---- Walk / diff ------------------------------------------------------------

def iter_images(root):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        top = rel_dir.split(os.sep)[0]
        if top in SKIP_DIRS:
            dirnames[:] = []
            continue
        for name in filenames:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            yield rel, full


def pending_files(conn, root):
    """Return list of (rel, full) that are new or changed since last embed."""
    have = {}
    for path, mtime, size in conn.execute("SELECT path, mtime, size FROM images"):
        have[path] = (mtime, size)
    out = []
    for rel, full in iter_images(root):
        try:
            st = os.stat(full)
        except OSError:
            continue
        prev = have.get(rel)
        if prev is None or abs(prev[0] - st.st_mtime) > 1e-6 or prev[1] != st.st_size:
            out.append((rel, full, st.st_mtime, st.st_size))
    return out


# ---- Embedding --------------------------------------------------------------

def load_image(full, preprocess):
    with Image.open(full) as im:
        im = im.convert("RGB")
        return preprocess(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="max pending files this run")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    conn = open_db()

    if args.stats:
        n = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        model = get_meta(conn, "model", "(none)")
        print(f"index.db: {n} images embedded, model={model}")
        print(f"root: {get_meta(conn, 'root', '(unset)')}")
        return

    # Model consistency guard.
    stored_model = get_meta(conn, "model")
    this_model = f"{MODEL_NAME}/{PRETRAINED}"
    if stored_model and stored_model != this_model:
        print(
            f"ERROR: index.db was built with '{stored_model}' but this script is "
            f"configured for '{this_model}'.\nDelete {DB_PATH} to re-embed, or "
            f"revert MODEL_NAME/PRETRAINED.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Scanning {root} for new/changed images...")
    pending = pending_files(conn, root)
    total_indexed = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    print(f"{total_indexed} already embedded; {len(pending)} pending.")
    if args.limit and len(pending) > args.limit:
        pending = pending[: args.limit]
        print(f"--limit: capping this run to {len(pending)}.")
    if not pending:
        print("Nothing to do. Index is up to date.")
        return

    device = pick_device()
    print(f"Loading {this_model} on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()
    dim = model.visual.output_dim

    set_meta(conn, "model", this_model)
    set_meta(conn, "dim", dim)
    set_meta(conn, "root", root)
    conn.commit()

    t0 = time.time()
    done = 0
    failed = 0
    batch_count = 0
    i = 0
    N = len(pending)

    while i < N:
        # Build a batch, skipping unreadable files.
        tensors, rows = [], []
        while len(tensors) < BATCH_SIZE and i < N:
            rel, full, mtime, size = pending[i]
            i += 1
            try:
                tensors.append(load_image(full, preprocess))
                rows.append((rel, mtime, size))
            except Exception as e:
                failed += 1
                if failed <= 20:
                    print(f"  skip (unreadable): {rel} — {e}")
        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu().numpy().astype(np.float32)

        for (rel, mtime, size), vec in zip(rows, feats):
            conn.execute(
                "INSERT INTO images(path, mtime, size, model, dim, embedding) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                "mtime=excluded.mtime, size=excluded.size, model=excluded.model, "
                "dim=excluded.dim, embedding=excluded.embedding",
                (rel, mtime, size, this_model, dim, vec.tobytes()),
            )
        done += len(rows)
        batch_count += 1
        if batch_count % COMMIT_EVERY == 0:
            conn.commit()

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0
        eta = (N - i) / rate if rate else 0
        print(
            f"\r  {done}/{N} embedded  ({rate:.1f} img/s, ETA {eta/60:.1f} min, "
            f"{failed} failed)",
            end="",
            flush=True,
        )

    conn.commit()
    print()
    print(f"Done. Embedded {done} images in {(time.time()-t0)/60:.1f} min "
          f"({failed} unreadable/skipped).")
    grand = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    print(f"Index now holds {grand} images at {DB_PATH}")


if __name__ == "__main__":
    main()
