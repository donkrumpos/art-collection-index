#!/usr/bin/env python3
"""
search.py — query the Art Collection CLIP index.

Semantic text search and visual "more like this" over the local embedding index
built by embed.py. On-device, no cloud.

Examples:
    python search.py "misty forest, lone figure"
    python search.py "art nouveau peacock border" -n 30
    python search.py --image "krampus/some-devil.jpg"       # similar images
    python search.py --image ~/Desktop/reference.png --open  # open top hits
    python search.py "sepia botanical engraving" --in "antiquarian prints"
    python search.py "steampunk airship" --html results.html # visual contact sheet

Flags:
    -n N        number of results (default 20)
    --image P   query by image (path absolute, or relative to collection root)
    --in PREFIX only match files whose path starts with PREFIX (folder filter)
    --open      open the top results with macOS `open` (Preview/Finder)
    --open-n K  how many to open with --open (default 5)
    --html FILE write an HTML contact sheet of results and open it
"""

import argparse
import os
import sqlite3
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "index.db")


def load_index():
    if not os.path.exists(DB_PATH):
        sys.exit(f"No index found at {DB_PATH}. Run embed.py first.")
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT value FROM meta WHERE key='model'").fetchone()
    model = row[0] if row else None
    root_row = conn.execute("SELECT value FROM meta WHERE key='root'").fetchone()
    root = root_row[0] if root_row else os.path.dirname(HERE)

    paths, vecs = [], []
    for path, emb in conn.execute("SELECT path, embedding FROM images"):
        paths.append(path)
        vecs.append(np.frombuffer(emb, dtype=np.float32))
    conn.close()
    if not paths:
        sys.exit("Index is empty. Run embed.py first.")
    mat = np.vstack(vecs)  # already L2-normalized at embed time
    return model, root, paths, mat


def get_model(model_id):
    """model_id looks like 'ViT-B-32/laion2b_s34b_b79k'."""
    import torch
    import open_clip

    name, pretrained = model_id.split("/", 1)
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(name)
    return model, preprocess, tokenizer, device


def encode_text(model_id, text):
    import torch

    model, _, tokenizer, device = get_model(model_id)
    with torch.no_grad():
        toks = tokenizer([text]).to(device)
        feat = model.encode_text(toks)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)[0]


def encode_image(model_id, img_path):
    import torch
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception:
        pass

    model, preprocess, _, device = get_model(model_id)
    with Image.open(img_path) as im:
        t = preprocess(im.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)[0]


def write_html(out_path, root, results):
    import html as htmlmod

    cells = []
    for score, rel in results:
        full = os.path.join(root, rel)
        uri = "file://" + full.replace(" ", "%20")
        label = htmlmod.escape(rel)
        cells.append(
            f'<figure><a href="{uri}" target="_blank">'
            f'<img loading="lazy" src="{uri}"></a>'
            f'<figcaption>{score:.3f} — {label}</figcaption></figure>'
        )
    doc = (
        "<!doctype html><meta charset=utf-8><title>Art Collection search</title>"
        "<style>body{background:#111;color:#ddd;font:13px/1.4 system-ui;margin:16px}"
        "h1{font-size:15px;font-weight:600}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}"
        "figure{margin:0}img{width:100%;height:220px;object-fit:contain;background:#000;border-radius:6px}"
        "figcaption{font-size:11px;color:#9aa;margin-top:4px;word-break:break-all}"
        "a{color:inherit}</style>"
        f"<h1>{len(results)} results</h1><div class=grid>" + "".join(cells) + "</div>"
    )
    with open(out_path, "w") as f:
        f.write(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="text query")
    ap.add_argument("--image", help="query by image path (abs or relative to root)")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--in", dest="prefix", help="only paths starting with PREFIX")
    ap.add_argument("--open", action="store_true", help="open top hits in macOS")
    ap.add_argument("--open-n", type=int, default=5)
    ap.add_argument("--html", help="write an HTML contact sheet and open it")
    args = ap.parse_args()

    if not args.query and not args.image:
        ap.error("give a text query or --image")

    model_id, root, paths, mat = load_index()
    if not model_id:
        sys.exit("Index has no model metadata; rebuild with embed.py.")

    if args.image:
        img = args.image
        if not os.path.isabs(img) and not os.path.exists(img):
            img = os.path.join(root, args.image)
        img = os.path.expanduser(img)
        if not os.path.exists(img):
            sys.exit(f"Image not found: {img}")
        qvec = encode_image(model_id, img)
        header = f'Similar to image: {args.image}'
    else:
        qvec = encode_text(model_id, args.query)
        header = f'Query: "{args.query}"'

    scores = mat @ qvec  # cosine similarity (all vectors normalized)

    # Optional folder filter.
    idx = np.arange(len(paths))
    if args.prefix:
        pref = args.prefix.rstrip("/")
        mask = np.array([p == pref or p.startswith(pref + os.sep) for p in paths])
        idx = idx[mask]
        scores_f = scores[mask]
        header += f'  [in: {args.prefix}]'
    else:
        scores_f = scores

    if len(idx) == 0:
        sys.exit("No files matched that --in filter.")

    k = min(args.n, len(idx))
    order = np.argsort(-scores_f)[:k]
    results = [(float(scores_f[o]), paths[idx[o]]) for o in order]

    print(header)
    print(f"{len(paths)} images in index · top {k}\n")
    for rank, (score, rel) in enumerate(results, 1):
        print(f"{rank:>3}. {score:.3f}  {rel}")

    if args.open:
        for _, rel in results[: args.open_n]:
            subprocess.run(["open", os.path.join(root, rel)], check=False)

    if args.html:
        out = os.path.abspath(args.html)
        write_html(out, root, results)
        print(f"\nWrote contact sheet: {out}")
        subprocess.run(["open", out], check=False)


if __name__ == "__main__":
    main()
