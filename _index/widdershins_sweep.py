#!/usr/bin/env python3
"""
widdershins_sweep.py — curate a cross-cutting "widdershins" collection.

Widdershins material is scattered across dozens of subject folders (esoterica,
krampus, herbals in rare-books, cryptids, memento mori, shadow puppets ...).
No single folder holds it. This runs a CLIP semantic sweep over a set of
Widdershins aesthetic AXES, banks the hits as a `widdershins` facet (so they
show up as filter chips in serve.py), and writes a review contact sheet grouped
by axis so the false positives can be pruned before finalizing.

Two-stage tag lifecycle:
  * source='widdershins-cand'  — this sweep's raw candidates (reviewable, prunable)
  * source='widdershins'       — promoted survivors (finalize step, later)

Usage:
    python widdershins_sweep.py                 # wide default sweep + review html
    python widdershins_sweep.py --per-axis 300 --floor 0.20
    python widdershins_sweep.py --clear         # wipe candidate tags and stop
"""
import argparse
import html as htmlmod
import os
import sqlite3

import numpy as np

from search import load_index, encode_text

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "index.db")

# --- The Widdershins axes. Each: slug -> (display, [natural CLIP prompts], use) ---
AXES = {
    "green-man": (
        "Green Man / foliate faces",
        ["a green man foliate face carving made of leaves",
         "a face formed from leaves and vines, forest spirit",
         "a foliate head sprouting branches and foliage"],
        "Tarot (XIX Green Man), murals, widdershins.ink",
    ),
    "cryptid-beast": (
        "Cryptids / uncanny beasts",
        ["an illustration of a strange cryptid creature",
         "a folkloric monster or uncanny beast",
         "a mysterious animal of legend, cryptozoology plate"],
        "Loodoo, beast throughline, shadow-puppets",
    ),
    "hedge-witch": (
        "Hedge witchery / folk magic",
        ["folk magic ritual objects and charms",
         "a witch's cottage herbs candles and talismans",
         "hedge witchcraft tools and folk remedies"],
        "Hedge Witch Almanac, sabbat dispatches",
    ),
    "esoteric-diagram": (
        "Esoteric / alchemical diagrams",
        ["an antique alchemical diagram with symbols",
         "an occult esoteric engraving with sigils and geometry",
         "a hermetic emblem, kabbalah or astrological chart"],
        "Cabinet of Curiosities, marquee plates, tarot",
    ),
    "herbal-botanical": (
        "Herbals / botanical / fungi",
        ["an antique botanical engraving of a plant",
         "a herbal illustration of medicinal plants",
         "a vintage scientific plate of mushrooms and fungi"],
        "HWA, seed/serotiny, dispatches",
    ),
    "krampus-folk-horror": (
        "Krampus / folk-horror masks",
        ["a krampus devil folk horror figure",
         "a carved folk horror mask or costumed seasonal demon",
         "a pagan winter monster with horns and chains"],
        "Krampusnacht immersive, Yonder",
    ),
    "wheel-sabbat": (
        "Wheel of the Year / celestial",
        ["a wheel of the year seasonal calendar",
         "an antique celestial chart of sun moon and stars",
         "a pagan sabbat seasonal festival illustration"],
        "Widdershins Wheel, sabbat cards",
    ),
    "memento-mori": (
        "Memento mori / bone",
        ["a memento mori skull and bones illustration",
         "a danse macabre skeleton engraving",
         "vanitas symbols of death and mortality"],
        "Spirit of Place, Death's Door, shadow-puppet",
    ),
    "mythic-forest": (
        "Mythic forest / fairy-tale",
        ["a dark enchanted forest with a lone figure",
         "a fairy tale illustration of a mysterious woodland",
         "a liminal misty forest, folklore scene"],
        "Stormroot, Terra Incognita, tarot",
    ),
    "woodcut-texture": (
        "Woodcut / engraving texture",
        ["a bold black and white woodcut print",
         "a hatched antique engraving with strong line texture",
         "a linocut relief print, high contrast folk style"],
        "Zine, print projects, marquee device",
    ),
    "grotesque-marginalia": (
        "Grotesques / marginalia",
        ["a medieval marginalia grotesque drollery",
         "a bizarre hybrid creature in a manuscript margin",
         "an ornamental grotesque figure, antique"],
        "Cabinet, zine",
    ),
    "mask-shadow": (
        "Masks / silhouette / shadow",
        ["a shadow puppet silhouette figure",
         "a black silhouette of a figure or creature",
         "a ceremonial mask, stark theatrical form"],
        "Shadow-puppet animation, immersive",
    ),
}


def excluded_paths(conn):
    """Skip scanned text/title pages — not usable reference imagery."""
    rows = conn.execute(
        "SELECT DISTINCT path FROM facets WHERE facet='kind' "
        "AND tag IN ('text page','title page','text with figure')"
    )
    return {r[0] for r in rows}


def axis_vector(model_id, prompts):
    """Average the L2-normalized text embeddings of an axis's prompts."""
    vecs = [encode_text(model_id, p) for p in prompts]
    v = np.mean(np.vstack(vecs), axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-axis", type=int, default=300,
                    help="max candidates kept per axis (wide default)")
    ap.add_argument("--floor", type=float, default=0.20,
                    help="minimum cosine similarity to keep")
    ap.add_argument("--html", default=os.path.join(HERE, "widdershins-review.html"))
    ap.add_argument("--clear", action="store_true",
                    help="delete candidate tags and exit")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=60000")

    if args.clear:
        n = conn.execute("DELETE FROM facets WHERE source='widdershins-cand'").rowcount
        conn.commit()
        print(f"Cleared {n} candidate tags.")
        return

    model_id, root, paths, mat = load_index()
    skip = excluded_paths(conn)
    print(f"{len(paths)} images · {len(skip)} text-pages excluded · "
          f"{len(AXES)} axes · per-axis={args.per_axis} floor={args.floor}\n")

    # fresh candidate set each run
    conn.execute("DELETE FROM facets WHERE source='widdershins-cand'")

    path_idx = {p: i for i, p in enumerate(paths)}
    keep_mask = np.array([p not in skip for p in paths])

    per_image = {}          # path -> {axis: score}
    grouped = {}            # axis -> [(score, path)] for the review sheet
    rows = []
    for slug, (name, prompts, _use) in AXES.items():
        qv = axis_vector(model_id, prompts)
        scores = mat @ qv
        scores = np.where(keep_mask, scores, -1.0)
        order = np.argsort(-scores)[: args.per_axis]
        kept = [(float(scores[o]), paths[o]) for o in order if scores[o] >= args.floor]
        grouped[slug] = kept
        for sc, p in kept:
            per_image.setdefault(p, {})[slug] = sc
            rows.append((p, "widdershins", slug, sc, "widdershins-cand"))
        print(f"  {slug:22s} {len(kept):4d}  (top {kept[0][0]:.3f})" if kept
              else f"  {slug:22s}    0")

    conn.executemany(
        "INSERT OR REPLACE INTO facets(path,facet,tag,score,source) VALUES(?,?,?,?,?)",
        rows,
    )
    # a rollup chip: everything that matched ANY axis
    conn.executemany(
        "INSERT OR REPLACE INTO facets(path,facet,tag,score,source) VALUES(?,?,?,?,?)",
        [(p, "collection", "widdershins", max(d.values()), "widdershins-cand")
         for p, d in per_image.items()],
    )
    conn.commit()

    print(f"\nBanked {len(rows)} axis-tags across {len(per_image)} unique images "
          f"(facet 'widdershins', source 'widdershins-cand').")

    # --- review contact sheet, grouped by axis, score-sorted ---
    sections = []
    for slug, (name, _prompts, use) in AXES.items():
        cells = []
        for sc, rel in grouped[slug]:
            uri = "file://" + os.path.join(root, rel).replace(" ", "%20")
            cells.append(
                f'<figure><a href="{uri}" target="_blank"><img loading=lazy src="{uri}">'
                f'</a><figcaption>{sc:.3f} — {htmlmod.escape(rel)}</figcaption></figure>'
            )
        sections.append(
            f'<h2>{htmlmod.escape(name)} <small>({len(grouped[slug])}) — '
            f'{htmlmod.escape(use)}</small></h2><div class=grid>' + "".join(cells) + "</div>"
        )
    doc = (
        "<!doctype html><meta charset=utf-8><title>Widdershins review</title>"
        "<style>body{background:#111;color:#ddd;font:13px/1.4 system-ui;margin:16px}"
        "h1{font-size:16px}h2{font-size:14px;margin-top:28px;border-top:1px solid #333;padding-top:10px}"
        "small{color:#8a8;font-weight:400}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;height:200px;object-fit:contain;background:#000;border-radius:6px}"
        "figcaption{font-size:10px;color:#9aa;margin-top:3px;word-break:break-all}a{color:inherit}</style>"
        f"<h1>Widdershins candidates — {len(per_image)} images across {len(AXES)} axes</h1>"
        + "".join(sections)
    )
    with open(args.html, "w") as f:
        f.write(doc)
    print(f"Review sheet: {args.html}")


if __name__ == "__main__":
    main()
