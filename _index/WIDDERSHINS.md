# The Widdershins collection (cross-cutting tag)

Widdershins-worthy references don't live in one folder — they're scattered across
`esoterica metaphysical`, `krampus`, `shadow puppets`, `engravings`, `fairy`,
herbals inside `rare books`, cryptids, memento mori, botanicals. This tag gathers
them into one browsable collection via a CLIP semantic sweep across 12 aesthetic
axes. Built by `widdershins_sweep.py` (2026-08-25).

## Tag lifecycle
- **`source='widdershins-cand'`** — the raw sweep candidates (what exists now).
  Reviewable + prunable. Re-running the sweep replaces this set.
- **`source='widdershins'`** — the promoted survivors after you prune (finalize step).

In `serve.py` (http://localhost:8756) these appear as a **`widdershins`** chip group
(one sub-chip per axis) plus a **`collection: widdershins`** rollup chip.

## First sweep result (per-axis=300, floor=0.20, "cast wide")
3,600 axis-tags across **2,830 unique images**. Every axis capped at 300, so the
tail of each is noisy by design — prune it. The signal is overlap: **134 images
match 3+ axes, 26 match 4+** — that's the Widdershins core.

| Axis | What it feeds |
|---|---|
| Green Man / foliate faces | Tarot (XIX Green Man done), murals, widdershins.ink |
| Cryptids / uncanny beasts | Loodoo, beast throughline, shadow-puppets |
| Hedge witchery / folk magic | Hedge Witch Almanac, sabbat dispatches |
| Esoteric / alchemical diagrams | Cabinet of Curiosities, marquee plates, tarot |
| Herbals / botanical / fungi | HWA, seed/serotiny, dispatches |
| Krampus / folk-horror masks | Krampusnacht immersive, Yonder |
| Wheel of the Year / celestial | Widdershins Wheel, sabbat cards |
| Memento mori / bone | Spirit of Place, Death's Door, shadow-puppet |
| Mythic forest / fairy-tale | Stormroot, Terra Incognita, tarot |
| Woodcut / engraving texture | Zine, print projects, marquee device |
| Grotesques / marginalia | Cabinet, zine |
| Masks / silhouette / shadow | Shadow-puppet animation, immersive |

## How to review + prune
1. **Grouped sheet:** `_index/widdershins-review.html` (open it) — every candidate
   by axis, score-sorted. Eyeball each axis; note which axes are mostly noise and
   where the good stuff stops (the score where quality falls off).
2. **Live gallery:** http://localhost:8756 — click the `widdershins` axis chips,
   combine with medium/style chips (e.g. widdershins + woodcut).

## Finalize (after you tell me what to cut)
Options I can run once you've reviewed:
- **Promote all** candidates to `source='widdershins'`.
- **Promote by threshold** — keep only score ≥ X per axis (raise the floor).
- **Drop weak axes** entirely (e.g. if `woodcut-texture` is too broad).
- **Keep only the multi-axis core** (3+ or 4+ axes) as a tight starter set.
- Prune individual images by path.

## Re-run / reset
```bash
_index/.venv/bin/python _index/widdershins_sweep.py            # re-sweep (replaces candidates)
_index/.venv/bin/python _index/widdershins_sweep.py --per-axis 500 --floor 0.24
_index/.venv/bin/python _index/widdershins_sweep.py --clear    # wipe candidates
```
Edit the `AXES` dict in `widdershins_sweep.py` to add/adjust axes and their prompts.
