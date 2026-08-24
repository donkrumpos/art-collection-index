# Art Collection — CLIP Index (Phase 1)

Local, on-device semantic + visual search over the whole reference library.
No cloud, no per-image API cost. Built per `../PLAN.md` step 1.

## What it is

- **`embed.py`** — walks the collection, computes a CLIP embedding for every image,
  stores it in `index.db` (SQLite). Incremental + resumable: rerun any time and it
  only processes new/changed files.
- **`search.py`** — query the index by text ("misty forest, lone figure") or by
  image ("more like this"). Ranks the whole library by cosine similarity.
- **`index.db`** — the index. ~40k rows, one L2-normalized float32 embedding each.
- **`.venv/`** — dedicated Python 3.12 environment (torch + open_clip). Not committed.

Model: **ViT-B-32 / laion2b_s34b_b79k** (512-dim), runs on Apple Silicon MPS.

## Setup (already done once)

```bash
python3.12 -m venv _index/.venv
_index/.venv/bin/pip install torch torchvision open_clip_torch pillow pillow-heif numpy tqdm
```

## Build / update the index

Run from the collection root (`~/Documents/Art Collection`):

```bash
_index/.venv/bin/python _index/embed.py            # embed everything new/changed
_index/.venv/bin/python _index/embed.py --stats     # how many are indexed
```

~21 img/s on an M4 Pro → full ~40k library in ~30 min. Safe to Ctrl-C and rerun;
it resumes. Add images later? Just rerun — only the new ones get embedded.

## Search

```bash
# semantic text search
_index/.venv/bin/python _index/search.py "art nouveau peacock border" -n 30

# visual "more like this" (path relative to collection, or any absolute path)
_index/.venv/bin/python _index/search.py --image "krampus/some-devil.jpg"
_index/.venv/bin/python _index/search.py --image ~/Desktop/reference.png

# folder filter
_index/.venv/bin/python _index/search.py "sepia botanical" --in "antiquarian prints"

# open the top 5 hits in Preview/Finder
_index/.venv/bin/python _index/search.py "steampunk airship" --open

# visual contact sheet in the browser (best way to actually SEE results)
_index/.venv/bin/python _index/search.py "misty forest lone figure" --html hits.html
```

Tip: add a shell alias so you don't type the venv path every time:

```bash
alias artsearch='~/Documents/"Art Collection"/_index/.venv/bin/python ~/Documents/"Art Collection"/_index/search.py'
# then:  artsearch "wilder mann straw costume" --html hits.html
```

## Live visual gallery (the good part)

Terminal results are just paths. To actually *browse*, run the local web app:

```bash
artserve                      # then open http://localhost:8756
```

Type a query, see the art in a grid, click **◆ similar** on any tile for
more-like-this, filter by folder, click an image to open it full-size. It's
fully on-device — the CLIP model loads on your first search, thumbnails are
generated on the fly and cached in `_index/.thumbs/`, and nothing ever leaves
the Mac. This is the pre-Astro search surface; the eventual Astro gallery
(PLAN Phase 5) can read the same `index.db`.

## Folder → seed tags (Phase 2, done)

```bash
arttags            # bank each image's current folder as tags (idempotent)
arttags --stats    # show the folder vocabulary
```

Stored in the `tags` table (`source='folder'`). This preserves today's sort
*before* any flatten/rename throws it away (PLAN Rule 2), and every later phase
(rename, LLM tags, gallery filters) reads it.

## Upgrading search quality later

ViT-B-32 is the fast default. For sharper semantic matching, edit the two lines at
the top of `embed.py`:

```python
MODEL_NAME, PRETRAINED = "ViT-L-14", "laion2b_s32b_b82k"   # 768-dim, ~3x slower
```

then `rm _index/index.db` and rerun `embed.py` (the script refuses to mix models in
one DB). Everything downstream (`search.py`) adapts automatically — it reads the
model id from the DB.

## Where this sits in the plan

Phase 1 of `../PLAN.md`. Next phases (not built yet): folder-name → seed tags,
LLM rich captions, descriptive renaming, an Astro search gallery. Those layer on
top of this same `index.db`.
