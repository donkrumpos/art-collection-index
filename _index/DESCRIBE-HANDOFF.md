# Handoff: Layer 3 — per-image AI dossiers (description + grounded ID + SD prompt)

Paste the block below into a fresh Claude Code session started in
`~/Documents/Art Collection`. Builds the "tell me everything about this image"
layer over the existing CLIP catalog.

## ONE DECISION TO MAKE FIRST (Foggy)

The vision model can run two ways — decide before building:

- **Cloud (Claude vision API):** best quality descriptions + SD prompts, ~1¢-ish
  per image, but images are sent to the API (not on-device) and it needs an
  ANTHROPIC_API_KEY with billing. Batch by folder to control cost.
- **Local VLM (on-device, owned):** e.g. Qwen2-VL or LLaVA via Ollama/MLX —
  free, nothing leaves the Mac (matches the collection's on-device ethos), but
  lower description quality and more setup. Slower.

Recommendation: **local VLM for a first pass over everything** (free, owned,
good enough for search/tags), then **cloud Claude vision on hero folders** where
you want gallery-grade dossiers + the best SD prompts. Pick per your values.

---

```
Build Layer 3 of the Art Collection index: a per-image AI dossier pipeline.
For each image produce a rich description, structured tags, a GROUNDED
identification, and a Stable Diffusion prompt — stored in index.db and surfaced
in the gallery. Read PLAN.md and _index/README.md first.

WHAT ALREADY EXISTS:
- ~/Documents/Art Collection/_index/index.db — SQLite. `images` (path +
  CLIP embedding), `facets` (medium/style/subject/tone/folder/kind tags),
  meta (model id, root). 39,199 images embedded (ViT-B-32/laion2b).
- _index/serve.py — the live gallery (browse + faceted filter + search +
  similar + reveal-in-Finder). Reads facets. You'll extend it.
- _index/.venv — Python 3.12 venv (torch, open_clip, PIL, numpy).
- _index/search.py — CLIP query; --image finds nearest neighbors in the repo.

BUILD  _index/describe.py:
1. Model backend (per Foggy's decision above): either the Anthropic API via the
   `anthropic` SDK (read the claude-api skill for the current vision model id +
   pricing — do NOT hardcode from memory), or a local VLM (Ollama/MLX). Make the
   backend a single swappable function describe_image(path, grounding) -> dict.
2. GROUNDED IDENTIFICATION (the honest-ID mechanism — important):
   For each image, use the existing CLIP index to find its top nearest neighbors.
   If a neighbor has a DESCRIPTIVE (non-hash) filename/folder, pass those as
   grounding hints to the vision model, e.g. "visually similar to:
   fantasy art/alan lee/Minas Ithil.jpg (0.86)". Instruct the model to assert an
   identification ONLY when the visual match clearly supports it; otherwise say
   "reads like X" with low confidence. NEVER let it invent artist/date/provenance
   from memory. Return id_confidence in {high, medium, low, none} + the basis.
   (This is how the prior session correctly ID'd an Alan Lee "Minas Ithil" card:
   by matching a labeled copy in the repo, not by guessing.)
3. Force STRUCTURED output (tool use / JSON):
   { caption, description, subjects[], medium, style[], palette[], mood,
     sd_prompt, sd_negative, identification, id_confidence, id_basis }
4. STORE:
   - New table descriptions(path PRIMARY KEY, caption, description, sd_prompt,
     sd_negative, dossier_json, identification, id_confidence, model, ts).
   - Also explode subjects/style/medium into the `facets` table with
     source='llm' so they power the existing faceted filters (higher quality
     than the zero-shot tags).
5. Batching + safety: --folder arg to do one folder at a time; resumable (skip
   paths already in `descriptions`); --limit for test runs; commit incrementally.
   Priority folders first: fantasy art, krampus, esoterica metaphysical, murals.
   Print a running cost estimate if using the API.

THEN extend _index/serve.py:
- /api/describe?path= returns the stored dossier.
- Each tile gets an "ⓘ info" button opening a panel with the description,
  identification (+confidence badge), tags, and a copy-able SD prompt.
- LLM facets (source='llm') appear in the filter chips alongside the auto tags.

CONSTRAINTS:
- Resumable and idempotent. Don't touch original image files.
- Do NOT fabricate provenance — grounded ID or explicit uncertainty only.
- Test on --folder krampus --limit 10 and SHOW Foggy the dossiers before
  running a whole folder. Get a thumbs-up on quality + cost per image first.

DELIVERABLE: _index/describe.py + serve.py info panel, demoed on ~10 images,
with a per-image cost figure and a short note on description quality.
```

---

Where this fits: PLAN.md Phase 3. It sharpens search, powers better filter
chips, gives every image an SD prompt, and — via grounded ID — tells you what a
piece actually is without hallucinating. The dossiers live in the same index.db.
