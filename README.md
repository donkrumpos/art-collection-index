# Art Collection Index

On-device semantic search for a 40,000-image art reference library.

I keep a library of about 40,000 art and design reference images and 200+ scanned books, roughly 43GB. At that size, folder names and filenames stop working. I wanted to search it the way I think about images, by subject, mood, medium, and composition, and to find images that look alike, without uploading the library to anyone's cloud.

This repository is the tooling only. The images, the index database, and the generated data stay on my machine.

## What it does

- **Text-to-image and image-to-image search.** CLIP embeddings for every image. Type "misty forest, lone figure" or pick an image and ask for more like it.
- **Automatic facets.** Zero-shot CLIP tagging for medium, style, subject, and tone, plus tags inferred from the folder structure. 388,000+ tags across the library.
- **Rich descriptions on request.** A per-image dossier (caption, subjects, medium, palette, mood, and identification) from a swappable vision backend: a local model through Ollama for bulk work, or Claude for gallery-grade identification.
- **A research corpus from scanned books.** 6,000+ book pages OCR'd on-device with Apple Vision, then indexed for full-text search and multilingual semantic search, so an English query finds the right passage in a Latin or German book.
- **A live web gallery.** Browse, stack facet filters, search, open "similar", read the dossier for any image, and reveal the file in Finder. Runs as an always-on local service.

## The part I care about most: grounded identification

A vision model asked "who made this?" will often answer confidently and wrongly. So identification here never comes from the model's memory. The model only identifies an image by matching it against **labeled neighbors already in the library**, and it has to report its confidence and say which neighbors it used.

I tested this with an A/B run. A small local model invented titles for images with meaningless hash filenames and marked them high confidence. Claude, given the same instruction, answered "reads like X, unidentified" with low confidence when no labeled neighbor existed, and cited the real neighbors when one did. The result shapes the design: the local model handles captions, tags, and search enrichment, where an error is cheap, and Claude handles identification, where an error misleads.

The same rule runs through the corrections. When OCR showed that 686 pages tagged as text were really plates and woodcuts, they were retagged with a recorded source (`source='ocr'`) in a reversible log, and the auto-tagger now leaves those corrections alone on reruns.

## Numbers

| | |
|---|---:|
| Images indexed | 39,576 |
| Facet tags | 388,110 |
| Rich AI descriptions | 1,974 |
| Books OCR'd into the research corpus | 200+ |

## Design choices

- **Local by default.** Embeddings, tagging, OCR, and search all run on the machine (Apple Silicon, MPS). The only cloud step is a description run, which I point at chosen folders; the images themselves stay local.
- **Portable.** The whole system moves between computers as one folder. A setup script rebuilds the environment and re-points the service; nothing is re-embedded or re-OCR'd.
- **Resumable everywhere.** Every long job commits as it goes and picks up where it stopped.
- **Tags keep their source.** CLIP, folder, LLM, and OCR tags coexist with a `source` column, so any layer can be rerun or rolled back without touching the others.

## Stack

Python 3.12 · open_clip (ViT-B-32, laion2b) on Apple MPS · SQLite + FTS5 · multilingual MiniLM embeddings · Apple Vision OCR (`ocrmac`) · Ollama (llava) or the Claude API for descriptions · a small stdlib HTTP server with a vanilla JS gallery

macOS only, because of Apple Vision OCR.

## Running it

```bash
bash _index/setup.sh              # build the Python 3.12 venv from requirements.txt
_index/.venv/bin/python _index/embed.py          # embed new or changed images
_index/.venv/bin/python _index/auto_tag.py       # zero-shot facets
_index/.venv/bin/python _index/search.py "misty forest, lone figure"
_index/.venv/bin/python _index/serve.py          # gallery at http://localhost:8756
bash _index/install-service.sh    # optional: always-on gallery service
```

Descriptions: `_index/describe.py --backend ollama` (free, local) or `--backend anthropic` (needs `ANTHROPIC_API_KEY`), with `--estimate` to price a run first.

More detail: `_index/README.md` (the index), `_index/PORTABILITY.md` (moving machines).
