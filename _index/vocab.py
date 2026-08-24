"""
vocab.py — the controlled vocabulary for auto-tagging (Layer 2).

Each facet maps concept LABELS to a natural-language PROMPT that CLIP scores
every image against. Edit freely: add/remove labels, tune prompts, then rerun
`python auto_tag.py` — it's fast (a few minutes, on-device, free) and idempotent.

- SINGLE_LABEL facets assign exactly one winner per image (the best match).
- MULTI_LABEL facets assign every label whose confidence clears the threshold.

Good prompts are specific and natural ("an antique engraving", not "engraving").
"""

# facet -> { label: prompt }
FACETS = {
    # kind: separates real artwork from scanned book/text pages so the gallery
    # can hide documents by default. Single-label (winner takes the image).
    "kind": {
        "artwork": "an illustration, drawing, painting, photograph or artwork",
        "text page": "a scanned page of dense printed text from a book",
        "title page": "a scanned book title page or table of contents",
        "text with figure": "a scanned book page with text and a small diagram",
    },
    "medium": {
        "engraving": "an antique engraving or etching print",
        "woodcut": "a woodcut or linocut print",
        "lithograph": "a vintage lithograph print",
        "pen and ink": "a pen and ink drawing",
        "pencil sketch": "a pencil sketch drawing",
        "watercolor": "a watercolor painting",
        "oil painting": "an oil painting",
        "gouache": "a gouache or acrylic painting",
        "photograph": "a photograph",
        "digital art": "a digital illustration or painting",
        "3d render": "a 3D computer render",
        "collage": "a collage or assemblage",
        "screenprint": "a screenprinted poster",
        "comic": "a comic book or cartoon illustration",
        "map": "an old map",
        "diagram": "a technical diagram or schematic drawing",
        "pattern": "a decorative textile or wallpaper pattern",
        "typography": "typography and lettering",
        "sculpture": "a photograph of a sculpture or physical object",
    },
    "style": {
        "art nouveau": "an art nouveau illustration",
        "art deco": "an art deco design",
        "victorian": "a victorian era illustration",
        "medieval": "a medieval illuminated manuscript",
        "renaissance": "a renaissance artwork",
        "baroque": "a baroque artwork",
        "folk art": "a naive folk art illustration",
        "steampunk": "a steampunk illustration",
        "fantasy": "a fantasy art illustration",
        "sci-fi": "a science fiction illustration",
        "psychedelic": "a psychedelic poster",
        "mid-century": "a mid-century modern design",
        "minimalist": "a minimalist design",
        "gothic": "a dark gothic illustration",
        "japanese": "a japanese ukiyo-e woodblock print",
        "childrens book": "a children's picture book illustration",
        "scientific": "a scientific or botanical illustration plate",
        "occult": "an esoteric occult or alchemical illustration",
    },
    "subject": {
        "human figure": "a human figure",
        "portrait": "a face or portrait",
        "animal": "an animal",
        "bird": "a bird",
        "insect": "an insect",
        "sea creature": "a fish or sea creature",
        "plant": "a plant or flower",
        "forest": "trees and forest",
        "landscape": "a landscape",
        "water": "water, ocean or a lake",
        "architecture": "a building or architecture",
        "interior": "an interior room",
        "machine": "a machine or mechanical device",
        "vehicle": "a vehicle",
        "ship": "a ship or boat",
        "celestial": "stars, planets or astronomy",
        "anatomy": "a skeleton or anatomical drawing",
        "mythical creature": "a mythical creature or monster",
        "devil": "a devil or demon",
        "skull": "a skull or death imagery",
        "ornament": "a decorative border or ornament",
        "still life object": "a single object or still life",
        "costume": "a costume or fashion illustration",
        "weapon": "a weapon or armor",
        "map subject": "a map or geographic chart",
        "abstract": "an abstract pattern",
    },
    "tone": {
        "sepia": "a sepia-toned brown antique image",
        "monochrome": "a black and white image",
        "muted": "a muted desaturated image",
        "vibrant": "a vibrant saturated color image",
        "warm": "an image with a warm red orange palette",
        "cool": "an image with a cool blue green palette",
        "pastel": "a soft pastel colored image",
    },
}

SINGLE_LABEL = {"kind", "medium", "tone"}
MULTI_LABEL = {"style", "subject"}

# Multi-label gate: rank concepts by raw CLIP similarity, keep the top ones whose
# score clears the floor (the single best is always kept so nothing is empty).
MULTI_FLOOR = 0.20      # absolute cosine similarity floor for a multi-label tag
MULTI_MAX = 4           # ...up to this many per facet

# kind-facet labels that count as "a document, not artwork" — the gallery hides
# these by default (with a toggle) and they're the OCR/research candidates.
DOCUMENT_KINDS = {"text page", "title page", "text with figure"}

# An image is only flagged as a document when the best document-kind score beats
# the 'artwork' score by at least this margin. Higher = fewer false positives
# (real art wrongly hidden) but more text pages slipping through. 'artwork' is
# always the default winner otherwise.
KIND_MARGIN = 0.03
