#!/usr/bin/env python3
"""
describe.py — Layer 3: per-image AI dossiers over the CLIP catalog.

For each image, a vision model produces a structured dossier:
  caption, description, subjects[], medium, style[], palette[], mood,
  sd_prompt, sd_negative, identification, id_confidence, id_basis

The identification is GROUNDED: before asking the model what a piece *is*, we
pull the image's nearest neighbors from the existing CLIP index (index.db) and
feed any DESCRIPTIVELY-named neighbors (artist folders, real titles — not hashes)
as hints. The model may assert an artist/title/source ONLY when a close, clearly-
labeled neighbor supports it; otherwise it says "reads like X" with low/none
confidence. It never invents provenance from memory.

Backends (swappable — one function `describe_image(path, grounding) -> dict`):
  --backend anthropic   Claude vision API (best quality). Model default
                        claude-opus-4-8 (--model to override). Needs
                        ANTHROPIC_API_KEY. Prints a running cost estimate.
  --backend ollama      Local VLM via Ollama (free, on-device, owned). Model
                        default llava:7b. Nothing leaves the Mac.

Storage (index.db, same DB as the CLIP index):
  - descriptions(path PK, caption, description, sd_prompt, sd_negative,
    dossier_json, identification, id_confidence, model, ts)
  - facets: subjects/style/medium exploded with source='llm' so they power the
    gallery's faceted filters alongside the zero-shot CLIP tags.

Resumable + idempotent: paths already in `descriptions` are skipped. Commit is
per-image. Safe to Ctrl-C and rerun.

Examples:
    # free local demo on 10 krampus images
    python describe.py --backend ollama --folder krampus --limit 10

    # gallery-grade dossiers via Claude on a hero folder (needs API key)
    ANTHROPIC_API_KEY=... python describe.py --backend anthropic --folder "fantasy art" --limit 25

    # priority sweep (fantasy art, krampus, esoterica, murals) — dry cost estimate
    python describe.py --backend anthropic --priority --estimate
"""

import argparse
import base64
import io
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "index.db")

# --- Anthropic vision pricing (per 1M tokens). Verified 2026-08-24 against the
#     claude-api skill. claude-opus-4-8: $5 in / $25 out.  Keep in sync if the
#     model changes; the table is keyed by model id. -------------------------
PRICING = {
    "claude-opus-4-8":  (5.0, 25.0),
    "claude-opus-4-7":  (5.0, 25.0),
    "claude-sonnet-5":  (3.0, 15.0),
    "claude-haiku-4-5": (1.0,  5.0),
}
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_OLLAMA_MODEL = "llava:7b"

# Priority folders (PLAN Phase 3): do these first.
PRIORITY_FOLDERS = ["fantasy art", "krampus", "esoterica metaphysical", "murals"]

# Send images downscaled — plenty for description, and it caps API image tokens.
MAX_EDGE = 1024
N_NEIGHBORS = 6          # neighbors pulled for grounding
GROUND_MIN_SCORE = 0.55  # ignore neighbors below this cosine sim entirely

HASH_RE = re.compile(r"^[0-9a-f]{16,}$", re.I)

# ---- Dossier schema (shared by both backends) ------------------------------

DOSSIER_FIELDS = ["caption", "description", "subjects", "medium", "style",
                  "palette", "mood", "sd_prompt", "sd_negative",
                  "identification", "id_confidence", "id_basis"]

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "caption": {"type": "string", "description": "One short sentence — what this image is."},
        "description": {"type": "string", "description": "2-4 sentences: subject, composition, medium, mood, notable detail."},
        "subjects": {"type": "array", "items": {"type": "string"}, "description": "Concrete nouns depicted (e.g. 'krampus', 'birch forest', 'lantern')."},
        "medium": {"type": "string", "description": "Best single guess at medium/technique (e.g. 'ink drawing', 'oil painting', 'engraving', 'digital')."},
        "style": {"type": "array", "items": {"type": "string"}, "description": "Style / era descriptors (e.g. 'folk horror', 'art nouveau', 'woodcut')."},
        "palette": {"type": "array", "items": {"type": "string"}, "description": "3-6 dominant colors in plain words (e.g. 'oxblood red', 'soot black', 'bone white')."},
        "mood": {"type": "string", "description": "Emotional register in a few words."},
        "sd_prompt": {"type": "string", "description": "A vivid Stable Diffusion prompt that would reproduce this image's subject + style + palette. Comma-separated tags/phrases."},
        "sd_negative": {"type": "string", "description": "A Stable Diffusion negative prompt (things to avoid)."},
        "identification": {"type": "string", "description": "What the piece actually is — artist/title/source IF grounded by a labeled neighbor, else 'reads like ...' or 'unidentified'. NEVER invent provenance."},
        "id_confidence": {"type": "string", "enum": ["high", "medium", "low", "none"]},
        "id_basis": {"type": "string", "description": "Why — cite the grounding neighbor that supports the ID, or 'no labeled match'."},
    },
    "required": DOSSIER_FIELDS,
}

SYSTEM_PROMPT = (
    "You are an art cataloguer building a reference dossier for one image in a "
    "personal art-reference library. Be concrete and specific; describe only "
    "what you can see.\n\n"
    "GROUNDED IDENTIFICATION — the one hard rule: you may state an artist, title, "
    "or source ONLY when a visually-similar neighbor with a DESCRIPTIVE (non-hash) "
    "filename or an artist folder is given below AND it clearly matches. In that "
    "case cite it and set id_confidence accordingly (a 0.85+ match with a clear "
    "title -> high; a looser or partial match -> medium/low). If no labeled "
    "neighbor supports an identification, write 'reads like <tradition/artist>' "
    "or 'unidentified', set id_confidence to 'low' or 'none', and set id_basis to "
    "'no labeled match'. NEVER invent an artist, title, date, or provenance from "
    "memory. A hash-named neighbor is NOT evidence of identity.\n\n"
    "Return ONLY a JSON object with exactly these keys: " + ", ".join(DOSSIER_FIELDS) + "."
)


# ---- Index / neighbors -----------------------------------------------------

def load_index(conn):
    root = os.path.dirname(HERE)  # portable: derive from script location, not stored absolute path
    paths, vecs = [], []
    for p, e in conn.execute("SELECT path, embedding FROM images ORDER BY path"):
        paths.append(p)
        vecs.append(np.frombuffer(e, dtype=np.float32))
    mat = np.vstack(vecs)
    index = {p: i for i, p in enumerate(paths)}
    return root, paths, mat, index


def is_descriptive(path):
    """A neighbor path is 'descriptive' if its filename stem is word-like (a real
    title), not a hex hash or a numeric/coded id. Artist subfolders also count."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if HASH_RE.match(stem):
        return False
    letters = re.sub(r"[^a-z]", "", stem.lower())
    if re.search(r"[a-z]{4,}", stem.lower()) and len(letters) >= 4:
        return True
    # descriptive artist/collection subfolder (2+ levels deep, word-like)
    parts = path.split(os.sep)[1:-1]
    return any(re.search(r"[a-z]{4,}", seg.lower()) for seg in parts)


def neighbors_for(i, paths, mat, index, k=N_NEIGHBORS):
    scores = mat @ mat[i]
    order = np.argsort(-scores)
    out = []
    for j in order:
        if j == i:
            continue
        s = float(scores[j])
        if s < GROUND_MIN_SCORE:
            break
        out.append((paths[j], s, is_descriptive(paths[j])))
        if len(out) >= k:
            break
    return out


def grounding_text(neigh):
    if not neigh:
        return "No visually-similar images are in the library."
    lines = ["Visually-similar images already in the library (cosine similarity):"]
    for p, s, desc in neigh:
        tag = "  [descriptive name — usable for ID]" if desc else "  [hash/coded name — not usable for ID]"
        lines.append(f"- {p} ({s:.2f}){tag}")
    return "\n".join(lines)


# ---- Image loading ---------------------------------------------------------

def load_jpeg_b64(abspath, max_edge=MAX_EDGE):
    with Image.open(abspath) as im:
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    data = buf.getvalue()
    return base64.b64encode(data).decode("ascii"), len(data)


# ---- Backends: describe_image(path, grounding) -> (dossier, usage) ----------

class AnthropicBackend:
    def __init__(self, model):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / profile
        self.in_price, self.out_price = PRICING.get(model, (5.0, 25.0))

    def describe(self, abspath, grounding):
        b64, _ = load_jpeg_b64(abspath)
        tool = {
            "name": "record_dossier",
            "description": "Record the structured dossier for this image.",
            "input_schema": JSON_SCHEMA,
        }
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "record_dossier"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": grounding + "\n\nCatalogue this image."},
                ],
            }],
        )
        dossier = None
        for block in msg.content:
            if block.type == "tool_use" and block.name == "record_dossier":
                dossier = block.input
                break
        if dossier is None:
            raise RuntimeError("model did not return a dossier tool call")
        u = msg.usage
        cost = (u.input_tokens / 1e6) * self.in_price + (u.output_tokens / 1e6) * self.out_price
        return dossier, {"in": u.input_tokens, "out": u.output_tokens, "cost": cost}


class OllamaBackend:
    def __init__(self, model, host="http://localhost:11434"):
        self.model = model
        self.host = host

    def describe(self, abspath, grounding):
        b64, _ = load_jpeg_b64(abspath)
        prompt = (SYSTEM_PROMPT + "\n\n" + grounding +
                  "\n\nCatalogue this image. Respond with the JSON object only.")
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.loads(r.read())
        dossier = json.loads(resp["response"])
        return dossier, {"in": 0, "out": 0, "cost": 0.0}


# ---- Normalization + storage -----------------------------------------------

def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [s.strip() for s in re.split(r"[;,]", str(v)) if s.strip()]


def _as_str(v):
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return "" if v is None else str(v)


def normalize(d):
    out = {}
    out["caption"] = _as_str(d.get("caption"))
    out["description"] = _as_str(d.get("description"))
    out["subjects"] = _as_list(d.get("subjects"))
    out["medium"] = _as_str(d.get("medium")).strip()
    out["style"] = _as_list(d.get("style"))
    out["palette"] = _as_list(d.get("palette"))
    out["mood"] = _as_str(d.get("mood"))
    out["sd_prompt"] = _as_str(d.get("sd_prompt"))
    out["sd_negative"] = _as_str(d.get("sd_negative"))
    out["identification"] = _as_str(d.get("identification"))
    conf = _as_str(d.get("id_confidence")).lower().strip()
    out["id_confidence"] = conf if conf in ("high", "medium", "low", "none") else "none"
    out["id_basis"] = _as_str(d.get("id_basis"))
    return out


def ensure_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS descriptions (
            path          TEXT PRIMARY KEY,
            caption       TEXT,
            description   TEXT,
            sd_prompt     TEXT,
            sd_negative   TEXT,
            dossier_json  TEXT,
            identification TEXT,
            id_confidence TEXT,
            model         TEXT,
            ts            REAL
        )
    """)
    # facets needs a 'source' column AND it must be part of the primary key, so
    # a CLIP tag and an LLM tag with the same (path,facet,tag) can coexist instead
    # of overwriting each other. The original PK was (path,facet,tag) — migrate it.
    info = list(conn.execute("PRAGMA table_info(facets)"))
    cols = [r[1] for r in info]
    pk_cols = [r[1] for r in info if r[5] > 0]           # r[5] = pk position
    if "source" not in cols or "source" not in pk_cols:
        conn.execute("""
            CREATE TABLE facets_new (
                path   TEXT NOT NULL,
                facet  TEXT NOT NULL,
                tag    TEXT NOT NULL,
                score  REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'clip',
                PRIMARY KEY (path, facet, tag, source)
            )""")
        if "source" in cols:
            conn.execute("INSERT OR IGNORE INTO facets_new "
                         "SELECT path,facet,tag,score,COALESCE(source,'clip') FROM facets")
        else:
            conn.execute("INSERT OR IGNORE INTO facets_new "
                         "SELECT path,facet,tag,score,'clip' FROM facets")
        conn.execute("DROP TABLE facets")
        conn.execute("ALTER TABLE facets_new RENAME TO facets")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facets_ft ON facets(facet, tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facets_path ON facets(path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facets_source ON facets(source)")
    conn.commit()


def store(conn, path, dossier, model):
    conn.execute(
        "INSERT OR REPLACE INTO descriptions(path,caption,description,sd_prompt,"
        "sd_negative,dossier_json,identification,id_confidence,model,ts) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (path, dossier["caption"], dossier["description"], dossier["sd_prompt"],
         dossier["sd_negative"], json.dumps(dossier), dossier["identification"],
         dossier["id_confidence"], model, time.time()),
    )
    # explode into facets (source='llm'); replace any prior llm rows for this path
    conn.execute("DELETE FROM facets WHERE path=? AND source='llm'", (path,))
    rows = []
    for tag in dossier["subjects"]:
        rows.append((path, "subject", tag.lower(), 1.0, "llm"))
    for tag in dossier["style"]:
        rows.append((path, "style", tag.lower(), 1.0, "llm"))
    if dossier["medium"]:
        rows.append((path, "medium", dossier["medium"].lower(), 1.0, "llm"))
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO facets(path,facet,tag,score,source) VALUES(?,?,?,?,?)",
            rows)
    conn.commit()


# ---- Target selection ------------------------------------------------------

def select_targets(conn, folders, limit, redo):
    done = set(r[0] for r in conn.execute("SELECT path FROM descriptions"))
    out = []
    if folders:
        for f in folders:
            pref = f.rstrip("/")
            for (p,) in conn.execute(
                "SELECT path FROM images WHERE path=? OR path LIKE ? ORDER BY path",
                (pref, pref + "/%")):
                if redo or p not in done:
                    out.append(p)
    else:
        for (p,) in conn.execute("SELECT path FROM images ORDER BY path"):
            if redo or p not in done:
                out.append(p)
    if limit:
        out = out[:limit]
    return out


# ---- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["anthropic", "ollama"], default="anthropic")
    ap.add_argument("--model", help="override model id for the backend")
    ap.add_argument("--folder", action="append", dest="folders",
                    help="restrict to a top-level folder (repeatable)")
    ap.add_argument("--priority", action="store_true",
                    help="do the priority folders: " + ", ".join(PRIORITY_FOLDERS))
    ap.add_argument("--limit", type=int, help="cap number of images (for tests)")
    ap.add_argument("--redo", action="store_true", help="re-describe even if already done")
    ap.add_argument("--estimate", action="store_true",
                    help="just print how many images + a cost estimate, don't run")
    ap.add_argument("--show", action="store_true", help="print each dossier as it's made")
    args = ap.parse_args()

    model = args.model or (DEFAULT_ANTHROPIC_MODEL if args.backend == "anthropic"
                           else DEFAULT_OLLAMA_MODEL)

    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_schema(conn)

    folders = args.folders or []
    if args.priority:
        folders = list(dict.fromkeys(folders + PRIORITY_FOLDERS))

    targets = select_targets(conn, folders, args.limit, args.redo)
    scope = (", ".join(folders) if folders else "whole library")

    if args.estimate:
        n = len(targets)
        print(f"Scope: {scope}")
        print(f"Images needing a dossier: {n:,}")
        if args.backend == "anthropic":
            inp, outp = PRICING.get(model, (5.0, 25.0))
            # rough: ~1.4k image tokens @1024px + ~0.5k prompt in; ~0.4k out
            per = (1900 / 1e6) * inp + (400 / 1e6) * outp
            print(f"Model: {model}  (${inp}/1M in, ${outp}/1M out)")
            print(f"Rough estimate: ~${per:.3f}/image  ->  ~${per*n:,.2f} total")
            print("(actual cost is metered per image during a real run)")
        else:
            print(f"Model: {model}  (local Ollama — free, on-device)")
        return

    if not targets:
        print(f"Nothing to do — every image in scope ({scope}) already has a dossier.")
        return

    print(f"Loading CLIP index for grounding...")
    root, paths, mat, index = load_index(conn)

    backend = (AnthropicBackend(model) if args.backend == "anthropic"
               else OllamaBackend(model))

    print(f"Backend: {args.backend} / {model}")
    print(f"Scope: {scope} — {len(targets):,} image(s)\n")

    total_cost = 0.0
    ok = fail = 0
    t0 = time.time()
    for n, path in enumerate(targets, 1):
        abspath = os.path.join(root, path)
        try:
            i = index.get(path)
            neigh = neighbors_for(i, paths, mat, index) if i is not None else []
            dossier_raw, usage = backend.describe(abspath, grounding_text(neigh))
            dossier = normalize(dossier_raw)
            store(conn, path, dossier, model)
            total_cost += usage["cost"]
            ok += 1
            badge = {"high": "★", "medium": "◈", "low": "·", "none": "—"}.get(
                dossier["id_confidence"], "—")
            cost_s = f"  ${usage['cost']:.3f}" if usage["cost"] else ""
            print(f"[{n}/{len(targets)}] {badge} {path}{cost_s}")
            print(f"        {dossier['caption']}")
            if dossier["id_confidence"] in ("high", "medium"):
                print(f"        ID: {dossier['identification']} "
                      f"({dossier['id_confidence']}) — {dossier['id_basis']}")
            if args.show:
                print("        " + json.dumps(dossier, ensure_ascii=False))
        except KeyboardInterrupt:
            print("\nInterrupted — progress saved (resumable).")
            break
        except Exception as e:
            fail += 1
            print(f"[{n}/{len(targets)}] !! {path}\n        ERROR: {e}")

    dt = time.time() - t0
    print(f"\nDone: {ok} ok, {fail} failed in {dt:.0f}s ({dt/max(ok,1):.1f}s/img).")
    if total_cost:
        print(f"Total metered cost: ${total_cost:.2f}  (${total_cost/max(ok,1):.3f}/image)")


if __name__ == "__main__":
    main()
