#!/usr/bin/env python3
"""
serve.py — the Art Collection master catalog + visual search engine.

A local, on-device web app over the CLIP index:
  - Browse the whole library (infinite scroll).
  - Faceted multi-select filters: Medium / Style / Subject / Tone / Folder.
    Pick many; within a facet = OR, across facets = AND. Combines with search.
  - Semantic text search ("sepia isometric engraving of a hexagonal box").
  - "◆ similar" on every tile for visual more-like-this.
No cloud, nothing leaves the Mac.

Run:
    python serve.py                 # then open http://localhost:8756
    python serve.py --port 9000

Facets come from the `facets` table (build/refresh with auto_tag.py).
Thumbnails are generated on the fly and cached in _index/.thumbs/.
"""

import argparse
import hashlib
import io
import json
import os
import sqlite3
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
THUMB_DIR = os.path.join(HERE, ".thumbs")
os.makedirs(THUMB_DIR, exist_ok=True)

import sys
sys.path.insert(0, HERE)
import vocab as V

# 'kind' is handled by the text-page toggle, not shown as chips.
FACET_ORDER = ["medium", "style", "subject", "tone", "folder"]
FACET_LABEL = {"medium": "Medium", "style": "Style / Era", "subject": "Subject",
               "tone": "Tone", "folder": "Folder"}

# ---- Index + facets loaded once into memory --------------------------------

print("Loading index...")
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
MODEL_ID = _conn.execute("SELECT value FROM meta WHERE key='model'").fetchone()[0]
ROOT = _conn.execute("SELECT value FROM meta WHERE key='root'").fetchone()[0]

_PATHS, _vecs = [], []
for _p, _emb in _conn.execute("SELECT path, embedding FROM images ORDER BY path"):
    _PATHS.append(_p)
    _vecs.append(np.frombuffer(_emb, dtype=np.float32))
MAT = np.vstack(_vecs)
PATHSET = set(_PATHS)
PATH_INDEX = {p: i for i, p in enumerate(_PATHS)}
N = len(_PATHS)

# Build (facet,tag) -> sorted index array, and per-facet ordered tag lists.
print("Loading facets...")
_members = {}   # (facet, tag) -> list[int]
for _p, _facet, _tag in _conn.execute("SELECT path, facet, tag FROM facets"):
    i = PATH_INDEX.get(_p)
    if i is not None:
        _members.setdefault((_facet, _tag), []).append(i)
TAGSETS = {k: np.array(sorted(v), dtype=int) for k, v in _members.items()}
FACET_TAGS = {}   # facet -> [(tag, count), ...] by count desc
for (facet, tag), arr in TAGSETS.items():
    FACET_TAGS.setdefault(facet, []).append((tag, len(arr)))
for facet in FACET_TAGS:
    FACET_TAGS[facet].sort(key=lambda x: (-x[1], x[0].lower()))

# Indices hidden from the gallery by default: scanned text pages + blank pages.
_hide_kinds = set(V.DOCUMENT_KINDS) | {"blank"}
_doc_arrays = [TAGSETS[("kind", t)] for t in _hide_kinds if ("kind", t) in TAGSETS]
DOC_IDX = (np.unique(np.concatenate(_doc_arrays)) if _doc_arrays
           else np.array([], dtype=int))
print(f"{N} images, model {MODEL_ID}, "
      f"{sum(len(v) for v in FACET_TAGS.values())} facet-tags, "
      f"{len(DOC_IDX)} document pages. Ready.")

# ---- Lazy CLIP text encoder (guarded) --------------------------------------

_model_lock = threading.Lock()
_model = _tokenizer = _device = None


def _ensure_model():
    global _model, _tokenizer, _device
    if _model is not None:
        return
    with _model_lock:
        if _model is not None:
            return
        import torch
        import open_clip

        name, pretrained = MODEL_ID.split("/", 1)
        _device = ("mps" if torch.backends.mps.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP {MODEL_ID} on {_device} (first query)...")
        m, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        _model = m.to(_device).eval()
        _tokenizer = open_clip.get_tokenizer(name)
        print("Model ready.")


def encode_text(text):
    import torch

    _ensure_model()
    with _model_lock:
        with torch.no_grad():
            f = _model.encode_text(_tokenizer([text]).to(_device))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)[0]


# ---- Filtering -------------------------------------------------------------

def candidates(selected):
    """selected: list of 'facet|tag'. Within facet OR, across facet AND.
    Returns a sorted index array, or None meaning 'everything'."""
    if not selected:
        return None
    by_facet = {}
    for s in selected:
        if "|" not in s:
            continue
        facet, tag = s.split("|", 1)
        by_facet.setdefault(facet, []).append(tag)
    cand = None
    for facet, tags in by_facet.items():
        union = None
        for t in tags:
            arr = TAGSETS.get((facet, t))
            if arr is None:
                continue
            union = arr if union is None else np.union1d(union, arr)
        if union is None:
            union = np.array([], dtype=int)
        cand = union if cand is None else np.intersect1d(cand, union)
    return cand


def effective(sel, docs):
    """Resolve selected facet filters + the document toggle to a final sorted
    index array. docs: 'hide' (default), 'show', or 'only'."""
    cand = candidates(sel)
    base = np.arange(N) if cand is None else cand
    if any(s.split("|", 1)[0] == "kind" for s in sel):
        return base  # explicit kind filter overrides the toggle
    if docs == "show":
        return base
    if docs == "only":
        return np.intersect1d(base, DOC_IDX)
    return np.setdiff1d(base, DOC_IDX)  # hide (default)


def _row(i, score=None):
    d = {"path": _PATHS[i], "folder": _PATHS[i].split(os.sep)[0]}
    if score is not None:
        d["score"] = float(score)
    return d


def rank(qvec, n, cand):
    scores = MAT @ qvec
    if cand is not None:
        order = cand[np.argsort(-scores[cand])[:n]] if len(cand) else []
    else:
        order = np.argsort(-scores)[:n]
    return [_row(i, scores[i]) for i in order]


def browse(cand, page, per):
    idx = np.arange(N) if cand is None else cand
    total = len(idx)
    chunk = idx[page * per: page * per + per]
    return total, [_row(i) for i in chunk]


# ---- Thumbnails ------------------------------------------------------------

def thumb_bytes(rel, size):
    key = hashlib.sha1(f"{rel}|{size}".encode()).hexdigest()
    cache = os.path.join(THUMB_DIR, key + ".jpg")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return f.read()
    with Image.open(os.path.join(ROOT, rel)) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        data = buf.getvalue()
    with open(cache, "wb") as f:
        f.write(data)
    return data


# ---- Page ------------------------------------------------------------------

PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>Art Collection — catalog</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--gold:#FAC359;--bg:#141310;--panel:#1e1c17;--ink:#e9e4d8;--dim:#9a927f;--line:#2c2921}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:10;background:rgba(20,19,16,.98);
 backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:12px 18px}
.top{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
h1{margin:0;font-size:15px;font-weight:600;letter-spacing:.02em;white-space:nowrap}
h1 span{color:var(--gold)}
input,button{font:inherit;color:var(--ink);background:var(--panel);
 border:1px solid #34302650;border-radius:8px;padding:9px 12px}
input#q{flex:1;min-width:240px}
input#q:focus{outline:none;border-color:var(--gold)}
button{cursor:pointer;background:var(--gold);color:#241f12;border:none;font-weight:600}
button.ghost{background:var(--panel);color:var(--ink);border:1px solid #3a3529;font-weight:500}
.facets{margin-top:10px;display:flex;flex-direction:column;gap:7px}
.fgroup{display:flex;gap:7px;align-items:flex-start}
.flabel{flex:0 0 74px;font-size:11px;color:var(--dim);text-transform:uppercase;
 letter-spacing:.05em;padding-top:5px}
.chips{display:flex;gap:5px;flex-wrap:wrap;flex:1;max-height:60px;overflow-y:auto}
.fgroup.open .chips{max-height:none}
.chip{cursor:pointer;background:var(--panel);color:var(--dim);border:1px solid var(--line);
 border-radius:999px;padding:3px 10px;font-size:12px;white-space:nowrap}
.chip:hover{color:var(--ink);border-color:#4a4436}
.chip.on{background:var(--gold);color:#241f12;border-color:var(--gold);font-weight:600}
.chip small{opacity:.55;margin-left:4px}
.more{font-size:11px;color:var(--gold);cursor:pointer;padding-top:5px;white-space:nowrap}
.meta{color:var(--dim);font-size:12px;margin-top:10px;display:flex;gap:12px;align-items:center}
.meta a{color:var(--gold);cursor:pointer}
.docs{margin-left:auto;color:var(--dim)}
.dbtn{cursor:pointer;border:1px solid var(--line);border-radius:6px;padding:2px 8px;
 margin-left:4px;font-size:11px;color:var(--dim)}
.dbtn.on{background:var(--gold);color:#241f12;border-color:var(--gold);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;padding:16px 18px}
figure{margin:0;background:var(--panel);border-radius:10px;overflow:hidden;border:1px solid #26231c;
 display:flex;flex-direction:column}
.imgwrap{position:relative;aspect-ratio:1;background:#0c0b09}
.imgwrap img{width:100%;height:100%;object-fit:contain;display:block;cursor:zoom-in}
.score{position:absolute;top:6px;left:6px;background:rgba(0,0,0,.66);color:var(--gold);
 font-size:11px;padding:2px 6px;border-radius:6px}
figcaption{padding:8px 9px;font-size:11px;color:var(--dim);word-break:break-all;
 flex:1;display:flex;flex-direction:column;gap:6px}
.fname{color:#cfc8b6}
.acts{display:flex;gap:6px;flex-wrap:wrap}
.sim{background:#2a2620;border:1px solid #3a3529;color:var(--ink);
 border-radius:6px;padding:3px 8px;font-size:11px;cursor:pointer}
.sim:hover{border-color:var(--gold);color:var(--gold)}
.foot{padding:10px 18px 70px;text-align:center;color:var(--dim)}
</style></head><body>
<header>
 <div class=top>
  <h1>Art Collection <span>·</span> catalog</h1>
  <input id=q placeholder="search: “sepia isometric engraving of a hexagonal box”…">
  <button onclick="doSearch()">Search</button>
  <button class=ghost onclick="clearAll()">Reset</button>
 </div>
 <div class=facets id=facets></div>
 <div class=meta>
  <span id=status>Loading…</span>
  <a id=clearf onclick="clearFilters()" style="display:none">clear filters</a>
  <span class=docs>text pages:
   <span class="dbtn on" data-d="hide" onclick="setDocs('hide')">hidden</span>
   <span class="dbtn" data-d="show" onclick="setDocs('show')">show</span>
   <span class="dbtn" data-d="only" onclick="setDocs('only')">only</span>
  </span>
 </div>
</header>
<div class=grid id=grid></div>
<div class=foot id=foot></div>
<script>
const grid=document.getElementById('grid'), status=document.getElementById('status'),
      foot=document.getElementById('foot'), facetsEl=document.getElementById('facets'),
      clearf=document.getElementById('clearf'), qEl=document.getElementById('q');
let mode='browse', query='', page=0, per=60, busy=false, done=false, total=0, docsMode='hide';
const selected=new Set();   // "facet|tag"
function setDocs(m){docsMode=m;
 document.querySelectorAll('.dbtn').forEach(b=>b.classList.toggle('on',b.dataset.d===m));
 rerun();}

function tile(r){
 const fig=document.createElement('figure');
 const wrap=document.createElement('div'); wrap.className='imgwrap';
 if(r.score!=null){const s=document.createElement('span');s.className='score';
  s.textContent=r.score.toFixed(3);wrap.appendChild(s);}
 const img=document.createElement('img'); img.loading='lazy';
 img.src='/thumb?path='+encodeURIComponent(r.path)+'&s=256';
 img.onclick=()=>window.open('/full?path='+encodeURIComponent(r.path));
 wrap.appendChild(img);
 const cap=document.createElement('figcaption');
 const nm=document.createElement('span'); nm.className='fname'; nm.textContent=r.path;
 const acts=document.createElement('span'); acts.className='acts';
 const sim=document.createElement('span'); sim.className='sim'; sim.textContent='◆ similar';
 sim.onclick=()=>doSimilar(r.path);
 const rev=document.createElement('span'); rev.className='sim'; rev.textContent='⌖ Finder';
 rev.onclick=()=>fetch('/reveal?path='+encodeURIComponent(r.path));
 acts.appendChild(sim); acts.appendChild(rev);
 cap.appendChild(nm); cap.appendChild(acts);
 fig.appendChild(wrap); fig.appendChild(cap);
 return fig;
}
function reset(){grid.innerHTML='';foot.textContent='';page=0;done=false;total=0;}
function append(items){items.forEach(r=>grid.appendChild(tile(r)));}
function selParams(){let p=[...selected].map(s=>'t='+encodeURIComponent(s));
 p.push('docs='+docsMode);return p.join('&');}

async function load(){
 if(busy||done) return; busy=true;
 const sel=selParams();
 let url;
 if(mode==='browse') url='/api/browse?page='+page+'&per='+per+(sel?'&'+sel:'');
 else if(mode==='search') url='/api/search?q='+encodeURIComponent(query)+'&n=200'+(sel?'&'+sel:'');
 else url='/api/similar?path='+encodeURIComponent(query)+'&n=150'+(sel?'&'+sel:'');
 const d=await (await fetch(url)).json();
 append(d.results);
 if(mode==='browse'){ total=d.total; page++; if(page*per>=total) done=true;
   status.textContent=total.toLocaleString()+' images'+(selected.size?' matching filters':'');
 } else { done=true;
   const label=mode==='search'?('“'+query+'”'):('images like '+query.split('/').pop());
   status.textContent=d.results.length+' results for '+label+(selected.size?' within filters':'');
 }
 foot.textContent=done?(grid.children.length+' shown'):'scroll for more…';
 busy=false;
 if(!done && document.body.offsetHeight<window.innerHeight+400) load();
}
function rerun(){reset(); if(mode==='browse')load(); else if(mode==='search')doSearch(); else load();}
function doSearch(){const v=qEl.value.trim();if(!v)return;mode='search';query=v;reset();load();}
function doSimilar(path){mode='similar';query=path;reset();window.scrollTo(0,0);load();}
function clearAll(){mode='browse';query='';qEl.value='';selected.clear();syncChips();rerun();}
function clearFilters(){selected.clear();syncChips();rerun();}
function syncChips(){
 [...facetsEl.querySelectorAll('.chip')].forEach(c=>c.classList.toggle('on',selected.has(c.dataset.k)));
 clearf.style.display=selected.size?'inline':'none';
}
function toggle(k){selected.has(k)?selected.delete(k):selected.add(k);syncChips();rerun();}

async function initFacets(){
 const d=await (await fetch('/api/facets')).json();
 d.facets.forEach(g=>{
  const row=document.createElement('div'); row.className='fgroup';
  const lab=document.createElement('div'); lab.className='flabel'; lab.textContent=g.label;
  const chips=document.createElement('div'); chips.className='chips';
  g.tags.forEach(t=>{const c=document.createElement('span');c.className='chip';
   c.dataset.k=g.facet+'|'+t.tag;
   c.innerHTML=t.tag+' <small>'+t.count.toLocaleString()+'</small>';
   c.onclick=()=>toggle(c.dataset.k); chips.appendChild(c);});
  row.appendChild(lab); row.appendChild(chips);
  if(g.tags.length>14){const m=document.createElement('span');m.className='more';m.textContent='＋ more';
   m.onclick=()=>{row.classList.toggle('open');m.textContent=row.classList.contains('open')?'– less':'＋ more';};
   row.appendChild(m);}
  facetsEl.appendChild(row);
 });
}
qEl.addEventListener('keydown',e=>{if(e.key==='Enter')doSearch();});
window.addEventListener('scroll',()=>{
 if(window.innerHeight+window.scrollY>=document.body.offsetHeight-500) load();
});
initFacets(); load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj), "application/json")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        sel = q.get("t", [])
        docs = q.get("docs", ["hide"])[0]
        try:
            if u.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")

            if u.path == "/api/facets":
                out = []
                for facet in FACET_ORDER:
                    tags = FACET_TAGS.get(facet, [])
                    out.append({
                        "facet": facet,
                        "label": FACET_LABEL.get(facet, facet),
                        "tags": [{"tag": t, "count": c} for t, c in tags],
                    })
                return self._json({"total": N, "facets": out})

            if u.path == "/api/browse":
                page = int(q.get("page", ["0"])[0])
                per = min(int(q.get("per", ["60"])[0]), 200)
                total, results = browse(effective(sel, docs), page, per)
                return self._json({"total": total, "results": results})

            if u.path == "/api/search":
                text = q.get("q", [""])[0]
                n = min(int(q.get("n", ["150"])[0]), 400)
                if not text.strip():
                    return self._json({"results": []})
                return self._json({"results": rank(encode_text(text), n, effective(sel, docs))})

            if u.path == "/api/similar":
                path = q.get("path", [""])[0]
                n = min(int(q.get("n", ["150"])[0]), 400)
                if path not in PATH_INDEX:
                    return self._json({"results": []})
                res = rank(MAT[PATH_INDEX[path]], n + 1, effective(sel, docs))
                return self._json({"results": [r for r in res if r["path"] != path][:n]})

            if u.path == "/thumb":
                path = q.get("path", [""])[0]
                size = min(int(q.get("s", ["256"])[0]), 1024)
                if path not in PATHSET:
                    return self._send(404, b"no", "text/plain")
                return self._send(200, thumb_bytes(path, size), "image/jpeg")

            if u.path == "/reveal":
                path = q.get("path", [""])[0]
                if path not in PATHSET:
                    return self._send(404, b"no", "text/plain")
                import subprocess
                subprocess.run(["open", "-R", os.path.join(ROOT, path)], check=False)
                return self._json({"ok": True})

            if u.path == "/full":
                path = q.get("path", [""])[0]
                if path not in PATHSET:
                    return self._send(404, b"no", "text/plain")
                full = os.path.join(ROOT, path)
                ext = os.path.splitext(full)[1].lower().lstrip(".")
                mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif",
                        "webp": "webp", "bmp": "bmp", "tif": "tiff",
                        "tiff": "tiff"}.get(ext, "jpeg")
                with open(full, "rb") as f:
                    return self._send(200, f.read(), f"image/{mime}")

            self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as e:
            self._send(500, f"error: {e}", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8756)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n  Art Collection catalog running at  {url}")
    print("  (Ctrl-C to stop)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
