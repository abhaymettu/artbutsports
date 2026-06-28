"""
Step 3: Match a sports photo to its closest artwork.

Usage:
    python match.py photo.jpg                 (top 5, whole-artwork gallery)
    python match.py photo.jpg 8               (top 8)
    python match.py photo.jpg --crops         (match a *region* of an artwork)
    python match.py photo.jpg --no-taste      (raw CLIP, ignore the adapter)

If a trained taste adapter exists at data/taste_adapter.pt it is applied to both
the gallery and the query before ranking, so matches reflect the curator's taste.

With --crops the gallery is multi-scale crops (build it with embed_art_crops.py),
so a sports photo can match a *detail* of a painting; the preview shows that crop.
Saves a side-by-side preview of the #1 match to match_preview.jpg.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from taste_model import TasteAdapter, apply_adapter, ADAPTER_PATH

WHOLE_EMB = "data/art_embeddings.npy"
WHOLE_META = "data/art_order.json"
WHOLE_DIR = "data/art"
CROP_EMB = "data/art_crops_emb.npy"
CROP_META = "data/art_crops_meta.json"
MODEL_NAME = "clip-ViT-B-32"


def load_gallery(use_crops):
    """Return (emb, rows) where each row dict has art_dir, art_file, box(optional), label."""
    if use_crops:
        emb = np.load(CROP_EMB)
        meta = json.load(open(CROP_META))
        rows = [
            {"art_dir": m["art_dir"], "art_file": m["art_file"], "box": m.get("box"),
             "label": f"{m.get('title','')} {m.get('artist','')}".strip(), "url": m.get("url", "")}
            for m in meta
        ]
    else:
        emb = np.load(WHOLE_EMB)
        meta = json.load(open(WHOLE_META))
        rows = [
            {"art_dir": WHOLE_DIR, "art_file": m["file"], "box": None,
             "label": f"{m['title']} by {m['artist']} ({m.get('date','')})", "url": m.get("url", "")}
            for m in meta
        ]
    return emb, rows


def crop_of(row):
    """Open the artwork and crop to the matching region (if any)."""
    img = Image.open(os.path.join(row["art_dir"], row["art_file"])).convert("RGB")
    return img.crop(tuple(row["box"])) if row.get("box") else img


def make_montage(query_path, row, out_path):
    q = Image.open(query_path).convert("RGB")
    a = crop_of(row)
    h = 500

    def resize_h(im):
        w = max(1, int(im.width * h / im.height))
        return im.resize((w, h))

    q, a = resize_h(q), resize_h(a)
    canvas = Image.new("RGB", (q.width + a.width + 20, h), "white")
    canvas.paste(q, (0, 0))
    canvas.paste(a, (q.width + 20, 0))
    canvas.save(out_path)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        print("usage: python match.py photo.jpg [top_k] [--crops] [--no-taste]")
        return
    query_path = args[0]
    top_k = int(args[1]) if len(args) > 1 else 5
    use_crops = "--crops" in flags

    emb, rows = load_gallery(use_crops)

    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([Image.open(query_path).convert("RGB")], convert_to_numpy=True)[0]
    q = q / np.linalg.norm(q)

    # Apply the taste adapter if present (sport side for query, art side for gallery).
    if "--no-taste" not in flags and os.path.exists(ADAPTER_PATH):
        adapter = TasteAdapter.load(ADAPTER_PATH)
        emb = apply_adapter(adapter, emb, "art")
        q = apply_adapter(adapter, q, "sport")
        print(f"(taste adapter{' + crops' if use_crops else ''})")
    else:
        print(f"(zero-shot CLIP{' + crops' if use_crops else ''})")

    sims = emb @ q
    idx = np.argsort(-sims)[:top_k]

    print(f"\nTop {top_k} matches for {query_path}:\n")
    for rank, i in enumerate(idx, 1):
        r = rows[i]
        box = f"  crop={r['box']}" if r.get("box") else ""
        print(f"{rank}. {r['label']}  sim={sims[i]:.3f}{box}")
        print(f"   {os.path.join(r['art_dir'], r['art_file'])}   {r['url']}")

    try:
        make_montage(query_path, rows[idx[0]], "match_preview.jpg")
        print("\nsaved side-by-side preview to match_preview.jpg")
    except Exception as e:
        print(f"\n(montage skipped: {e})")


if __name__ == "__main__":
    main()
