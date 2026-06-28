"""
Build a region-level art gallery: embed multi-scale crops of every artwork.

Like embed_art.py, but each artwork contributes many crops (via crop_tiler), so a
sports photo can match a *detail* of a painting. CLIP embeds every crop; we save
the vectors plus, for each row, which artwork and which pixel box it came from.

Inputs:  data/art_order.json + data/art/*.jpg   (or --pairs to tile his pair art)
Outputs: data/art_crops_emb.npy   (M, 512) L2-normalized
         data/art_crops_meta.json [{art_file, box, title, artist, url}], row-aligned
"""

import os
import json
import argparse

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from crop_tiler import iter_crops

MODEL_NAME = "clip-ViT-B-32"
EMB_PATH = "data/art_crops_emb.npy"
META_PATH = "data/art_crops_meta.json"


def _met_sources():
    with open("data/art_order.json") as f:
        meta = json.load(f)
    for m in meta:
        yield "data/art", m["file"], m


def _pair_sources():
    with open("data/pairs.json") as f:
        pairs = json.load(f)
    for p in pairs:
        yield "data/pairs", p["art_file"], {"title": p["id"], "artist": "", "url": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true",
                    help="tile the curator's pair artworks instead of the Met DB")
    ap.add_argument("--scales", default="1.0,0.6,0.4",
                    help="comma list of window sizes as fraction of short side")
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()
    scales = tuple(float(x) for x in args.scales.split(","))

    sources = _pair_sources() if args.pairs else _met_sources()

    print(f"loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    crops, meta = [], []
    for art_dir, fname, info in sources:
        path = os.path.join(art_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        for box, crop in iter_crops(img, scales=scales, overlap=args.overlap):
            crops.append(crop)
            meta.append({
                "art_dir": art_dir,
                "art_file": fname,
                "box": list(box),
                "title": info.get("title", ""),
                "artist": info.get("artist", ""),
                "url": info.get("url", ""),
            })

    print(f"embedding {len(crops)} crops from {len(set(m['art_file'] for m in meta))} artworks ...")
    emb = model.encode(crops, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    np.save(EMB_PATH, emb)
    with open(META_PATH, "w") as f:
        json.dump(meta, f)
    print(f"\nsaved {len(meta)} crop embeddings to {EMB_PATH}")


if __name__ == "__main__":
    main()
