"""
Step B: Cache frozen CLIP embeddings for both sides of every pair.

Training (train_taste.py) runs many epochs over the data, so we embed the images
with CLIP exactly once here and train on the cached vectors -- no image decoding
in the training loop. Same model + L2-normalize convention as embed_art.py so the
adapter sees vectors identical to what retrieval produces.

Output: data/pairs_emb.npz  with arrays
    sport (N, 512), art (N, 512)   -- L2-normalized, row-aligned
    ids   (N,)                     -- pair ids, for traceability
"""

import os
import json

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

PAIRS_PATH = "data/pairs.json"
PAIRS_DIR = "data/pairs"
EMB_PATH = "data/pairs_emb.npz"

MODEL_NAME = "clip-ViT-B-32"  # must match embed_art.py / match.py


def _normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)
    if not pairs:
        print(f"{PAIRS_PATH} is empty -- run collect_pairs.py first.")
        return

    print(f"loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    sport_imgs, art_imgs, ids = [], [], []
    for p in pairs:
        sp = os.path.join(PAIRS_DIR, p["sport_file"])
        ap = os.path.join(PAIRS_DIR, p["art_file"])
        try:
            s = Image.open(sp).convert("RGB")
            a = Image.open(ap).convert("RGB")
        except Exception as e:
            print(f"  skip {p['id']}: {e}")
            continue
        sport_imgs.append(s)
        art_imgs.append(a)
        ids.append(p["id"])

    print(f"embedding {len(ids)} pairs ...")
    sport_emb = _normalize(
        model.encode(sport_imgs, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    )
    art_emb = _normalize(
        model.encode(art_imgs, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    )

    np.savez(EMB_PATH, sport=sport_emb, art=art_emb, ids=np.array(ids))
    print(f"\nsaved {len(ids)} pair embeddings to {EMB_PATH}")


if __name__ == "__main__":
    main()
