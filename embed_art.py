"""
Step 2: Embed every artwork with CLIP and save the vectors.

CLIP maps any image into a 512-dim vector where visually/semantically similar
images land near each other. We embed the whole art database once, normalize
the vectors (so dot product equals cosine similarity), and save to disk.

First run downloads the CLIP model (~600MB) and torch. That is a one-time cost.
"""

import os
import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

META_PATH = "data/art_metadata.json"
ART_DIR = "data/art"
EMB_PATH = "data/art_embeddings.npy"
ORDER_PATH = "data/art_order.json"

MODEL_NAME = "clip-ViT-B-32"  # fast, CPU-friendly. Swap to clip-ViT-L-14 for quality.


def main():
    with open(META_PATH) as f:
        meta = json.load(f)

    print(f"loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    images, kept = [], []
    for m in meta:
        path = os.path.join(ART_DIR, m["file"])
        try:
            images.append(Image.open(path).convert("RGB"))
            kept.append(m)
        except Exception as e:
            print(f"  skip {path}: {e}")

    print(f"embedding {len(images)} artworks ...")
    emb = model.encode(images, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    np.save(EMB_PATH, emb)
    with open(ORDER_PATH, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"\nsaved {len(kept)} embeddings to {EMB_PATH}")


if __name__ == "__main__":
    main()
