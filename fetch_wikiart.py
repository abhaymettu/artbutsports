"""
Build a large, diverse art gallery from WikiArt (bulk, no rate limits).

The Met's API throttles per-image requests; instead we stream the Hugging Face
`huggan/wikiart` dataset (81k paintings) in bulk and keep the figural genres the
curator actually matches on (religious / history / genre / nude / portrait scenes).

Images are downscaled to 512px on save -- CLIP only sees 224px, so this is lossless
for matching and cuts disk ~3x. Output matches embed_art.py's expected layout, and
we APPEND to any existing data/art + metadata (e.g. the Met images already pulled).

Run:  python fetch_wikiart.py [--target 3000] [--max-px 512]
"""

import os
import json
import argparse

from datasets import load_dataset

OUT_DIR = "data/art"
META_PATH = "data/art_metadata.json"

# Figural genres -- bodies, poses, drama. Skip landscape/cityscape/still_life/abstract.
KEEP_GENRES = {
    "religious_painting", "genre_painting", "nude_painting",
    "portrait", "illustration",
}


def downscale(img, max_px):
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        s = max_px / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3000, help="new WikiArt images to add")
    ap.add_argument("--max-px", type=int, default=512, help="downscale long side to this")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Append to whatever is already there (keeps the Met images).
    meta = []
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    have = len(meta)
    print(f"existing gallery: {have} artworks")

    ds = load_dataset("huggan/wikiart", split="train", streaming=True).shuffle(
        seed=0, buffer_size=10000
    )
    genre_names = ds.features["genre"].names
    artist_names = ds.features["artist"].names
    style_names = ds.features["style"].names

    added = 0
    for ex in ds:
        if added >= args.target:
            break
        genre = genre_names[ex["genre"]]
        if genre not in KEEP_GENRES:
            continue
        fname = f"wikiart_{have + added}.jpg"
        try:
            downscale(ex["image"], args.max_px).save(
                os.path.join(OUT_DIR, fname), "JPEG", quality=88
            )
        except Exception as e:
            print(f"  skip: {e}")
            continue
        meta.append({
            "id": fname.rsplit(".", 1)[0],
            "file": fname,
            "title": f"{style_names[ex['style']].replace('_',' ')} ({genre.replace('_',' ')})",
            "artist": artist_names[ex["artist"]].replace("_", " "),
            "date": "",
            "url": "https://www.wikiart.org/",
        })
        added += 1
        if added % 100 == 0:
            print(f"  [{added}/{args.target}] genre={genre}")
            with open(META_PATH, "w") as f:
                json.dump(meta, f, indent=2)

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nadded {added} WikiArt images; gallery now {len(meta)} artworks in {OUT_DIR}/")


if __name__ == "__main__":
    main()
