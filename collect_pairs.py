"""
Step A: Assemble the curator's sports<->art pairs into a clean dataset.

This is the *gating* step of taste-learning: the model can only be as good as
these pairs. We deliberately do NOT scrape Instagram here -- that's a ToS / image
copyright grey area. Instead you supply the raw images and this script normalizes
them into a consistent layout the rest of the pipeline consumes.

Two input modes (you can use either or both):

  1. COMPOSITE mode -- point --raw-dir at a folder of the curator's composite
     posts (default data/pairs_raw; use data/scrape_x for gallery-dl output).
     Each image is split into a sport half and an art half, orientation-aware:
     landscape posts split LEFT|RIGHT, portrait posts split TOP/BOTTOM. The sports
     photo is assumed first (left/top), art second (flip the whole set with
     --art-first).

  2. MANIFEST mode -- write data/pairs_manifest.json listing explicit pairs:
       [{"id": "001", "sport": "path_or_url", "art": "path_or_url",
         "source_url": "https://..."}]
     Local paths are copied; http(s) URLs are downloaded.

Output (consumed by embed_pairs.py):
    data/pairs/<id>_sport.jpg
    data/pairs/<id>_art.jpg
    data/pairs.json   ->  [{id, sport_file, art_file, source_url}]
"""

import os
import csv
import json
import shutil
import argparse

import requests
from PIL import Image

RAW_DIR = "data/pairs_raw"
MANIFEST_PATH = "data/pairs_manifest.json"
OUT_DIR = "data/pairs"
PAIRS_PATH = "data/pairs.json"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _save_rgb(img, path):
    img.convert("RGB").save(path, "JPEG", quality=92)


def _load_image(src):
    """Load an image from a local path or an http(s) URL."""
    if isinstance(src, str) and src.lower().startswith(("http://", "https://")):
        r = requests.get(src, timeout=60)
        r.raise_for_status()
        from io import BytesIO

        return Image.open(BytesIO(r.content))
    return Image.open(src)


def _split_composite(im, art_first=False, gap_frac=0.0):
    """
    Split one composite into (sport_half, art_half).

    The curator lays pairs out by orientation: landscape posts are split
    LEFT|RIGHT, portrait posts TOP/BOTTOM. The sports photo is consistently the
    first half (left or top); the artwork is second. ``art_first`` flips that.
    """
    w, h = im.size
    if w >= h:  # landscape -> vertical seam, split left | right
        mid = w // 2
        gap = int(w * gap_frac / 2)
        first = im.crop((0, 0, mid - gap, h))
        second = im.crop((mid + gap, 0, w, h))
    else:       # portrait -> horizontal seam, split top / bottom
        mid = h // 2
        gap = int(h * gap_frac / 2)
        first = im.crop((0, 0, w, mid - gap))
        second = im.crop((0, mid + gap, w, h))
    return (second, first) if art_first else (first, second)


def from_composites(raw_dir, out_dir, art_first=False, gap_frac=0.0):
    """Split every composite in raw_dir (orientation-aware) into sport/art halves."""
    if not os.path.isdir(raw_dir):
        return []
    pairs = []
    files = sorted(
        f for f in os.listdir(raw_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS
    )
    for f in files:
        pid = os.path.splitext(f)[0]
        try:
            im = Image.open(os.path.join(raw_dir, f)).convert("RGB")
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
        sport_im, art_im = _split_composite(im, art_first=art_first, gap_frac=gap_frac)

        sport_file, art_file = f"{pid}_sport.jpg", f"{pid}_art.jpg"
        _save_rgb(sport_im, os.path.join(out_dir, sport_file))
        _save_rgb(art_im, os.path.join(out_dir, art_file))
        pairs.append(
            {"id": pid, "sport_file": sport_file, "art_file": art_file, "source_url": ""}
        )
        print(f"  [composite] {pid}")
    return pairs


def _read_manifest(path):
    if path.lower().endswith(".csv"):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    with open(path) as f:
        return json.load(f)


def from_manifest(out_dir, manifest_path):
    """Copy/download explicit pairs listed in a JSON or CSV manifest."""
    if not os.path.isfile(manifest_path):
        return []
    pairs = []
    for i, row in enumerate(_read_manifest(manifest_path)):
        pid = str(row.get("id") or i)
        try:
            sport_im = _load_image(row["sport"])
            art_im = _load_image(row["art"])
        except Exception as e:
            print(f"  skip {pid}: {e}")
            continue
        sport_file, art_file = f"{pid}_sport.jpg", f"{pid}_art.jpg"
        _save_rgb(sport_im, os.path.join(out_dir, sport_file))
        _save_rgb(art_im, os.path.join(out_dir, art_file))
        pairs.append(
            {
                "id": pid,
                "sport_file": sport_file,
                "art_file": art_file,
                "source_url": row.get("source_url", ""),
            }
        )
        print(f"  [manifest] {pid}")
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Build the curator's sport<->art pair set.")
    ap.add_argument(
        "--raw-dir",
        default=RAW_DIR,
        help=f"folder of composite posts to split (default {RAW_DIR}; e.g. data/scrape_x)",
    )
    ap.add_argument(
        "--art-first",
        action="store_true",
        help="composites have ART first (left/top) and sport second (default: sport first)",
    )
    ap.add_argument(
        "--gap-frac",
        type=float,
        default=0.0,
        help="fraction dropped at the center seam to remove a divider line (e.g. 0.02)",
    )
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    pairs = []
    pairs += from_composites(
        args.raw_dir, OUT_DIR, art_first=args.art_first, gap_frac=args.gap_frac
    )
    pairs += from_manifest(OUT_DIR, args.manifest)

    # De-dup by id (manifest wins over composite on collision).
    by_id = {}
    for p in pairs:
        by_id[p["id"]] = p
    pairs = list(by_id.values())

    with open(PAIRS_PATH, "w") as f:
        json.dump(pairs, f, indent=2)

    if not pairs:
        print(
            "\nNo pairs found. Add side-by-side images to data/pairs_raw/ "
            f"and/or write {MANIFEST_PATH}. See this file's docstring."
        )
    else:
        print(f"\nsaved {len(pairs)} pairs to {OUT_DIR}/ and {PAIRS_PATH}")


if __name__ == "__main__":
    main()
