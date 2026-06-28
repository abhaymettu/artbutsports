"""
Step 1: Build the art database from the Met Museum's free open API.

Downloads ~300 figural artworks (paintings + sculpture) plus their metadata.
No API key needed. Results are cached so you can re-run it safely.
"""

import os
import json
import time
import random
import requests

MET_SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECTS = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
MET_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

# The Met API now sits behind Cloudflare and 403s the default requests UA, so
# use a shared session with a browser-like User-Agent for every call.
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
})


def get_with_retry(url, params=None, timeout=30, tries=4):
    """
    GET via the shared session, backing off on Cloudflare rate-limit (403/429).

    The Met's API throttles bursts: after a few dozen rapid calls it returns 403
    for everything. On a 403/429 we pause with growing delays so the limiter
    resets, then retry. Returns the Response (caller checks status) or None.
    """
    delay = 5
    for attempt in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
        except Exception:
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code in (403, 429) and attempt < tries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        return r
    return r

# Departments with dramatic figural work (poses / bodies / action):
#   11 = European Paintings        21 = Modern and Contemporary Art
#   13 = Greek and Roman Art       14 = European Sculpture & Decorative Arts
#    9 = Drawings and Prints       19 = Photographs
DEPARTMENTS = [11, 21, 13, 14, 9, 19]

# Terms that surface poses / drama / bodies, which is what this account matches on
SEARCH_TERMS = [
    "portrait", "battle", "dance", "saint", "figure", "mythology",
    "martyr", "hunt", "wrestling", "horse", "angel", "procession",
    "fight", "fall", "victory", "triumph", "athlete", "nude",
    "warrior", "leap", "struggle", "embrace", "crowd", "lament",
    "river", "death", "hero", "boxer", "runner", "gesture",
]

TARGET = 3000
OUT_DIR = "data/art"
META_PATH = "data/art_metadata.json"


def collect_object_ids():
    """
    Build a large candidate pool two ways and merge them:

      1. Department listings (/objects?departmentIds=) -- one request per
         department returns *every* object id in it, for sheer volume.
      2. Figural term searches -- fewer ids but biased toward poses/bodies/drama.

    Image availability is filtered later in fetch_details_and_images.
    """
    seen, ids = set(), []

    def add(new_ids):
        for oid in (new_ids or []):
            if oid not in seen:
                seen.add(oid)
                ids.append(oid)

    # 1. Department-wide listings (huge pool).
    for dept in DEPARTMENTS:
        try:
            r = get_with_retry(MET_OBJECTS, params={"departmentIds": dept})
            r.raise_for_status()
            add(r.json().get("objectIDs"))
        except Exception as e:
            print(f"  dept listing failed ({dept}): {e}")
        time.sleep(1.0)

    # 2. Figural term searches (relevance boost; failures are non-fatal).
    for term in SEARCH_TERMS:
        try:
            r = get_with_retry(MET_SEARCH, params={"q": term, "hasImages": "true"})
            r.raise_for_status()
            add(r.json().get("objectIDs"))
        except Exception as e:
            print(f"  search failed ({term}): {e}")
        time.sleep(1.0)

    random.shuffle(ids)
    print(f"collected {len(ids)} candidate object ids")
    return ids


def fetch_details_and_images(ids, target):
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = []
    for oid in ids:
        if len(meta) >= target:
            break
        try:
            r = get_with_retry(MET_OBJECT.format(oid))
            if r.status_code != 200:
                continue
            obj = r.json()
        except Exception:
            continue

        img_url = obj.get("primaryImageSmall")
        if not img_url:
            continue
        try:
            img = SESSION.get(img_url, timeout=60)
            if img.status_code != 200:
                continue
        except Exception:
            continue

        fname = f"{oid}.jpg"
        with open(os.path.join(OUT_DIR, fname), "wb") as f:
            f.write(img.content)

        meta.append({
            "id": oid,
            "file": fname,
            "title": obj.get("title") or "Untitled",
            "artist": obj.get("artistDisplayName") or "Unknown",
            "date": obj.get("objectDate") or "",
            "url": obj.get("objectURL") or "",
        })
        print(f"  [{len(meta)}/{target}] {meta[-1]['title']}")
        time.sleep(0.3)
        # Persist incrementally so a mid-run rate-limit block never discards
        # progress or overwrites the metadata with an empty list.
        if len(meta) % 50 == 0:
            with open(META_PATH, "w") as f:
                json.dump(meta, f, indent=2)

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {len(meta)} artworks to {OUT_DIR}/ and metadata to {META_PATH}")
    return meta


if __name__ == "__main__":
    ids = collect_object_ids()
    fetch_details_and_images(ids, TARGET)
