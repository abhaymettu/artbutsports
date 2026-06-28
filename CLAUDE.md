# CLAUDE.md

ArtButSports MVP — a CLIP-based retrieval pipeline that matches a sports photo to the
closest artwork in the Met Museum's open collection. No training, no scraping; pure
nearest-neighbor on CLIP embeddings. This is the baseline to tune from.

## Architecture

Three standalone scripts run in order, each reading/writing files under `data/`:

1. `fetch_art.py` — pulls figural artworks from the Met open API (no key needed) into
   `data/art/*.jpg` and writes `data/art_metadata.json`. Searches departments 11
   (European Paintings) and 21 (Modern/Contemporary) across pose/drama-oriented terms.
   Idempotent: re-running re-downloads but is safe.
2. `embed_art.py` — CLIP-embeds every image (`clip-ViT-B-32`, 512-dim), L2-normalizes
   so dot product == cosine similarity, and writes `data/art_embeddings.npy` plus
   `data/art_order.json` (the row-aligned metadata; images that fail to open are
   skipped, so this order — not `art_metadata.json` — is the source of truth for
   embedding rows).
3. `match.py <photo.jpg> [top_k]` — embeds the query, ranks by `emb @ q`, prints the
   top matches, and saves a side-by-side `match_preview.jpg`.

Key invariant: **`art_embeddings.npy` rows align positionally with `art_order.json`**,
not with `art_metadata.json`. If you change embedding logic, keep them in sync.

The model name `clip-ViT-B-32` is hardcoded in `embed_art.py`, `match.py`, and
`embed_pairs.py` — changing it requires editing all three and re-running the embed steps.

## Taste adapter (learning the curator's taste)

A second pipeline trains a lightweight projection head on the curator's real
sports↔art pairs so retrieval reflects his matches, not generic CLIP similarity.
CLIP stays frozen; only the head trains (contrastive InfoNCE). Flow:

- `collect_pairs.py` → `data/pairs/` + `data/pairs.json` (manual: composite-split or manifest; no scraping)
- `embed_pairs.py` → `data/pairs_emb.npz` (frozen CLIP vectors for both sides)
- `train_taste.py` → `data/taste_adapter.pt` + `data/pairs_split.json` (held-out ids)
- `eval_taste.py` → Recall@1/5/10 + median rank, **adapter vs zero-shot baseline**
- `taste_model.py` — shared `TasteAdapter` (residual MLP, L2-normed) + `apply_adapter()`;
  imported by training, eval, and `match.py`.

`match.py` auto-applies the adapter when `data/taste_adapter.pt` exists (sport side for
the query, art side for the gallery), applied to the *already-saved* CLIP embeddings —
no heavy re-embed. `--no-taste` forces the raw baseline. Default shared head; `--split`
in `train_taste.py` uses two heads.

### Data source: scraping @ArtButSports on X

Pairs come from `x.com/ArtButSports` via `gallery-dl --cookies-from-browser chrome:Default
-D data/scrape_x "https://x.com/ArtButSports/media"` (needs the user's logged-in Chrome
session; macOS Keychain must be unlocked). Each post is a single composite image, **not**
two separate images. `collect_pairs.py --raw-dir data/scrape_x` splits them
orientation-aware: **landscape → sport LEFT | art RIGHT, portrait → sport TOP / art BOTTOM**
(sport always first; `--art-first` flips). Use `--gap-frac 0.015` to drop the divider seam.

### Gallery source: WikiArt (not the Met API)

The retrieval gallery is built by `fetch_wikiart.py`, which bulk-streams the HF
`huggan/wikiart` dataset (filtered to figural genres: religious/genre/nude/portrait/
illustration), downscales to 512px, and **appends** to `data/art/` + `art_metadata.json`.
This replaced `fetch_art.py` (the Met API now sits behind Cloudflare: needs a browser
UA + backoff, and still throttles to ~14 img/min — too slow for bulk). `fetch_art.py`
still works for small top-ups. Current gallery ≈ 3420 artworks (3000 WikiArt + Met).

### Region-level (crop) matching

`crop_tiler.py` (`iter_crops`) yields the full image plus multi-scale sliding sub-windows,
so a sports photo can match a *detail* of a painting, not just the whole canvas.
`embed_art_crops.py` builds `data/art_crops_emb.npy` + `data/art_crops_meta.json` (each row
carries `art_dir`, `art_file`, `box`); `--pairs` tiles the curator's art instead of the Met.
`match.py --crops` retrieves over that gallery and the preview shows the winning crop.

## Setup & run

```bash
source venv/bin/activate          # venv/ already exists in the working dir
pip install -r requirements.txt   # sentence-transformers, torch, pillow, numpy, requests
python fetch_art.py               # network; minutes
python embed_art.py               # first run downloads CLIP (~600MB) + torch, then cached
python match.py my_sports_pic.jpg
```

## Notes

- Current state: ~55 artworks fetched/embedded (README mentions a TARGET of 300 — raise
  `TARGET` in `fetch_art.py` and re-fetch to grow the database).
- Everything runs CPU-friendly; no GPU assumed.
- See `README.md` for the tuning roadmap (larger CLIP model, pose/keypoint signal,
  contrastive fine-tuning on real pairs).
