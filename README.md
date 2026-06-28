# ArtButSports

Drop in a sports photo, get back the artwork it "rhymes" with — the way the
[@ArtButSports](https://x.com/ArtButSports) account pairs sports moments with classical art.

Under the hood: CLIP embeds every artwork and your photo into the same vector space and
ranks by cosine similarity. On top of that, a small **taste adapter** — trained on the
curator's *actual* sports↔art pairs — re-weights that space toward the kinds of matches
he makes. You can also match against a **crop** of a painting, so a detail of a large
canvas can win, not just the whole image.

> **Data is not included in this repo** (it's large and not redistributable — scraped
> posts, WikiArt images, embeddings, the trained model). Everything under `data/` is
> git-ignored. The steps below rebuild it locally.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
The first embed/match step downloads the CLIP model (~600 MB) + torch — one-time, then cached.

## 1. Build the art gallery (what you search against)

```bash
python fetch_wikiart.py            # ~3000 figural paintings from HF huggan/wikiart -> data/art/
python embed_art.py                # whole-image embeddings -> data/art_embeddings.npy
python embed_art_crops.py --scales 0.5 --overlap 0.3   # region crops -> data/art_crops_emb.npy
```
`fetch_wikiart.py` streams in bulk (no rate limits) and keeps only figural genres
(religious / genre / nude / portrait / illustration), downscaled to 512px. `fetch_art.py`
(Met Museum API) still works for small top-ups but is heavily rate-limited.

## 2. Learn the curator's taste (optional but recommended)

Train a lightweight projection head on his real pairs with a contrastive (InfoNCE) loss.
CLIP stays frozen; only the head trains — cheap and low-overfit.

**a. Collect his pairs.** His posts are single composite images (sport + art side-by-side).
Pull them from X with [`gallery-dl`](https://github.com/mikf/gallery-dl) using your own
logged-in browser session (no password needed):

```bash
gallery-dl --cookies-from-browser chrome -D data/scrape_x \
  "https://x.com/ArtButSports/media"
python collect_pairs.py --raw-dir data/scrape_x --gap-frac 0.015
```
`collect_pairs.py` splits each composite orientation-aware (landscape → sport left / art
right; portrait → sport top / art bottom; `--art-first` flips). You can also supply pairs
manually via `data/pairs_raw/` or a `data/pairs_manifest.json` — see its docstring.

> Note: scraping is subject to X's ToS and the images are copyrighted. Keep this to
> personal / research use.

**b. Embed, train, evaluate:**
```bash
python embed_pairs.py              # frozen CLIP embeddings of both sides -> data/pairs_emb.npz
python train_taste.py --weight-decay 1e-3   # train head -> data/taste_adapter.pt
python eval_taste.py               # Recall@1/5/10 + median rank, adapter vs baseline
```

## 3. Match a photo

```bash
python match.py my_photo.jpg            # taste-aware, whole artworks
python match.py my_photo.jpg --crops    # match a detail/region of a painting
python match.py my_photo.jpg --no-taste # raw CLIP, for comparison
```
`match.py` auto-uses `data/taste_adapter.pt` if present. Each run prints the top matches
and saves a side-by-side `match_preview.jpg`.

## Files

| File | Role |
|------|------|
| `fetch_wikiart.py` / `fetch_art.py` | build the art gallery (WikiArt / Met) |
| `embed_art.py` / `embed_art_crops.py` | CLIP-embed the gallery (whole / region crops) |
| `crop_tiler.py` | multi-scale crop generator |
| `collect_pairs.py` | normalize the curator's composite posts into sport/art pairs |
| `embed_pairs.py` | cache CLIP embeddings of both sides of each pair |
| `taste_model.py` | the projection-head adapter (shared/load/apply) |
| `train_taste.py` / `eval_taste.py` | train the adapter / measure it vs baseline |
| `match.py` | rank the gallery for a query photo |

## Ideas to push further

- **More pairs** is the biggest lever — the adapter overfits at a few hundred pairs.
- Bigger CLIP (`clip-ViT-L-14`) in the embed/match steps for quality.
- A pose/keypoint signal concatenated into the adapter input (his matches rhyme on pose).

## Credits & licensing

Inspired by [@ArtButSports](https://x.com/ArtButSports) (LJ Rader). Art from
[WikiArt](https://www.wikiart.org/) (non-commercial research use) and the
[Met Open Access](https://www.metmuseum.org/art/collection) collection. Sports photos and
curated pairings belong to their respective owners. This is a personal research project.
