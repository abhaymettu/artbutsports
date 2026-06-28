"""
Step D: Did the adapter actually learn the curator's taste? Measure it.

Held-out retrieval metric. The gallery is every artwork in the pair set (and,
with --include-met, the Met database too). For each *held-out* sports photo we
rank its true paired artwork in the gallery by cosine similarity, then report
Recall@1/5/10 and median rank. We report the trained adapter AND the zero-shot
CLIP baseline on the same split -- the headline is the delta between them.

Caveat: "true artwork in gallery" is a proxy. The curator may have several valid
matches for one photo, so this under-counts good results -- but it is consistent,
so the adapter-vs-baseline comparison is fair.

Inputs:  data/pairs_emb.npz, data/pairs_split.json, data/taste_adapter.pt
         (optional) data/art_embeddings.npy as extra gallery distractors
Run:     python eval_taste.py [--include-met]
"""

import json
import argparse

import numpy as np

from taste_model import TasteAdapter, apply_adapter, ADAPTER_PATH

EMB_PATH = "data/pairs_emb.npz"
SPLIT_PATH = "data/pairs_split.json"
MET_EMB_PATH = "data/art_embeddings.npy"


def recall_and_rank(query, gallery, true_idx):
    """
    query (Q, d), gallery (G, d) both L2-normalized. true_idx (Q,) gives each
    query's correct row in the gallery. Returns metrics dict.
    """
    sims = query @ gallery.T                       # (Q, G)
    order = np.argsort(-sims, axis=1)              # ranked gallery indices per query
    ranks = np.array([np.where(order[i] == true_idx[i])[0][0] for i in range(len(query))])
    return {
        "R@1": float(np.mean(ranks < 1)),
        "R@5": float(np.mean(ranks < 5)),
        "R@10": float(np.mean(ranks < 10)),
        "median_rank": float(np.median(ranks) + 1),  # 1-based for readability
        "gallery_size": gallery.shape[0],
        "n_queries": len(query),
    }


def _print(name, m):
    print(
        f"{name:9s}  R@1 {m['R@1']:.3f}  R@5 {m['R@5']:.3f}  R@10 {m['R@10']:.3f}  "
        f"median_rank {m['median_rank']:.0f}  (Q={m['n_queries']}, gallery={m['gallery_size']})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-met", action="store_true",
                    help="add the Met art DB as extra distractors in the gallery")
    args = ap.parse_args()

    data = np.load(EMB_PATH, allow_pickle=True)
    sport = data["sport"].astype("float32")   # (N, d)
    art = data["art"].astype("float32")       # (N, d), row-aligned with sport
    ids = [str(x) for x in data["ids"]]

    with open(SPLIT_PATH) as f:
        val_ids = set(json.load(f)["val_ids"])
    val_rows = [i for i, pid in enumerate(ids) if pid in val_ids]
    if not val_rows:
        print("No held-out pairs found -- run train_taste.py first.")
        return
    val_rows = np.array(val_rows)

    # Gallery = all pair artworks; the true match for query row r is the same row r.
    extra = None
    if args.include_met:
        extra = np.load(MET_EMB_PATH).astype("float32")  # already L2-normalized

    print(f"Evaluating on {len(val_rows)} held-out sports photos.\n")

    # ---- baseline: raw zero-shot CLIP -------------------------------------
    base_gallery = art if extra is None else np.vstack([art, extra])
    base = recall_and_rank(sport[val_rows], base_gallery, val_rows)
    _print("baseline", base)

    # ---- adapter ----------------------------------------------------------
    model = TasteAdapter.load(ADAPTER_PATH)
    a_sport = apply_adapter(model, sport, "sport")
    a_art = apply_adapter(model, art, "art")
    a_extra = apply_adapter(model, extra, "art") if extra is not None else None
    adapt_gallery = a_art if a_extra is None else np.vstack([a_art, a_extra])
    adapt = recall_and_rank(a_sport[val_rows], adapt_gallery, val_rows)
    _print("adapter", adapt)

    print(
        f"\nDelta  R@1 {adapt['R@1'] - base['R@1']:+.3f}   "
        f"R@5 {adapt['R@5'] - base['R@5']:+.3f}   "
        f"R@10 {adapt['R@10'] - base['R@10']:+.3f}"
    )


if __name__ == "__main__":
    main()
