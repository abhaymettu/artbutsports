"""
Step C: Train the taste adapter with a CLIP-style contrastive loss.

We freeze CLIP (already baked into the cached vectors) and train only a small
projection head so paired (sport, art) embeddings land near each other while
non-paired ones are pushed apart -- symmetric InfoNCE with in-batch negatives,
the same objective CLIP itself was trained with.

Inputs:  data/pairs_emb.npz   (from embed_pairs.py)
Outputs: data/taste_adapter.pt  (the trained head, via taste_model.TasteAdapter.save)
         data/pairs_split.json  (which pair ids are val -- so eval uses the same held-out set)

Run:  python train_taste.py [--epochs 200] [--split] [--lr 1e-3]
"""

import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from taste_model import TasteAdapter, ADAPTER_PATH

EMB_PATH = "data/pairs_emb.npz"
SPLIT_PATH = "data/pairs_split.json"


def info_nce(sport, art, temp):
    """Symmetric InfoNCE. sport/art are (B, dim), L2-normalized. Positives on diagonal."""
    logits = (sport @ art.t()) / temp
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def train_val_split(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    return idx[n_val:], idx[:n_val]  # train, val


def run_epoch(model, sport, art, idx, temp, batch_size, opt=None):
    train = opt is not None
    model.train(train)
    perm = np.random.permutation(idx) if train else idx
    total, n_batches = 0.0, 0
    for start in range(0, len(perm), batch_size):
        b = perm[start : start + batch_size]
        if len(b) < 2:  # InfoNCE needs >=2 examples for a negative
            continue
        s = model.encode_sport(sport[b])
        a = model.encode_art(art[b])
        loss = info_nce(s, a, temp)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total += loss.item()
        n_batches += 1
    return total / max(1, n_batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--val-frac", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--split",
        action="store_true",
        help="use two separate heads for sport/art (default: one shared head)",
    )
    args = ap.parse_args()

    data = np.load(EMB_PATH, allow_pickle=True)
    sport = torch.from_numpy(data["sport"].astype("float32"))
    art = torch.from_numpy(data["art"].astype("float32"))
    ids = list(data["ids"])
    n, dim = sport.shape
    print(f"{n} pairs, dim={dim}")

    train_idx, val_idx = train_val_split(n, args.val_frac, args.seed)
    with open(SPLIT_PATH, "w") as f:
        json.dump(
            {"val_ids": [str(ids[i]) for i in val_idx],
             "train_ids": [str(ids[i]) for i in train_idx]},
            f, indent=2,
        )
    print(f"train={len(train_idx)}  val={len(val_idx)}")

    torch.manual_seed(args.seed)
    model = TasteAdapter(dim=dim, hidden=args.hidden, shared=not args.split)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_state = float("inf"), None
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, sport, art, train_idx, args.temp, args.batch_size, opt)
        if len(val_idx) >= 2:
            with torch.no_grad():
                va = run_epoch(model, sport, art, val_idx, args.temp, args.batch_size)
        else:
            va = float("nan")
        if ep % 20 == 0 or ep == 1:
            print(f"epoch {ep:4d}  train {tr:.4f}  val {va:.4f}")
        # Keep the best val checkpoint (falls back to train loss if val too small).
        score = va if len(val_idx) >= 2 else tr
        if score < best_val:
            best_val, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.save(ADAPTER_PATH)
    print(f"\nsaved adapter to {ADAPTER_PATH} (best score {best_val:.4f})")


if __name__ == "__main__":
    main()
