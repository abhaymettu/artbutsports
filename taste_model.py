"""
Shared taste-adapter module.

The adapter is a small projection head trained on top of frozen CLIP embeddings
(see train_taste.py). It re-weights CLIP's 512-dim space toward the kinds of
sports<->art matches the curator actually makes. Training and retrieval both
import from here so they apply *exactly* the same transform.

A trained adapter is saved as a single checkpoint dict:
    {"config": {...}, "shared": {...}, "sport": {...}, "art": {...}}
where the per-side keys hold torch state_dicts (or are absent when shared).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_PATH = "data/taste_adapter.pt"


class ProjectionHead(nn.Module):
    """512 -> hidden -> dim MLP with a residual + L2-normalized output."""

    def __init__(self, dim=512, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        # Residual so the head starts near identity and only learns a correction.
        out = x + self.net(x)
        return F.normalize(out, dim=-1)


class TasteAdapter(nn.Module):
    """
    Wraps one shared head (default) or two separate heads for sport/art.

    A shared head keeps the two modalities in a single comparable space and
    halves the parameter count -- the right default for small datasets. Flip
    ``shared=False`` once there are enough pairs to support two heads.
    """

    def __init__(self, dim=512, hidden=512, shared=True):
        super().__init__()
        self.shared = shared
        self.dim = dim
        self.hidden = hidden
        if shared:
            self.head = ProjectionHead(dim, hidden)
        else:
            self.sport_head = ProjectionHead(dim, hidden)
            self.art_head = ProjectionHead(dim, hidden)

    def encode_sport(self, x):
        return self.head(x) if self.shared else self.sport_head(x)

    def encode_art(self, x):
        return self.head(x) if self.shared else self.art_head(x)

    # ---- persistence -------------------------------------------------------
    def save(self, path=ADAPTER_PATH):
        torch.save(
            {
                "config": {"dim": self.dim, "hidden": self.hidden, "shared": self.shared},
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path=ADAPTER_PATH, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        cfg = ckpt["config"]
        model = cls(dim=cfg["dim"], hidden=cfg["hidden"], shared=cfg["shared"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


def apply_adapter(model, emb, side):
    """
    Transform a numpy array of CLIP embeddings (N, dim) through the adapter.

    ``side`` is "sport" or "art". Returns an L2-normalized numpy array, ready
    for cosine retrieval via a plain dot product. Used by match.py / eval_taste.py.
    """
    import numpy as np

    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(np.asarray(emb, dtype="float32"))
        if t.ndim == 1:
            t = t.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        out = model.encode_sport(t) if side == "sport" else model.encode_art(t)
        out = out.cpu().numpy()
    return out[0] if squeeze else out
