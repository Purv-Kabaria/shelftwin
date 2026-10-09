"""Product identity: catalogue retrieval with abstention (spec Section 3.6).

Rule: the expected product of a slot NEVER enters the match score. Identity is
decided from appearance alone; location is compared afterwards by the planogram
aligner, so a misplaced pack is not "corrected" into the expected product.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UNKNOWN_SKU = "unknown"


@dataclass
class IdentityConfig:
    accept_sim: float = 0.75
    accept_margin: float = 0.15
    k: int = 10


class Embedder:
    """DINOv2 ViT-S/14 + 256-d head, as trained in notebooks/04_embedder_arcface.ipynb.

    checkpoint: .pt saved by the notebook ({'backbone', 'head', 'dim', 'source': 'timm'|'hub'}).
    If None, uses the pretrained DINOv2 CLS token (384-d) zero-shot.
    """

    def __init__(self, checkpoint: str | None = None, device: str | None = None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
        source = (ck or {}).get("source", "timm")
        if source == "timm":
            import timm
            kw = (ck or {}).get("timm_kwargs") or {"img_size": 224}
            self.backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=ck is None,
                                              num_classes=0, **kw)
        else:
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        self.head = None
        if ck:
            self.backbone.load_state_dict(ck["backbone"])
            self.head = torch.nn.Linear(384, ck.get("dim", 256))
            self.head.load_state_dict(ck["head"])
            self.head.to(self.device).eval()
        self.backbone.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def __call__(self, crops_bgr: list[np.ndarray], batch: int = 64) -> np.ndarray:
        import cv2
        torch = self.torch
        out = []
        for i in range(0, len(crops_bgr), batch):
            ims = [cv2.resize(c[..., ::-1], (224, 224), interpolation=cv2.INTER_AREA) for c in crops_bgr[i:i + batch]]
            x = torch.from_numpy(np.stack(ims)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - self.mean) / self.std
            with torch.no_grad():
                f = self.backbone(x)
                if self.head is not None:
                    f = self.head(f)
                f = torch.nn.functional.normalize(f, dim=1)
            out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 256), np.float32)


class Catalogue:
    """Gallery of SKU embeddings. FAISS IndexFlatIP if available, else numpy."""

    def __init__(self, dim: int, cfg: IdentityConfig | None = None):
        self.dim = dim
        self.cfg = cfg or IdentityConfig()
        self.vecs = np.zeros((0, dim), np.float32)
        self.labels: list[str] = []
        self.meta: dict[str, dict] = {}
        try:
            import faiss
            self._faiss = faiss
            self.index = faiss.IndexFlatIP(dim)
        except ImportError:
            self._faiss = None
            self.index = None

    def add(self, sku_id: str, embeddings: np.ndarray, meta: dict | None = None) -> None:
        """Add 3-5 gallery images of a product: no retraining needed."""
        e = _l2norm(np.asarray(embeddings, np.float32).reshape(-1, self.dim))
        self.vecs = np.concatenate([self.vecs, e])
        self.labels += [sku_id] * len(e)
        if meta:
            self.meta[sku_id] = meta
        if self.index is not None:
            self.index.add(e)

    def search(self, q: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        k = min(k or self.cfg.k, len(self.labels))
        q = _l2norm(np.asarray(q, np.float32).reshape(-1, self.dim))
        if self.index is not None:
            return self.index.search(q, k)
        sims = q @ self.vecs.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, 1), idx

    def identify(self, q: np.ndarray) -> list[dict]:
        """Per-query: best SKU, s1 (top SKU sim), s2 (best *other* SKU sim), accept or 'unknown'."""
        if not self.labels:
            return [dict(sku_id=UNKNOWN_SKU, s1=0.0, s2=0.0, accepted=False, candidates=[]) for _ in range(len(q))]
        sims, idx = self.search(q, k=max(self.cfg.k, 20))
        out = []
        for srow, irow in zip(sims, idx):
            best: dict[str, float] = {}
            for s, i in zip(srow, irow):
                if i < 0:
                    continue
                lab = self.labels[i]
                best[lab] = max(best.get(lab, -1.0), float(s))
            ranked = sorted(best.items(), key=lambda x: -x[1])
            s1 = ranked[0][1]
            s2 = ranked[1][1] if len(ranked) > 1 else 0.0
            ok = s1 >= self.cfg.accept_sim and (s1 - s2) >= self.cfg.accept_margin
            out.append(dict(sku_id=ranked[0][0] if ok else UNKNOWN_SKU, s1=s1, s2=s2, accepted=ok,
                            candidates=ranked[:3]))
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vecs.npy", self.vecs)
        (path / "labels.json").write_text(json.dumps({"labels": self.labels, "meta": self.meta,
                                                      "accept_sim": self.cfg.accept_sim,
                                                      "accept_margin": self.cfg.accept_margin}))

    @classmethod
    def load(cls, path: str | Path) -> "Catalogue":
        path = Path(path)
        vecs = np.load(path / "vecs.npy")
        d = json.loads((path / "labels.json").read_text())
        cat = cls(vecs.shape[1], IdentityConfig(d.get("accept_sim", 0.75), d.get("accept_margin", 0.15)))
        labels = d["labels"]
        for lab in dict.fromkeys(labels):
            m = np.array([x == lab for x in labels])
            cat.add(lab, vecs[m], d["meta"].get(lab))
        return cat


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
