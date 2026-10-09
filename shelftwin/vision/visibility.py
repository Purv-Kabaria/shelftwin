"""Person/cart visibility mask (spec Section 3.3).

v_s = 1 - |mask ∩ slot| / |slot|. The same mask is used to black out people in
every saved crop (privacy).
"""
from __future__ import annotations

import cv2
import numpy as np

COCO_OCCLUDERS = {"person", "backpack", "handbag", "suitcase"}


class VisibilityMasker:
    def __init__(self, weights: str = "yolo11n-seg.pt", imgsz: int = 640, conf: float = 0.3,
                 classes: set[str] | None = None, device=None, model=None):
        if model is None:
            from ultralytics import YOLO
            model = YOLO(weights)
        self.model = model
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.classes = classes or (COCO_OCCLUDERS | {"cart", "basket", "shopping cart"})

    def mask(self, bgr: np.ndarray) -> np.ndarray:
        """Returns a uint8 {0,1} occlusion mask the size of the frame."""
        h, w = bgr.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, device=self.device,
                               retina_masks=True, verbose=False)[0]
        if r.masks is None:
            return out
        names = r.names
        for m, c in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            if names[c] in self.classes:
                mm = m if m.shape == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                out |= (mm > 0.5).astype(np.uint8)
        # Dilate: hands and arms around the body also hide packs
        k = max(3, int(0.01 * max(h, w)) | 1)
        return cv2.dilate(out, np.ones((k, k), np.uint8))


def slot_visibility(mask: np.ndarray, slot_polygons: dict[str, np.ndarray]) -> dict[str, float]:
    """slot_polygons: slot_id -> (K,2) image-pixel polygon."""
    vis = {}
    for sid, poly in slot_polygons.items():
        sm = np.zeros_like(mask)
        cv2.fillPoly(sm, [np.asarray(poly, dtype=np.int32)], 1)
        area = int(sm.sum())
        vis[sid] = 1.0 if area == 0 else 1.0 - float((sm & mask).sum()) / area
    return vis


def redact(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Black out masked (people) pixels before any crop is stored."""
    out = bgr.copy()
    out[mask.astype(bool)] = 0
    return out
