"""Per-bay change gate (spec Section 3.2): only changed bays go to heavy models."""
from __future__ import annotations

import cv2
import numpy as np


class ChangeGate:
    def __init__(self, width: int = 160, pixel_thresh: int = 25, changed_frac: float = 0.02,
                 force_every_s: float = 60.0):
        self.width = width
        self.pixel_thresh = pixel_thresh
        self.changed_frac = changed_frac
        self.force_every_s = force_every_s
        self._ref: dict[str, np.ndarray] = {}
        self._last_full: float | None = None

    def _small(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        scale = self.width / w
        g = cv2.cvtColor(cv2.resize(bgr, (self.width, max(1, int(h * scale)))), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (5, 5), 0)

    def changed_bays(self, bgr: np.ndarray, t: float, bays: dict[str, tuple[int, int, int, int]]) -> list[str]:
        """bays: bay_id -> (x1, y1, x2, y2) ROI in full-frame pixels."""
        if self._last_full is None or t - self._last_full >= self.force_every_s:
            self._last_full = t
            for bay_id, roi in bays.items():
                self._ref[bay_id] = self._small(self._crop(bgr, roi))
            return list(bays)
        out = []
        for bay_id, roi in bays.items():
            cur = self._small(self._crop(bgr, roi))
            ref = self._ref.get(bay_id)
            if ref is None or ref.shape != cur.shape:
                out.append(bay_id)
            elif (cv2.absdiff(cur, ref) > self.pixel_thresh).mean() > self.changed_frac:
                out.append(bay_id)
            if bay_id in out:
                self._ref[bay_id] = cur
        return out

    @staticmethod
    def _crop(bgr, roi):
        x1, y1, x2, y2 = roi
        return bgr[y1:y2, x1:x2]
