"""Frame quality gate (spec Section 3.2): blur, glare, exposure -> q in [0, 1]."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityConfig:
    blur_threshold: float | None = None   # None: calibrate (0.5 x median of good frames)
    glare_v: int = 245
    glare_max_frac: float = 0.08
    lum_min: float = 40.0
    lum_max: float = 220.0


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def glare_fraction(bgr: np.ndarray, v_thresh: int = 245) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float((hsv[..., 2] > v_thresh).mean())


class QualityGate:
    def __init__(self, cfg: QualityConfig | None = None):
        self.cfg = cfg or QualityConfig()
        self._calib: list[float] = []

    def calibrate(self, frames: list[np.ndarray]) -> float:
        scores = [blur_score(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
        self.cfg.blur_threshold = 0.5 * float(np.median(scores))
        return self.cfg.blur_threshold

    def score(self, bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> dict:
        """Returns dict(q, blur, glare, lum). roi = (x1, y1, x2, y2) for per-bay glare."""
        img = bgr if roi is None else bgr[roi[1]:roi[3], roi[0]:roi[2]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b = blur_score(gray)
        g = glare_fraction(img, self.cfg.glare_v)
        lum = float(gray.mean())

        thr = self.cfg.blur_threshold
        if thr is None:  # self-calibrate from the first frames
            self._calib.append(b)
            thr = 0.5 * float(np.median(self._calib))
        # Each term is 0.5 exactly at its rejection threshold (the engine's quality floor).
        q_blur = float(np.clip((b - 0.5 * thr) / thr, 0.0, 1.0)) if thr > 0 else 1.0
        q_glare = float(np.clip(1.5 - g / self.cfg.glare_max_frac, 0.0, 1.0))
        if self.cfg.lum_min <= lum <= self.cfg.lum_max:
            q_exp = 1.0
        else:
            dist = (self.cfg.lum_min - lum) if lum < self.cfg.lum_min else (lum - self.cfg.lum_max)
            q_exp = float(np.clip(1.0 - dist / 40.0, 0.0, 1.0))
        q = q_blur * q_glare * q_exp
        return dict(q=q, blur=b, glare=g, lum=lum, q_blur=q_blur, q_glare=q_glare, q_exp=q_exp)
