"""Shelf geometry: homography to shelf cm, slot assignment, coverage, drift (spec Section 3.5)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Slot:
    slot_id: str
    shelf: int
    x_cm: float
    width_cm: float
    y_cm: float          # top of the shelf row in shelf coordinates
    height_cm: float
    sku_id: str | None = None
    facings: int = 1

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x_cm, self.y_cm, self.x_cm + self.width_cm, self.y_cm + self.height_cm)


class BayGeometry:
    """Maps image pixels to the shelf plane of one bay (cm), via a homography H.

    corners_px: 4 image points (TL, TR, BR, BL) of the bay face.
    size_cm:    (width_cm, height_cm) of that face.
    """

    def __init__(self, corners_px, size_cm: tuple[float, float]):
        self.size_cm = size_cm
        W, Hc = size_cm
        src = np.asarray(corners_px, dtype=np.float32)
        dst = np.array([[0, 0], [W, 0], [W, Hc], [0, Hc]], dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def identity(cls, image_w: int, image_h: int, px_per_cm: float = 10.0) -> "BayGeometry":
        """Front-parallel camera: pixels scale linearly to cm."""
        return cls([[0, 0], [image_w, 0], [image_w, image_h], [0, image_h]],
                   (image_w / px_per_cm, image_h / px_per_cm))

    def to_cm(self, pts_px) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_px(self, pts_cm) -> np.ndarray:
        pts = np.asarray(pts_cm, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def box_to_cm(self, xyxy) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = xyxy
        c = self.to_cm([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())

    def slot_polygon_px(self, slot: Slot) -> np.ndarray:
        x1, y1, x2, y2 = slot.rect
        return self.to_px([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def assign_to_slots(boxes_cm: list[tuple], slots: list[Slot]) -> dict[str, list[int]]:
    """A pack belongs to the slot containing its bottom-centre; width overlap breaks ties."""
    out: dict[str, list[int]] = {s.slot_id: [] for s in slots}
    for i, (x1, y1, x2, y2) in enumerate(boxes_cm):
        cx, by = (x1 + x2) / 2, y2
        best, best_ov = None, 0.0
        for s in slots:
            sx1, sy1, sx2, sy2 = s.rect
            if not (sy1 - 0.15 * s.height_cm <= by <= sy2 + 0.15 * s.height_cm):
                continue
            inside = sx1 <= cx <= sx2
            ov = max(0.0, min(x2, sx2) - max(x1, sx1)) + (1e6 if inside else 0.0)
            if ov > best_ov:
                best, best_ov = s.slot_id, ov
        if best is not None:
            out[best].append(i)
    return out


def slot_coverage(boxes_cm: list[tuple], slot: Slot, res_cm: float = 0.5) -> float:
    """c_s = |union(boxes) ∩ slot| / |slot|, rasterised on the shelf plane."""
    sx1, sy1, sx2, sy2 = slot.rect
    nx = max(1, int(round((sx2 - sx1) / res_cm)))
    ny = max(1, int(round((sy2 - sy1) / res_cm)))
    grid = np.zeros((ny, nx), dtype=bool)
    for x1, y1, x2, y2 in boxes_cm:
        ix1 = int(np.clip((x1 - sx1) / res_cm, 0, nx))
        ix2 = int(np.clip((x2 - sx1) / res_cm, 0, nx))
        iy1 = int(np.clip((y1 - sy1) / res_cm, 0, ny))
        iy2 = int(np.clip((y2 - sy1) / res_cm, 0, ny))
        if ix2 > ix1 and iy2 > iy1:
            grid[iy1:iy2, ix1:ix2] = True
    return float(grid.mean())


def cluster_rows(bottoms_y: np.ndarray, min_gap: float) -> np.ndarray:
    """1-D clustering of box bottoms into shelf rows. Returns row index per box (top row = 0)."""
    if len(bottoms_y) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(bottoms_y)
    labels = np.zeros(len(bottoms_y), dtype=int)
    row = 0
    for a, b in zip(order[:-1], order[1:]):
        labels[a] = row
        if bottoms_y[b] - bottoms_y[a] > min_gap:
            row += 1
    labels[order[-1]] = row
    return labels


class DriftChecker:
    """ORB matching to a reference frame; large reprojection error pauses alerts."""

    def __init__(self, ref_bgr: np.ndarray, max_err_px: float = 8.0, n_features: int = 1500):
        self.orb = cv2.ORB_create(n_features)
        self.max_err_px = max_err_px
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def check(self, bgr: np.ndarray) -> dict:
        kp, des = self.orb.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)
        if des is None or self.ref_des is None:
            return dict(ok=False, err_px=float("inf"), matches=0)
        m = sorted(self.matcher.match(self.ref_des, des), key=lambda x: x.distance)[:300]
        if len(m) < 12:
            return dict(ok=False, err_px=float("inf"), matches=len(m))
        src = np.float32([self.ref_kp[x.queryIdx].pt for x in m])
        dst = np.float32([kp[x.trainIdx].pt for x in m])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return dict(ok=False, err_px=float("inf"), matches=len(m))
        # Drift = how far reference points move under the estimated camera motion
        err = float(np.median(np.linalg.norm(cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2) - src, axis=1)))
        return dict(ok=err <= self.max_err_px, err_px=err, matches=len(m))
