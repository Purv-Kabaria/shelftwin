"""Tiling of wide frames and merging tile detections (spec Section 3.4)."""
from __future__ import annotations

import numpy as np


def make_tiles(width: int, height: int, n_x: int = 2, overlap: float = 0.10) -> list[tuple[int, int, int, int]]:
    """Split horizontally into n_x full-height tiles with the given overlap."""
    if n_x <= 1:
        return [(0, 0, width, height)]
    tile_w = int(np.ceil(width / (n_x - (n_x - 1) * overlap)))
    step = (width - tile_w) / (n_x - 1)
    tiles = []
    for i in range(n_x):
        x1 = int(round(i * step))
        tiles.append((x1, 0, min(width, x1 + tile_w), height))
    return tiles


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.55) -> np.ndarray:
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return np.array(keep, dtype=int)


def weighted_merge(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight weighted box fusion: cluster by IoU around the top box, average coordinates."""
    order = np.argsort(-scores)
    used = np.zeros(len(boxes), dtype=bool)
    out_b, out_s = [], []
    for i in order:
        if used[i]:
            continue
        ious = iou_matrix(boxes[i:i + 1], boxes)[0]
        grp = np.where((ious >= iou_thr) & ~used)[0]
        used[grp] = True
        w = scores[grp]
        out_b.append((boxes[grp] * w[:, None]).sum(0) / w.sum())
        out_s.append(float(w.max()))
    if not out_b:
        return np.zeros((0, 4)), np.zeros(0)
    return np.array(out_b), np.array(out_s)


def merge_tile_detections(dets: list[dict], iou_thr: float = 0.55) -> list[dict]:
    """dets: [{bbox_xyxy, cls, conf}] in full-frame coords, from all tiles."""
    out = []
    for c in sorted({d["cls"] for d in dets}):
        ds = [d for d in dets if d["cls"] == c]
        b = np.array([d["bbox_xyxy"] for d in ds], dtype=float)
        s = np.array([d["conf"] for d in ds], dtype=float)
        mb, ms = weighted_merge(b, s, iou_thr)
        out += [dict(bbox_xyxy=mb[i].tolist(), cls=c, conf=float(ms[i])) for i in range(len(ms))]
    return out
