"""Build a 2-class (pack, gap) dataset from a prepared SKU-110K YOLO dataset.

Gap labels come from two sources:
  * Real gaps (all splits): in a shelf row (box bottoms clustered), a horizontal space
    between neighbouring packs wider than ~0.6 x the row's median pack width, not
    covered by any other pack. This is a geometric heuristic on real photos.
  * Synthetic gaps (TRAIN split only, spec 5.2.7): erase a run of 1-4 neighbouring packs
    and fill with inpainting, a dark "shelf back" texture, or a real gap patch.

Every image is then cut into square crops resized to 640 px, which matches the
deployment tiling (a 1080p frame split into two ~1080 px tiles, each run at 640).
"""
from __future__ import annotations

import os
import zlib
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from kaggle_common import log

PACK, GAP = 0, 1
cv2.setNumThreads(1)


# ----------------------------------------------------------------------------- labels IO
def read_yolo(lbl: Path, w: int, h: int) -> np.ndarray:
    """-> (N,4) xyxy pixels, class 0 only."""
    rows = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def write_yolo(path: Path, boxes: np.ndarray, classes: np.ndarray, w: int, h: int):
    lines = []
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        lines.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ----------------------------------------------------------------------------- geometry
def cluster_rows(boxes: np.ndarray) -> np.ndarray:
    """Row index per box from box bottoms (gap-based 1-D clustering)."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    bottoms = boxes[:, 3]
    med_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    order = np.argsort(bottoms)
    labels = np.zeros(len(boxes), int)
    row = 0
    for a, b in zip(order[:-1], order[1:]):
        labels[a] = row
        if bottoms[b] - bottoms[a] > 0.5 * med_h:
            row += 1
    labels[order[-1]] = row
    return labels


def covered_fraction(region, boxes: np.ndarray) -> float:
    """Share of `region` covered by the union of `boxes` (coarse raster)."""
    x1, y1, x2, y2 = region
    W, H = max(1, int(x2 - x1)), max(1, int(y2 - y1))
    s = max(1.0, max(W, H) / 64)
    gw, gh = max(1, int(W / s)), max(1, int(H / s))
    grid = np.zeros((gh, gw), bool)
    for bx1, by1, bx2, by2 in boxes:
        ix1 = int(np.clip((bx1 - x1) / s, 0, gw)); ix2 = int(np.clip((bx2 - x1) / s, 0, gw))
        iy1 = int(np.clip((by1 - y1) / s, 0, gh)); iy2 = int(np.clip((by2 - y1) / s, 0, gh))
        if ix2 > ix1 and iy2 > iy1:
            grid[iy1:iy2, ix1:ix2] = True
    return float(grid.mean())


def real_gaps(boxes: np.ndarray, rows: np.ndarray, min_rel: float = 0.6, min_px: float = 12) -> list[list[float]]:
    gaps = []
    for r in np.unique(rows):
        rb = boxes[rows == r]
        if len(rb) < 3:
            continue
        rb = rb[np.argsort(rb[:, 0])]
        med_w = float(np.median(rb[:, 2] - rb[:, 0]))
        for a, b in zip(rb[:-1], rb[1:]):
            space = b[0] - a[2]
            if space > max(min_px, min_rel * med_w):
                g = [float(a[2]), float(min(a[1], b[1])), float(b[0]), float(max(a[3], b[3]))]
                if covered_fraction(g, boxes) < 0.15:
                    gaps.append(g)
    return gaps


# ----------------------------------------------------------------------------- synthesis
def _shelf_back_fill(img, x1, y1, x2, y2, rng):
    h, w = y2 - y1, x2 - x1
    base = np.percentile(img.reshape(-1, 3), 8, axis=0) * rng.uniform(0.5, 1.1)
    grad = np.linspace(rng.uniform(0.55, 0.85), 1.0, h)[:, None, None]  # darker top (shadow)
    patch = base[None, None, :] * grad + rng.normal(0, rng.uniform(2, 7), (h, w, 3))
    patch = cv2.GaussianBlur(np.clip(patch, 0, 255).astype(np.uint8), (5, 5), 0)
    img[y1:y2, x1:x2] = patch


def synth_gaps(img: np.ndarray, boxes: np.ndarray, rows: np.ndarray, rng: random.Random,
               real_gap_patches: list[np.ndarray], max_gaps: int = 2, fill_probs=(0.35, 0.45, 0.20)):
    """Erase runs of packs. Returns (img, kept_pack_boxes, new_gap_boxes).
    fill_probs: probabilities of (inpaint, dark shelf-back texture, real gap patch)."""
    H, W = img.shape[:2]
    keep = np.ones(len(boxes), bool)
    gaps = []
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    row_ids = [r for r in np.unique(rows) if (rows == r).sum() >= 4]
    rng.shuffle(row_ids)
    for r in row_ids[:rng.randint(1, max_gaps)]:
        idx = np.where((rows == r) & keep)[0]
        if len(idx) < 4:
            continue
        idx = idx[np.argsort(boxes[idx, 0])]
        k = rng.randint(1, min(4, len(idx) - 2))
        s = rng.randint(1, len(idx) - k - 1)  # never the row ends: a gap sits between packs
        run = idx[s:s + k]
        x1, y1 = boxes[run, 0].min(), boxes[run, 1].min()
        x2, y2 = boxes[run, 2].max(), boxes[run, 3].max()
        x1, y1 = int(max(0, x1 - 1)), int(max(0, y1 - 1))
        x2, y2 = int(min(W, x2 + 1)), int(min(H, y2 + 1))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        mode = rng.random()
        p_inp, p_back = fill_probs[0], fill_probs[0] + fill_probs[1]
        if mode < p_inp:
            m = np.zeros((H, W), np.uint8)
            m[y1:y2, x1:x2] = 255
            # inpaint only a local window: fast and uses nearby shelf texture
            px, py = (x2 - x1), (y2 - y1)
            wx1, wy1 = max(0, x1 - px), max(0, y1 - py // 2)
            wx2, wy2 = min(W, x2 + px), min(H, y2 + py // 2)
            win = cv2.inpaint(img[wy1:wy2, wx1:wx2], m[wy1:wy2, wx1:wx2], 5, cv2.INPAINT_TELEA)
            img[wy1:wy2, wx1:wx2] = win
        elif mode < p_back or not real_gap_patches:
            _shelf_back_fill(img, x1, y1, x2, y2, nrng)
        else:
            patch = real_gap_patches[rng.randrange(len(real_gap_patches))]
            img[y1:y2, x1:x2] = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        keep[run] = False
        # Drop other packs that were mostly erased
        for j in np.where(keep)[0]:
            bx1, by1, bx2, by2 = boxes[j]
            inter = max(0, min(bx2, x2) - max(bx1, x1)) * max(0, min(by2, y2) - max(by1, y1))
            if inter > 0.3 * (bx2 - bx1) * (by2 - by1):
                keep[j] = False
        gaps.append([x1, y1, x2, y2])
    return img, boxes[keep], gaps


# ----------------------------------------------------------------------------- crops
def random_crops(H: int, W: int, n: int, rng: random.Random, lo=0.55, hi=0.9) -> list[tuple[int, int, int, int]]:
    out = []
    short = min(H, W)
    for _ in range(n):
        s = int(short * rng.uniform(lo, hi))
        x = rng.randint(0, W - s)
        y = rng.randint(0, H - s)
        out.append((x, y, x + s, y + s))
    return out


def clip_boxes(boxes: np.ndarray, crop, min_vis: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, bool)
    cx1, cy1, cx2, cy2 = crop
    b = boxes.copy()
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], cx1, cx2)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], cy1, cy2)
    vis = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / np.maximum(area, 1e-6)
    keep = (vis >= min_vis) & (b[:, 2] - b[:, 0] >= 3) & (b[:, 3] - b[:, 1] >= 3)
    b = b[keep]
    b[:, [0, 2]] -= cx1
    b[:, [1, 3]] -= cy1
    return b, keep


def _process(args):
    img_path, lbl_path, out_dir, split, seed, n_crops, out_size, p_synth, fill_probs = args
    try:
        rng = random.Random(seed)
        img = cv2.imread(img_path)
        if img is None:
            return dict(status="bad")
        H, W = img.shape[:2]
        packs = read_yolo(Path(lbl_path), W, H)
        if len(packs) < 3:
            return dict(status="few")
        rows = cluster_rows(packs)
        gaps = real_gaps(packs, rows)
        n_real = len(gaps)
        n_syn = 0
        if split == "train" and rng.random() < p_synth:
            patches = []
            for g in gaps:
                x1, y1, x2, y2 = (int(v) for v in g)
                if x2 - x1 > 4 and y2 - y1 > 4:
                    patches.append(img[y1:y2, x1:x2].copy())
            img, packs, syn = synth_gaps(img, packs, rows, rng, patches, fill_probs=fill_probs)
            gaps += syn
            n_syn = len(syn)
        gaps_a = np.array(gaps, np.float32).reshape(-1, 4)
        stem = Path(img_path).stem
        crops = random_crops(H, W, n_crops, rng) if split == "train" else [_center_crop(H, W)]
        for ci, crop in enumerate(crops):
            pb, _ = clip_boxes(packs, crop)
            gb, _ = clip_boxes(gaps_a, crop, min_vis=0.6)
            s = crop[2] - crop[0]
            sc = out_size / s
            im = cv2.resize(img[crop[1]:crop[3], crop[0]:crop[2]], (out_size, out_size),
                            interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
            name = f"{stem}_c{ci}"
            cv2.imwrite(str(Path(out_dir) / "images" / split / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            boxes = np.concatenate([pb, gb]) * sc
            cls = np.concatenate([np.full(len(pb), PACK), np.full(len(gb), GAP)])
            write_yolo(Path(out_dir) / "labels" / split / f"{name}.txt", boxes, cls, out_size, out_size)
        return dict(status="ok", real=n_real, syn=n_syn)
    except Exception as e:
        return dict(status="bad", err=str(e))


def _center_crop(H, W):
    """Val/test: one deterministic crop covering most of the short side."""
    s = int(min(H, W) * 0.9)
    x, y = (W - s) // 2, (H - s) // 2
    return (x, y, x + s, y + s)


def build_pack_gap(src_root: Path, out: Path, limits: dict | None = None, n_crops: int = 2,
                   out_size: int = 640, p_synth: float = 0.6, workers: int | None = None, seed: int = 0,
                   fill_probs=(0.35, 0.45, 0.20)) -> Path:
    limits = limits or {}
    if (out / "data.yaml").exists():
        log("Reusing", out)
        return out / "data.yaml"
    jobs = []
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        ims = sorted((src_root / "images" / split).glob("*.jpg"))
        if limits.get(split):
            random.Random(seed).shuffle(ims)
            ims = ims[:limits[split]]
        for i, im in enumerate(ims):
            jobs.append((str(im), str(src_root / "labels" / split / f"{im.stem}.txt"), str(out), split,
                         seed * 1_000_003 + zlib.crc32(im.name.encode()) % 1_000_003, n_crops, out_size, p_synth, tuple(fill_probs)))
    stats = dict(ok=0, bad=0, few=0, real=0, syn=0)
    with Pool(workers or max(1, os.cpu_count() or 2)) as pool:
        for i, r in enumerate(pool.imap_unordered(_process, jobs, chunksize=4)):
            stats[r["status"]] += 1
            stats["real"] += r.get("real", 0)
            stats["syn"] += r.get("syn", 0)
            if (i + 1) % 1000 == 0:
                log(f"  {i + 1}/{len(jobs)} {stats}")
    log("pack+gap build:", stats)
    if stats["ok"] == 0:
        raise RuntimeError("No pack+gap images were produced.")
    has_test = any((out / "images" / "test").glob("*.jpg"))
    yaml = [f"path: {out}", "train: images/train", "val: images/val"] + (["test: images/test"] if has_test else [])
    yaml += ["names:", "  0: pack", "  1: gap"]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    return out / "data.yaml"


# ----------------------------------------------------------------------------- metric
def gap_recall_at_precision(weights: Path, data_root: Path, split: str = "test", target_p: float = 0.9,
                            iou_thr: float = 0.5, imgsz: int = 640, max_images: int = 2000) -> dict:
    """Gap recall at a fixed precision (spec 6.3), by sweeping the confidence threshold."""
    import torch
    from ultralytics import YOLO
    model = YOLO(str(weights))
    ims = sorted((data_root / "images" / split).glob("*.jpg"))[:max_images]
    if not ims:
        return {}
    dev = 0 if torch.cuda.is_available() else "cpu"
    scores, tps, n_gt = [], [], 0
    for i in range(0, len(ims), 32):
        batch = ims[i:i + 32]
        res = model.predict([str(p) for p in batch], imgsz=imgsz, conf=0.01, iou=0.6, max_det=1000,
                            device=dev, verbose=False, half=dev != "cpu")
        for p, r in zip(batch, res):
            h, w = r.orig_shape
            gt = []
            lbl = data_root / "labels" / split / f"{p.stem}.txt"
            for line in (lbl.read_text().splitlines() if lbl.exists() else []):
                q = line.split()
                if len(q) >= 5 and int(q[0]) == GAP:
                    cx, cy, bw, bh = (float(v) for v in q[1:5])
                    gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            gt = np.array(gt).reshape(-1, 4)
            n_gt += len(gt)
            if r.boxes is None or len(r.boxes) == 0:
                continue
            cls = r.boxes.cls.cpu().numpy().astype(int)
            pb = r.boxes.xyxy.cpu().numpy()[cls == GAP]
            pc = r.boxes.conf.cpu().numpy()[cls == GAP]
            used = np.zeros(len(gt), bool)
            for j in np.argsort(-pc):
                tp = False
                if len(gt):
                    ious = _iou(pb[j], gt)
                    ious[used] = 0
                    k = int(np.argmax(ious))
                    if ious[k] >= iou_thr:
                        used[k] = True
                        tp = True
                scores.append(float(pc[j]))
                tps.append(tp)
    if not scores or n_gt == 0:
        return dict(n_gt=n_gt, recall_at_p=0.0, conf_at_p=None)
    order = np.argsort(-np.array(scores))
    tp = np.cumsum(np.array(tps)[order])
    fp = np.cumsum(~np.array(tps)[order])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_gt
    ok = np.where(prec >= target_p)[0]
    if not len(ok):
        return dict(n_gt=n_gt, recall_at_p=0.0, conf_at_p=None, max_precision=float(prec.max()))
    j = ok[np.argmax(rec[ok])]
    return dict(n_gt=int(n_gt), target_precision=target_p, recall_at_p=float(rec[j]),
                conf_at_p=float(np.array(scores)[order][j]), images=len(ims))


def _iou(b, g):
    x1 = np.maximum(b[0], g[:, 0]); y1 = np.maximum(b[1], g[:, 1])
    x2 = np.minimum(b[2], g[:, 2]); y2 = np.minimum(b[3], g[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1]); ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (a + ag - inter + 1e-9)
