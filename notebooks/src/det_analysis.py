"""Detector error analysis and cheap eval-time experiments (no retraining).

* match predictions to ground truth (IoU >= 0.5, greedy by confidence)
* recall by pack size (relative to the image) and by shelf density (boxes per image)
* counting error per image (what shelf auditing actually needs)
* worst images gallery: GT green, true positives blue, false positives red, misses yellow
* sweeps: NMS IoU and max_det at evaluation time; full image vs 2-tile inference
* latency per image (batch 1, FP16) -> speed/accuracy trade-off for the edge
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from kaggle_common import WORK, log


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match(pred: np.ndarray, conf: np.ndarray, gt: np.ndarray, thr: float = 0.5):
    """Greedy matching by confidence. Returns (tp flag per pred, matched flag per gt)."""
    tp = np.zeros(len(pred), bool)
    used = np.zeros(len(gt), bool)
    if len(pred) and len(gt):
        ious = iou_matrix(pred, gt)
        for i in np.argsort(-conf):
            row = np.where(used, -1.0, ious[i])
            j = int(np.argmax(row))
            if row[j] >= thr:
                tp[i], used[j] = True, True
    return tp, used


def read_labels(path: Path, w: int, h: int, cls: int | None = 0) -> np.ndarray:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            q = line.split()
            if len(q) >= 5 and (cls is None or int(q[0]) == cls):
                cx, cy, bw, bh = (float(v) for v in q[1:5])
                out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, np.float32).reshape(-1, 4)


def collect(weights, img_dir: Path, lbl_dir: Path, imgsz: int, conf: float = 0.25, iou: float = 0.6,
            max_det: int = 1000, cls: int = 0, n: int | None = None, batch: int = 16) -> list[dict]:
    """Predict on images and match to labels. One dict per image."""
    import cv2
    import torch
    from ultralytics import YOLO
    model = YOLO(str(weights))
    dev = 0 if torch.cuda.is_available() else "cpu"
    ims = sorted(img_dir.glob("*.jpg"))[:n]
    recs = []
    for i in range(0, len(ims), batch):
        chunk = ims[i:i + batch]
        res = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                            device=dev, half=dev != "cpu", verbose=False)
        for p, r in zip(chunk, res):
            h, w = r.orig_shape
            gt = read_labels(lbl_dir / f"{p.stem}.txt", w, h, cls)
            if r.boxes is not None and len(r.boxes):
                k = r.boxes.cls.cpu().numpy().astype(int) == cls
                pb, pc = r.boxes.xyxy.cpu().numpy()[k], r.boxes.conf.cpu().numpy()[k]
            else:
                pb, pc = np.zeros((0, 4)), np.zeros(0)
            tp, matched = match(pb, pc, gt)
            gsz = np.sqrt(np.clip((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]), 0, None) / (w * h)) if len(gt) else np.zeros(0)
            recs.append(dict(image=str(p), w=w, h=h, gt=gt, pred=pb, conf=pc, tp=tp, matched=matched, gt_rel_size=gsz))
    return recs


def summarize(recs: list[dict]) -> dict:
    n_gt = sum(len(r["gt"]) for r in recs)
    n_pred = sum(len(r["pred"]) for r in recs)
    n_tp = sum(int(r["tp"].sum()) for r in recs)
    sizes = np.concatenate([r["gt_rel_size"] for r in recs]) if recs else np.zeros(0)
    hit = np.concatenate([r["matched"] for r in recs]) if recs else np.zeros(0, bool)
    out = dict(images=len(recs), gt=n_gt, pred=n_pred, precision=n_tp / max(1, n_pred), recall=n_tp / max(1, n_gt))
    if len(sizes):
        qs = np.quantile(sizes, [0, .2, .4, .6, .8, 1.0])
        out["recall_by_size_quintile"] = [
            dict(rel_size=f"{qs[i]:.3f}-{qs[i + 1]:.3f}",
                 recall=float(hit[(sizes >= qs[i]) & (sizes <= qs[i + 1])].mean())) for i in range(5)]
    dens = np.array([len(r["gt"]) for r in recs])
    bins = [0, 50, 100, 150, 250, 10_000]
    out["recall_by_density"] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r, d in zip(recs, dens) if lo <= d < hi]
        g = sum(len(r["gt"]) for r in sel)
        if g:
            out["recall_by_density"].append(dict(boxes_per_image=f"{lo}-{hi}", images=len(sel),
                                                 recall=sum(int(r["matched"].sum()) for r in sel) / g))
    cnt_err = np.array([len(r["pred"]) - len(r["gt"]) for r in recs], float)
    out["count_error"] = dict(mae=float(np.abs(cnt_err).mean()), bias=float(cnt_err.mean()),
                              mape=float(np.mean(np.abs(cnt_err) / np.maximum(1, dens))))
    # false positives: duplicates (overlap a matched GT) vs background
    dup = bg = 0
    for r in recs:
        fp = r["pred"][~r["tp"]]
        if len(fp) and len(r["gt"]):
            m = iou_matrix(fp, r["gt"]).max(1)
            dup += int((m >= 0.3).sum()); bg += int((m < 0.3).sum())
        else:
            bg += len(fp)
    out["false_positive_types"] = dict(duplicate_or_misaligned=dup, background=bg)
    return out


def worst_gallery(recs: list[dict], path: Path, k: int = 4):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def f1(r):
        tp = r["tp"].sum(); p = tp / max(1, len(r["pred"])); q = tp / max(1, len(r["gt"]))
        return 2 * p * q / max(1e-9, p + q)
    worst = sorted(recs, key=f1)[:k]
    fig, axs = plt.subplots(1, k, figsize=(5 * k, 6))
    for ax, r in zip(np.atleast_1d(axs), worst):
        img = cv2.imread(r["image"])[:, :, ::-1].copy()
        for b in r["gt"][r["matched"]].astype(int):
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 200, 0), 1)
        for b in r["gt"][~r["matched"]].astype(int):
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (255, 220, 0), 3)
        for b in r["pred"][~r["tp"]].astype(int):
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (255, 0, 0), 2)
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"F1 {f1(r):.2f} | gt {len(r['gt'])} pred {len(r['pred'])}", fontsize=9)
    fig.suptitle("worst images: yellow = missed pack, red = false positive, green = found")
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def sweep_val(weights, data_yaml: Path, imgsz: int, split: str = "test", ious=(0.5, 0.6, 0.7),
              max_dets=(300, 1000)) -> list[dict]:
    """Eval-time knobs need no retraining: NMS IoU (dense, touching packs) and max_det."""
    import torch
    from ultralytics import YOLO
    m = YOLO(str(weights))
    dev = 0 if torch.cuda.is_available() else "cpu"
    rows = []
    for md in max_dets:
        for iou in ious:
            r = m.val(data=str(data_yaml), split=split, imgsz=imgsz, iou=iou, max_det=md, conf=0.001, device=dev,
                      half=dev != "cpu", plots=False, verbose=False, project=str(WORK / "runs"), name="sweep",
                      exist_ok=True)
            rows.append(dict(nms_iou=iou, max_det=md, mAP50=float(r.box.map50), mAP50_95=float(r.box.map)))
            log("sweep", rows[-1])
    return rows


def tiled_vs_full(weights, img_dir: Path, lbl_dir: Path, imgsz: int, n: int = 150, overlap: float = 0.1,
                  conf: float = 0.25) -> dict:
    """Deployment splits a frame into 2 tiles along its long side. Does that beat one full pass?"""
    import cv2
    import torch
    import torchvision
    from ultralytics import YOLO
    model = YOLO(str(weights))
    dev = 0 if torch.cuda.is_available() else "cpu"
    out = {"full": [0, 0, 0], "tiled": [0, 0, 0]}  # tp, pred, gt
    for p in sorted(img_dir.glob("*.jpg"))[:n]:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = read_labels(lbl_dir / f"{p.stem}.txt", w, h, 0)
        r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.6, max_det=1000, device=dev, verbose=False)[0]
        pb, pc = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        tp, _ = match(pb, pc, gt)
        out["full"] = [out["full"][0] + tp.sum(), out["full"][1] + len(pb), out["full"][2] + len(gt)]
        L = max(h, w)
        tile = int(np.ceil(L / (2 - overlap)))
        boxes, confs = [], []
        for off in (0, L - tile):
            crop = img[:, off:off + tile] if w >= h else img[off:off + tile]
            rr = model.predict(crop, imgsz=imgsz, conf=conf, iou=0.6, max_det=1000, device=dev, verbose=False)[0]
            b = rr.boxes.xyxy.cpu().numpy().copy()
            if w >= h:
                b[:, [0, 2]] += off
            else:
                b[:, [1, 3]] += off
            boxes.append(b); confs.append(rr.boxes.conf.cpu().numpy())
        b, c = np.concatenate(boxes), np.concatenate(confs)
        if len(b):
            keep = torchvision.ops.nms(torch.from_numpy(b).float(), torch.from_numpy(c).float(), 0.6).numpy()
            b, c = b[keep], c[keep]
        tp, _ = match(b, c, gt)
        out["tiled"] = [out["tiled"][0] + tp.sum(), out["tiled"][1] + len(b), out["tiled"][2] + len(gt)]
    res = {k: dict(precision=float(v[0] / max(1, v[1])), recall=float(v[0] / max(1, v[2]))) for k, v in out.items()}
    log("tiled vs full:", res)
    return res


def latency_ms(weights, imgsz: int, n: int = 50, half: bool = True) -> float:
    """Median batch-1 latency on this GPU (relative speed between configurations)."""
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        return float("nan")
    m = YOLO(str(weights))
    x = (np.random.default_rng(0).random((imgsz, imgsz, 3)) * 255).astype(np.uint8)
    for _ in range(5):
        m.predict(x, imgsz=imgsz, device=0, half=half, verbose=False, max_det=1000)
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        m.predict(x, imgsz=imgsz, device=0, half=half, verbose=False, max_det=1000)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
