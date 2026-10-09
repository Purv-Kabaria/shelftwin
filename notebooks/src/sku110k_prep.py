"""Find SKU-110K under /kaggle/input (any layout) and build a resized YOLO dataset.

Supported sources, tried in order:
  1. Original CSV annotations (annotations_{train,val,test}.csv, no header:
     image,x1,y1,x2,y2,class,image_width,image_height) plus an images folder anywhere.
  2. A YOLO-format copy (images/ + labels/ folders with train/val/test in the path).
  3. Streaming download of the official archive (13.6 GB) that resizes on the fly,
     so the archive itself is never stored.

Images are resized so the long side is `max_side` (SKU-110K photos are ~3000 px and
decoding them is the data-loader bottleneck on Kaggle's 4 CPUs).
"""
from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from kaggle_common import IMG_EXT, INPUT_ROOT, TMP, disk_free_gb, find_files, log, print_input_tree, walk_limited

OFFICIAL_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
SPLITS = ("train", "val", "test")
cv2.setNumThreads(1)


# ----------------------------------------------------------------------------- CSV parsing
def _is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def read_sku_csv(path: Path) -> dict[str, dict]:
    """Returns {image_name: {'w': W, 'h': H, 'boxes': [[x1,y1,x2,y2], ...]}}. Tolerates a header row."""
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 8 or not _is_number(row[1]):
                continue  # header or malformed
            name = os.path.basename(row[0].strip())
            x1, y1, x2, y2 = (float(v) for v in row[1:5])
            w, h = float(row[6]), float(row[7])
            d = out.setdefault(name, {"w": w, "h": h, "boxes": []})
            d["boxes"].append([x1, y1, x2, y2])
    return out


def boxes_to_yolo(boxes, w: float, h: float, min_px: float = 2.0) -> list[str]:
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0.0, min(w, x1)), max(0.0, min(w, x2))))
        y1, y2 = sorted((max(0.0, min(h, y1)), max(0.0, min(h, y2))))
        if x2 - x1 < min_px or y2 - y1 < min_px:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


# ----------------------------------------------------------------------------- image IO
def decode_resized(data_or_path, max_side: int, expect_wh: tuple[float, float] | None = None):
    """Decode (fast reduced decode when possible) and resize so the long side <= max_side.

    Returns (image, (orig_w, orig_h)) or (None, reason). Orientation: we try the raw
    pixel order first (EXIF ignored); if the CSV size matches the EXIF-rotated order
    instead, we use that. Boxes in the CSV must match the decoded orientation.
    """
    if isinstance(data_or_path, (str, Path)):
        buf = np.fromfile(str(data_or_path), dtype=np.uint8)
    else:
        buf = np.frombuffer(data_or_path, dtype=np.uint8)
    if buf.size == 0:
        return None, "empty file"

    reduce_flag = cv2.IMREAD_COLOR
    if expect_wh is not None:
        long_side = max(expect_wh)
        if long_side >= 4 * max_side:
            reduce_flag = cv2.IMREAD_REDUCED_COLOR_4
        elif long_side >= 2 * max_side:
            reduce_flag = cv2.IMREAD_REDUCED_COLOR_2
    factor = {cv2.IMREAD_COLOR: 1, cv2.IMREAD_REDUCED_COLOR_2: 2, cv2.IMREAD_REDUCED_COLOR_4: 4}[reduce_flag]

    img = cv2.imdecode(buf, reduce_flag | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return None, "decode failed"
    h, w = img.shape[:2]
    orig_w, orig_h = w * factor, h * factor
    if expect_wh is not None:
        ew, eh = expect_wh
        def close(a, b):
            return abs(a - b) <= max(4 * factor, 0.01 * b)
        if not (close(orig_w, ew) and close(orig_h, eh)):
            alt = cv2.imdecode(buf, reduce_flag)  # EXIF-applied orientation
            if alt is not None and close(alt.shape[1] * factor, ew) and close(alt.shape[0] * factor, eh):
                img = alt
                h, w = img.shape[:2]
            elif abs((w / h) - (ew / eh)) > 0.02:
                return None, f"size mismatch {orig_w}x{orig_h} vs csv {ew:.0f}x{eh:.0f}"
            orig_w, orig_h = ew, eh  # same aspect ratio: trust the CSV scale
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return img, (orig_w, orig_h)


def _job(args):
    src, dst_img, dst_lbl, boxes, wh, max_side = args
    try:
        if Path(dst_img).exists() and Path(dst_lbl).exists():
            return "skip"
        hint = wh
        if hint is None:
            # YOLO-format source: read the size from the JPEG header (cheap) so decode_resized can use
            # libjpeg's reduced-resolution decode (2-4x faster for ~3000 px SKU-110K photos)
            try:
                from PIL import Image
                with Image.open(src) as im:
                    hint = im.size
            except Exception:
                hint = None
        img, info = decode_resized(src, max_side, hint)
        if img is None:
            return f"bad:{Path(src).name}:{info}"
        w, h = wh if wh else info
        lines = boxes_to_yolo(boxes, w, h) if boxes is not None else None
        if lines is not None and not lines:
            return f"bad:{Path(src).name}:no valid boxes"
        cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if lines is not None:
            Path(dst_lbl).write_text("\n".join(lines) + "\n")
        return "ok"
    except Exception as e:  # never let one image kill the pool
        return f"bad:{Path(src).name}:{e}"


# ----------------------------------------------------------------------------- discovery
def index_images(root: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for dp, _, fns in walk_limited(root, 8):
        for f in fns:
            if Path(f).suffix.lower() in IMG_EXT:
                idx.setdefault(f, Path(dp) / f)
    return idx


def find_csv_source(root: Path = INPUT_ROOT):
    csvs = find_files(lambda f: f.lower().startswith("annotations_") and f.lower().endswith(".csv"), root)
    by_split = {}
    for c in csvs:
        for s in SPLITS:
            if c.stem.lower() == f"annotations_{s}":
                by_split.setdefault(s, c)
    if "train" not in by_split:
        return None
    # The dataset root is the attached input folder that holds the CSVs
    rel = by_split["train"].relative_to(root)
    ds_root = root / rel.parts[0] if len(rel.parts) > 1 else root
    return by_split, ds_root


def find_yolo_source(root: Path = INPUT_ROOT):
    """Find (images_dir, labels_dir) pairs per split in an existing YOLO layout."""
    pairs, counts = {}, {}
    for dp, dns, fns in walk_limited(root, 6):
        p = Path(dp)
        parts = [x.lower() for x in p.parts]
        if "labels" not in parts:
            continue
        n_txt = sum(f.endswith(".txt") for f in fns)
        if n_txt == 0:
            continue
        i = len(parts) - 1 - parts[::-1].index("labels")
        # split folder after (labels/train) or before (train/labels) the 'labels' folder; 'valid' counts as val
        near = parts[i + 1:] + parts[max(0, i - 1):i]
        split = next((s for x in near for s in SPLITS if x == s or x.startswith(s)), None)
        if split is None:
            continue
        img_dir = Path(*p.parts[:i], "images" if p.parts[i] == "labels" else "Images", *p.parts[i + 1:])
        if img_dir.exists() and n_txt > counts.get(split, 0):
            pairs[split], counts[split] = (img_dir, p), n_txt
    return pairs if counts.get("train", 0) >= 20 else None


# ----------------------------------------------------------------------------- builders
def _prepare_out(out: Path):
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)


def _run_jobs(jobs, workers):
    stats = defaultdict(int)
    bad = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            key = r.split(":")[0]
            stats[key] += 1
            if key == "bad" and len(bad) < 20:
                bad.append(r)
            if (i + 1) % 1000 == 0:
                log(f"  {i + 1}/{len(jobs)} {dict(stats)}")
    return dict(stats), bad


def build_from_csv(by_split, ds_root, out: Path, max_side: int, limits: dict, workers: int, seed=0):
    log("Indexing images under", ds_root)
    idx = index_images(ds_root)
    log(f"  {len(idx)} image files indexed")
    _prepare_out(out)
    jobs, missing = [], 0
    for s, csv_path in by_split.items():
        ann = read_sku_csv(csv_path)
        names = sorted(ann)
        if limits.get(s):
            random.Random(seed).shuffle(names)
            names = sorted(names[:limits[s]])
        for n in names:
            src = idx.get(n)
            if src is None:
                missing += 1
                continue
            d = ann[n]
            stem = Path(n).stem
            jobs.append((str(src), str(out / "images" / s / f"{stem}.jpg"), str(out / "labels" / s / f"{stem}.txt"),
                         d["boxes"], (d["w"], d["h"]), max_side))
        log(f"  {s}: {len(names)} annotated images")
    if missing:
        log(f"  WARNING: {missing} annotated images not found on disk (skipped)")
    if not jobs:
        raise RuntimeError("CSV annotations found but no matching images. Check the attached dataset.")
    return _run_jobs(jobs, workers)


def build_from_yolo(pairs, out: Path, max_side: int, limits: dict, workers: int, seed=0):
    _prepare_out(out)
    jobs = []
    for s, (img_dir, lbl_dir) in pairs.items():
        ims = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        if limits.get(s):
            random.Random(seed).shuffle(ims)
            ims = ims[:limits[s]]
        for im in ims:
            lbl = lbl_dir / f"{im.stem}.txt"
            if not lbl.exists():
                continue
            dst_lbl = out / "labels" / s / f"{im.stem}.txt"
            # YOLO labels are normalised, so they survive resizing; force class 0 (pack)
            lines = [("0 " + " ".join(l.split()[1:5])) for l in lbl.read_text().splitlines() if len(l.split()) >= 5]
            if not lines:
                continue
            dst_lbl.write_text("\n".join(lines) + "\n")
            jobs.append((str(im), str(out / "images" / s / f"{im.stem}.jpg"), str(dst_lbl), None, None, max_side))
    if "val" not in pairs:  # carve a val split from train
        log("  no val split found: moving 5% of train to val")
    return _run_jobs(jobs, workers)


def build_from_stream(out: Path, max_side: int, limits: dict, url: str = OFFICIAL_URL, threads: int = 4):
    """Stream the official tar.gz, resize every image on the fly, keep only small files."""
    import tarfile
    import urllib.request
    from concurrent.futures import FIRST_COMPLETED, wait
    raw_dir = TMP / "sku_stream"
    (raw_dir / "images").mkdir(parents=True, exist_ok=True)
    (raw_dir / "annotations").mkdir(parents=True, exist_ok=True)

    def work(name, data):
        img, _ = decode_resized(data, max_side, None)
        if img is not None:
            cv2.imwrite(str(raw_dir / "images" / (Path(name).stem + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    for attempt in range(2):
        try:
            log(f"Streaming {url} (attempt {attempt + 1}); this takes ~20-40 min")
            with urllib.request.urlopen(url, timeout=120) as resp, \
                    tarfile.open(fileobj=resp, mode="r|gz") as tar, \
                    ThreadPoolExecutor(threads) as ex:
                futs, n = [], 0
                for m in tar:
                    if not m.isfile():
                        continue
                    base = os.path.basename(m.name)
                    if base.lower().endswith(".csv"):
                        (raw_dir / "annotations" / base).write_bytes(tar.extractfile(m).read())
                    elif Path(base).suffix.lower() in IMG_EXT:
                        if (raw_dir / "images" / (Path(base).stem + ".jpg")).exists():
                            continue
                        futs.append(ex.submit(work, base, tar.extractfile(m).read()))
                        n += 1
                        # Backpressure: never hold more than ~32 undecoded images (~100 MB) in RAM
                        while len(futs) > 32:
                            _, pending = wait(futs, return_when=FIRST_COMPLETED)
                            futs = list(pending)
                        if n % 1000 == 0:
                            log(f"  streamed {n} images, disk free {disk_free_gb(TMP):.1f} GB")
            break
        except Exception as e:
            log("stream failed:", e)
            if attempt == 1:
                raise
    # Now we have resized images + original CSVs. Labels are normalised with CSV sizes.
    by_split = {s: raw_dir / "annotations" / f"annotations_{s}.csv" for s in SPLITS
                if (raw_dir / "annotations" / f"annotations_{s}.csv").exists()}
    _prepare_out(out)
    stats = defaultdict(int)
    for s, p in by_split.items():
        ann = read_sku_csv(p)
        names = sorted(ann)
        if limits.get(s):
            random.Random(0).shuffle(names)
            names = names[:limits[s]]
        for n in names:
            src = raw_dir / "images" / (Path(n).stem + ".jpg")
            if not src.exists():
                stats["missing"] += 1
                continue
            d = ann[n]
            lines = boxes_to_yolo(d["boxes"], d["w"], d["h"])
            if not lines:
                stats["bad"] += 1
                continue
            os.replace(src, out / "images" / s / src.name)
            (out / "labels" / s / f"{src.stem}.txt").write_text("\n".join(lines) + "\n")
            stats["ok"] += 1
    return dict(stats), []


def ensure_val_split(out: Path, frac: float = 0.05, seed: int = 0):
    vi = out / "images" / "val"
    if any(vi.glob("*.jpg")):
        return
    ims = sorted((out / "images" / "train").glob("*.jpg"))
    random.Random(seed).shuffle(ims)
    for im in ims[:max(1, int(len(ims) * frac))]:
        os.replace(im, vi / im.name)
        lbl = out / "labels" / "train" / f"{im.stem}.txt"
        if lbl.exists():
            os.replace(lbl, out / "labels" / "val" / lbl.name)


def write_data_yaml(out: Path, names: dict[int, str]) -> Path:
    has_test = any((out / "images" / "test").glob("*.jpg"))
    lines = [f"path: {out}", "train: images/train", "val: images/val"]
    if has_test:
        lines.append("test: images/test")
    lines.append("names:")
    lines += [f"  {k}: {v}" for k, v in names.items()]
    p = out / "data.yaml"
    p.write_text("\n".join(lines) + "\n")
    return p


def count_split(out: Path) -> dict:
    return {s: len(list((out / "images" / s).glob("*.jpg"))) for s in SPLITS}


def prepare_sku110k(out: Path = TMP / "sku110k", max_side: int = 1024, limits: dict | None = None,
                    workers: int | None = None, allow_download: bool = True) -> Path:
    """Main entry. Returns the path of data.yaml. Idempotent: reuses finished work."""
    limits = limits or {}
    workers = workers or max(1, os.cpu_count() or 2)  # all cores: decode+resize is CPU-bound
    yaml_path = out / "data.yaml"
    if yaml_path.exists() and count_split(out)["train"] > 0:
        log("Reusing prepared dataset:", count_split(out))
        return yaml_path

    src = find_csv_source()
    if src:
        by_split, ds_root = src
        log("Found SKU-110K CSV annotations:", {k: str(v) for k, v in by_split.items()})
        stats, bad = build_from_csv(by_split, ds_root, out, max_side, limits, workers)
    else:
        pairs = find_yolo_source()
        if pairs:
            log("Found YOLO-format SKU-110K:", {k: str(v[0]) for k, v in pairs.items()})
            stats, bad = build_from_yolo(pairs, out, max_side, limits, workers)
        elif allow_download:
            print_input_tree()
            log("SKU-110K not found in inputs; falling back to streaming the official archive.")
            if disk_free_gb(TMP) < 8:
                raise RuntimeError("Not enough free disk for the streamed dataset (need ~8 GB).")
            stats, bad = build_from_stream(out, max_side, limits)
        else:
            print_input_tree()
            raise FileNotFoundError("SKU-110K not found. Attach a SKU-110K dataset via 'Add Input'.")
    log("Build stats:", stats)
    for b in bad[:10]:
        log("  ", b)
    ensure_val_split(out)
    counts = count_split(out)
    log("Final split sizes:", counts)
    if counts["train"] < 10 or counts["val"] < 1:
        raise RuntimeError(f"Prepared dataset is too small: {counts}")
    return write_data_yaml(out, {0: "pack"})
