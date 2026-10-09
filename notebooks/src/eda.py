"""Exploratory data analysis for the decisions the training notebooks have to make.

SKU-110K (from the annotation CSVs, no image decoding, so it's fast):
  * boxes per image -> max_det, and why YOLO26's 300-box cap matters
  * box size in pixels *after* resizing to 640 / 896 / 1024 -> how many packs become tiny
    (< 8 / 16 px) at each training resolution -> imgsz choice
  * aspect ratios, degenerate / out-of-image boxes, image sizes and orientation
  * shelf rows per image and real-gap statistics (gap_synth heuristic) -> gap class feasibility
RP2K (directory scan + image headers only):
  * images per class (imbalance), classes usable for open-set eval, image sizes
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from kaggle_common import log


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ============================================================================ SKU-110K
def load_sku_tables(by_split: dict) -> "dict":
    """by_split: split -> CSV path (sku110k_prep.find_csv_source). Returns pandas DataFrames."""
    import pandas as pd
    cols = ["image", "x1", "y1", "x2", "y2", "cls", "W", "H"]
    out = {}
    for s, p in by_split.items():
        df = pd.read_csv(p, header=None, names=cols)
        df = df[pd.to_numeric(df["x1"], errors="coerce").notna()]  # drop a header row if present
        for c in ("x1", "y1", "x2", "y2", "W", "H"):
            df[c] = df[c].astype(float)
        out[s] = df
    return out


def sku_eda(by_split: dict, out_dir: Path, sizes=(640, 896, 1024), gap_sample: int = 800, seed: int = 0) -> dict:
    """EDA from the original annotation CSVs."""
    import pandas as pd
    tables = load_sku_tables(by_split)
    df = pd.concat([t.assign(split=s) for s, t in tables.items()], ignore_index=True)
    return _sku_eda_df(df, out_dir, sizes, gap_sample, seed, source="original CSV annotations")


def load_yolo_table(root: Path, splits=("train", "val", "test"), workers: int = 16):
    """Boxes in pixels of a prepared YOLO dataset (images/<split>, labels/<split>). Image sizes come
    from JPEG headers (fast). Prepared images are resized, but ratios are preserved, so every
    relative statistic (box size vs image, boxes per image, gaps) is the same as on the originals."""
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    def one(args):
        split, im = args
        try:
            with Image.open(im) as x:
                W, H = x.size
        except Exception:
            return []
        lbl = root / "labels" / split / f"{im.stem}.txt"
        rows = []
        for line in (lbl.read_text().splitlines() if lbl.exists() else []):
            q = line.split()
            if len(q) >= 5:
                cx, cy, bw, bh = (float(v) for v in q[1:5])
                rows.append((im.name, (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H,
                             int(q[0]), float(W), float(H), split))
        return rows
    jobs = [(sp, im) for sp in splits for im in sorted((root / "images" / sp).glob("*.jpg"))]
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for lst in ex.map(one, jobs) for r in lst]
    return pd.DataFrame(rows, columns=["image", "x1", "y1", "x2", "y2", "cls", "W", "H", "split"])


def sku_eda_yolo(root: Path, out_dir: Path, sizes=(640, 896, 1024), gap_sample: int = 800, seed: int = 0) -> dict:
    """EDA from a prepared YOLO-format dataset (when the attached copy has no annotation CSVs)."""
    df = load_yolo_table(root)
    rep = _sku_eda_df(df, out_dir, sizes, gap_sample, seed, source=f"YOLO labels in {root}")
    rep["note"] = "image sizes are of the prepared (resized) copy; all box statistics are relative"
    (out_dir / "eda_sku110k.json").write_text(json.dumps(rep, indent=2, default=float))
    return rep


def show_sku_samples_yolo(root: Path, out_dir: Path, n: int = 4, seed: int = 0):
    """Prepared images with packs (green) and heuristic real gaps (red) drawn."""
    import cv2
    import gap_synth as gs
    ims = sorted((root / "images" / "train").glob("*.jpg"))
    random.Random(seed).shuffle(ims)
    plt = _plt()
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 6))
    for ax, p in zip(np.atleast_1d(axs), ims[:n]):
        img = cv2.imread(str(p))[:, :, ::-1].copy()
        h, w = img.shape[:2]
        b = gs.read_yolo(root / "labels" / "train" / f"{p.stem}.txt", w, h)
        for x1, y1, x2, y2 in b.astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        gaps = np.array(gs.real_gaps(b, gs.cluster_rows(b)) if len(b) >= 3 else []).reshape(-1, 4).astype(int)
        for x1, y1, x2, y2 in gaps:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        ax.imshow(img); ax.set_title(f"{p.stem}: {len(b)} packs, {len(gaps)} gaps", fontsize=9); ax.axis("off")
    fig.suptitle("green = annotated pack, red = heuristic real gap (used as gap labels)")
    fig.tight_layout(); fig.savefig(out_dir / "eda_sku110k_samples.png", dpi=80); plt.close(fig)


def _sku_eda_df(df, out_dir: Path, sizes, gap_sample: int, seed: int, source: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = dict(source=source)
    tables = {s: None for s in df.split.unique()}
    df = df.assign(image=df.split.astype(str) + "/" + df.image.astype(str))  # file names may repeat across splits
    df["w"], df["h"] = df.x2 - df.x1, df.y2 - df.y1
    per_img = df.groupby("image").size()
    imgs = df.groupby("image").agg(W=("W", "first"), H=("H", "first"), split=("split", "first"))
    rep["images"] = {s: int((imgs.split == s).sum()) for s in tables}
    rep["boxes_total"] = int(len(df))
    rep["boxes_per_image"] = {k: float(v) for k, v in per_img.describe(percentiles=[.5, .9, .99]).items()}
    rep["images_over_300_boxes"] = int((per_img > 300).sum())
    rep["images_over_1000_boxes"] = int((per_img > 1000).sum())
    rep["degenerate_boxes"] = int(((df.w <= 1) | (df.h <= 1)).sum())
    rep["out_of_image_boxes"] = int(((df.x1 < -1) | (df.y1 < -1) | (df.x2 > df.W + 1) | (df.y2 > df.H + 1)).sum())
    rep["image_size"] = dict(W=imgs.W.describe().to_dict(), H=imgs.H.describe().to_dict(),
                             portrait_share=float((imgs.H > imgs.W).mean()))
    long_side = np.maximum(df.W, df.H)
    short_box = np.minimum(df.w, df.h)
    rep["box_aspect_h_over_w"] = (df.h / df.w.clip(lower=1)).describe(percentiles=[.05, .5, .95]).to_dict()
    tiny = {}
    for s in sizes:
        px = short_box * (s / long_side)          # box short side in pixels at this training size
        tiny[s] = dict(lt8=float((px < 8).mean()), lt16=float((px < 16).mean()), lt32=float((px < 32).mean()),
                       median_px=float(px.median()))
    rep["box_short_side_after_resize"] = tiny

    # rows and real gaps (geometry heuristic used for gap labels), on a sample of images
    import gap_synth as gs
    names = list(per_img.index)
    random.Random(seed).shuffle(names)
    rows_per_img, gaps_per_img, gap_w_rel = [], [], []
    g = df.set_index("image")
    for n in names[:gap_sample]:
        b = g.loc[[n], ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        if len(b) < 3:
            continue
        rows = gs.cluster_rows(b)
        rows_per_img.append(len(np.unique(rows)))
        gaps = gs.real_gaps(b, rows)
        gaps_per_img.append(len(gaps))
        med_w = float(np.median(b[:, 2] - b[:, 0]))
        gap_w_rel += [(x2 - x1) / max(med_w, 1) for x1, _, x2, _ in gaps]
    rep["rows_per_image"] = dict(mean=float(np.mean(rows_per_img)), median=float(np.median(rows_per_img)))
    rep["real_gaps_per_image"] = dict(mean=float(np.mean(gaps_per_img)), share_with_gap=float(np.mean(np.array(gaps_per_img) > 0)),
                                      p90=float(np.percentile(gaps_per_img, 90)))
    rep["real_gap_width_in_pack_widths"] = dict(median=float(np.median(gap_w_rel)) if gap_w_rel else None)

    # ---- plots
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].hist(per_img, bins=60); ax[0].axvline(300, color="r", ls="--", label="YOLO26 cap (300)")
    ax[0].set_title("boxes per image"); ax[0].legend()
    for s in sizes:
        px = (short_box * (s / long_side)).clip(upper=120)
        ax[1].hist(px, bins=60, histtype="step", label=f"imgsz {s}")
    ax[1].axvline(16, color="r", ls="--"); ax[1].set_title("box short side (px) after resize"); ax[1].legend()
    ax[2].hist(gaps_per_img, bins=range(0, max(gaps_per_img + [1]) + 2)); ax[2].set_title("real gaps per image (heuristic)")
    fig.tight_layout(); fig.savefig(out_dir / "eda_sku110k.png", dpi=110); plt.close(fig)

    # ---- recommendations derived from the numbers
    rec = []
    rec.append(f"max_det >= {int(per_img.max())} needed (max boxes in one image); use 1000."
               if per_img.max() > 300 else "max_det 300 is enough.")
    best = next((s for s in sizes if tiny[s]["lt16"] < 0.05), sizes[-1])
    rec.append(f"Smallest training size with < 5% of packs under 16 px: {best} "
               f"(<16 px share: " + ", ".join(f"{s}: {tiny[s]['lt16']:.1%}" for s in sizes) + ").")
    rec.append(f"{rep['real_gaps_per_image']['share_with_gap']:.0%} of images contain at least one real gap; "
               f"mean {rep['real_gaps_per_image']['mean']:.2f} per image -> "
               + ("enough real gap labels to evaluate on real photos." if rep['real_gaps_per_image']['mean'] > 0.5
                  else "few real gaps: synthetic gaps are needed for training, evaluate on the real ones."))
    rep["recommendations"] = rec
    (out_dir / "eda_sku110k.json").write_text(json.dumps(rep, indent=2, default=float))
    for r in rec:
        log("REC:", r)
    return rep


def show_sku_samples(by_split: dict, out_dir: Path, n: int = 4, seed: int = 0):
    """A few images with boxes and heuristic real gaps drawn (decodes only n images)."""
    import cv2
    import gap_synth as gs
    from sku110k_prep import index_images
    tables = load_sku_tables({"train": by_split["train"]})["train"]
    root = Path(by_split["train"]).parents[1]
    idx = index_images(root)
    names = sorted(tables.image.unique())
    random.Random(seed).shuffle(names)
    plt = _plt()
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 6))
    for ax, name in zip(np.atleast_1d(axs), names[:n]):
        p = idx.get(name)
        if p is None:
            continue
        img = cv2.imread(str(p))[:, :, ::-1].copy()
        b = tables[tables.image == name][["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        for x1, y1, x2, y2 in b.astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 4)
        for x1, y1, x2, y2 in np.array(gs.real_gaps(b, gs.cluster_rows(b))).reshape(-1, 4).astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
        ax.imshow(img); ax.set_title(f"{name}: {len(b)} packs"); ax.axis("off")
    fig.suptitle("green = annotated pack, red = heuristic real gap")
    fig.tight_layout(); fig.savefig(out_dir / "eda_sku110k_samples.png", dpi=80); plt.close(fig)


# ============================================================================ RP2K
def rp2k_eda(items: list[tuple[str, str]], out_dir: Path, sample: int = 1500, seed: int = 0) -> dict:
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    cnt = Counter(l for _, l in items)
    v = np.array(sorted(cnt.values()))
    rep = dict(images=len(items), classes=len(cnt),
               per_class=dict(min=int(v.min()), median=float(np.median(v)), mean=float(v.mean()), max=int(v.max()),
                              p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90))),
               classes_lt6=int((v < 6).sum()), classes_ge40=int((v >= 40).sum()),
               gini=float(1 - 2 * np.sum(np.cumsum(v) / v.sum()) / len(v) + 1 / len(v)))
    rng = random.Random(seed)
    sizes = []
    for p, _ in rng.sample(items, min(sample, len(items))):
        try:
            with Image.open(p) as im:  # header only
                sizes.append(im.size)
        except Exception:
            pass
    s = np.array(sizes)
    rep["image_size"] = dict(w_median=float(np.median(s[:, 0])), h_median=float(np.median(s[:, 1])),
                             short_lt112=float((s.min(1) < 112).mean())) if len(s) else {}
    cap = 40
    rep["train_images_if_capped_at_40_per_class"] = int(np.minimum(v, cap).sum() * 0.7)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(v, bins=60); ax[0].set_title("images per class"); ax[0].set_yscale("log")
    if len(s):
        ax[1].scatter(s[:, 0], s[:, 1], s=3, alpha=0.4); ax[1].set_title("image size (w, h)")
    fig.tight_layout(); fig.savefig(out_dir / "eda_rp2k.png", dpi=110); plt.close(fig)
    rec = [f"{rep['classes']} classes, median {rep['per_class']['median']:.0f} images/class; "
           f"{rep['classes_lt6']} classes have < 6 images (excluded from open-set eval).",
           f"Capping at {cap} images/class gives ~{rep['train_images_if_capped_at_40_per_class']:,} training images "
           f"(~{rep['train_images_if_capped_at_40_per_class'] / 400 / 60:.1f} min/epoch at ~400 img/s on T4)."]
    if rep["image_size"] and rep["image_size"]["short_lt112"] > 0.2:
        rec.append("Many images are very small: 224 px input already upsamples; a larger input brings little.")
    rep["recommendations"] = rec
    (out_dir / "eda_rp2k.json").write_text(json.dumps(rep, indent=2, default=float))
    for r in rec:
        log("REC:", r)
    return rep
