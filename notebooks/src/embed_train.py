"""Product embedder: DINOv2 ViT-S/14 + 256-d head trained with ArcFace (spec Section 6.5).

Protocol (open-set, by class so test products are never seen in training):
  70% of SKUs train | 10% val-open (model selection + threshold calibration) | 20% test-open.
In val/test, half of the SKUs are put in the gallery (3-5 images each, like the store
catalogue) and half are held out as "new products" that must be answered "unknown".

ArcFace and the P x K sampler are implemented here (no extra pip packages). The margin
is in radians (0.5 rad = 28.6 deg, the pytorch-metric-learning default).
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from kaggle_common import IMG_EXT, INPUT_ROOT, WORK, cpu_workers, log, print_input_tree, save_json, walk_limited

DIM_IN = 384

try:  # torch is always present on Kaggle; optional only so metric helpers can be unit-tested anywhere
    import torch as _torch
    _Module = _torch.nn.Module
except ImportError:  # pragma: no cover
    _Module = object


# ============================================================================ data discovery
def _has_image(d: Path, probe: int = 100) -> bool:
    try:
        with os.scandir(d) as it:
            for n, e in enumerate(it):
                if Path(e.name).suffix.lower() in IMG_EXT and e.is_file():
                    return True
                if n >= probe:
                    return False
    except OSError:
        pass
    return False


def _class_dirs_with_images(d: Path, threads: int = 32) -> list[Path]:
    """Subfolders that contain images. Listed in parallel: Kaggle's /kaggle/input has high
    per-directory latency, so a sequential scan of ~2k class folders takes many minutes."""
    from concurrent.futures import ThreadPoolExecutor
    try:
        with os.scandir(d) as it:
            subs = [Path(e.path) for e in it if e.is_dir()]
    except OSError:
        return []
    with ThreadPoolExecutor(threads) as ex:
        flags = list(ex.map(_has_image, subs))
    return [s for s, f in zip(subs, flags) if f]


def _list_images(cd: Path) -> list[tuple[str, str]]:
    try:
        with os.scandir(cd) as it:
            return [(e.path, cd.name) for e in it if Path(e.name).suffix.lower() in IMG_EXT and e.is_file()]
    except OSError:
        return []


def discover_folder_dataset(root: Path = INPUT_ROOT, min_classes: int = 100) -> list[tuple[str, str]] | None:
    """Finds layouts like <root>/train/<class>/*.jpg (and merges sibling test/val dirs)."""
    from concurrent.futures import ThreadPoolExecutor
    best = None
    for dp, dns, _ in walk_limited(root, 5):
        if len(dns) < min_classes:
            continue
        cls_dirs = _class_dirs_with_images(Path(dp))
        if len(cls_dirs) >= min_classes and (best is None or len(cls_dirs) > len(best[1])):
            best = (Path(dp), cls_dirs)
        if len(cls_dirs) >= min_classes:
            dns[:] = []  # don't descend into class folders
    if best is None:
        return None
    top, top_dirs = best
    class_dirs = list(top_dirs)
    # Merge sibling splits (e.g. all/train + all/test); our open-set split is by class anyway
    if top.name.lower() in {"train", "training"}:
        for sib in top.parent.iterdir():
            if sib != top and sib.is_dir() and sib.name.lower() in {"test", "val", "valid", "validation"}:
                class_dirs += _class_dirs_with_images(sib)
    with ThreadPoolExecutor(32) as ex:
        items = [x for lst in ex.map(_list_images, class_dirs) for x in lst]
    log(f"Folder dataset at {top} ({len(class_dirs)} class folders incl. sibling splits): {len(items)} images")
    return items


def discover_csv_dataset(root: Path = INPUT_ROOT) -> list[tuple[str, str]] | None:
    """Finds a CSV with an image-path column and a label column (e.g. guie-rp2k meta.csv)."""
    import pandas as pd
    csvs = [Path(dp) / f for dp, _, fns in walk_limited(root, 5) for f in fns if f.lower().endswith(".csv")]
    for c in sorted(csvs, key=lambda p: -p.stat().st_size):
        try:
            df = pd.read_csv(c, nrows=2000)
        except Exception:
            continue
        path_col = next((col for col in df.columns if df[col].astype(str).str.lower()
                         .str.contains(r"\.(?:jpg|jpeg|png|webp|bmp)$", regex=True).mean() > 0.9), None)
        if path_col is None:
            continue
        cands = [col for col in df.columns if col != path_col and df[col].dtype == object or
                 (col != path_col and df[col].dtype.kind in "iu")]
        cands = [col for col in cands if 2 <= df[col].nunique() <= len(df) // 2]
        if not cands:
            continue
        label_col = max(cands, key=lambda col: df[col].nunique())
        df = pd.read_csv(c, usecols=[path_col, label_col]).dropna()
        # Resolve paths: relative to the CSV dir, or by basename search under the dataset root
        base = c.parent
        sample = str(df[path_col].iloc[0])
        def resolve_rel(p):
            return str(base / p)
        if not Path(resolve_rel(sample)).exists():
            idx = {}
            ds_root = root / c.relative_to(root).parts[0]
            for dp, _, fns in walk_limited(ds_root, 8):
                for f in fns:
                    if Path(f).suffix.lower() in IMG_EXT:
                        idx.setdefault(f, os.path.join(dp, f))
            paths = [idx.get(os.path.basename(str(p))) for p in df[path_col]]
        else:
            paths = [resolve_rel(p) for p in df[path_col]]
        items = [(p, str(l)) for p, l in zip(paths, df[label_col]) if p]
        if len(items) > 100:
            log(f"CSV dataset {c} path_col={path_col} label_col={label_col}: {len(items)} images")
            return items
    return None


def discover_dataset() -> list[tuple[str, str]]:
    """(path, label) for every image. The scan result is cached (cache/rp2k_index.json.gz) and
    reused by every later phase and session, so the slow directory scan happens only once."""
    import gzip
    import json as _json
    import random as _random
    from kaggle_common import find_cached
    cached = find_cached("rp2k_index.json.gz")
    if cached is not None:
        try:
            items = [tuple(x) for x in _json.loads(gzip.decompress(cached.read_bytes()))]
            probe = _random.Random(0).sample(items, min(20, len(items)))
            if items and all(os.path.exists(p) for p, _ in probe):
                log(f"Dataset index from cache {cached}: {len(items)} images")
                return items
            log("Cached dataset index is stale (paths moved); rescanning")
        except Exception as e:
            log("Could not read cached dataset index:", e)
    t0 = time.time()
    items = discover_folder_dataset() or discover_csv_dataset()
    if items:
        out = WORK / "cache" / "rp2k_index.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(gzip.compress(_json.dumps(items, ensure_ascii=False).encode("utf-8")))
        log(f"Scanned {len(items)} images in {time.time() - t0:.0f}s; index cached at {out}")
    if not items:
        print_input_tree()
        raise FileNotFoundError(
            "No product-classification dataset found under /kaggle/input. Attach RP2K "
            "(e.g. 'khyeh0719/rp2k-dataset') via Add Input, or any dataset laid out as <class>/<images>.")
    return items


def make_splits(items, seed=0, frac=(0.7, 0.1, 0.2), max_per_class_train=40, max_per_class_eval=30,
                min_eval_imgs=6, max_classes: int | None = None):
    by = defaultdict(list)
    for p, l in items:
        by[l].append(p)
    classes = sorted(by)
    rng = random.Random(seed)
    rng.shuffle(classes)
    if max_classes:
        classes = classes[:max_classes]
    n = len(classes)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    tr_c, va_c, te_c = classes[:n_tr], classes[n_tr:n_tr + n_va], classes[n_tr + n_va:]
    def take(cs, cap, min_n):
        out = {}
        for c in cs:
            ps = sorted(by[c])
            rng.shuffle(ps)
            if len(ps) >= min_n:
                out[c] = ps[:cap]
        return out
    splits = dict(train=take(tr_c, max_per_class_train, 2), val=take(va_c, max_per_class_eval, min_eval_imgs),
                  test=take(te_c, max_per_class_eval, min_eval_imgs))
    for k, v in splits.items():
        log(f"  {k}: {len(v)} classes, {sum(map(len, v.values()))} images")
    return splits


# ============================================================================ fast subset cache
def _resize_job(args):
    src, dst, max_side = args
    try:
        from PIL import Image
        with Image.open(src) as im:
            im.draft("RGB", (max_side, max_side))  # JPEG: decode directly at reduced scale (fast)
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.BILINEAR, reducing_gap=2.0)
            im.save(dst, quality=92)
        return True
    except Exception:
        return False


def subset_key(cfg: dict) -> str:
    import hashlib
    k = f"{cfg['seed']}-{cfg['max_per_class_train']}-{cfg.get('max_classes')}-{cfg.get('cache_side', 256)}"
    return "rp2k_subset_" + hashlib.md5(k.encode()).hexdigest()[:8]


def load_or_build_splits(cfg: dict, tmp_root: Path) -> dict:
    """Splits with paths into a small pre-resized copy of only the images we use.

    First session: discover the dataset, split, resize the subset in parallel (all CPU cores),
    cache it as one tar (Kaggle output / Drive). Later sessions restore the tar and skip both
    the slow directory scan and full-size JPEG decoding during training."""
    import json as _json
    from multiprocessing import Pool
    from kaggle_common import cached_dir, cpu_workers
    name = subset_key(cfg)
    side = cfg.get("cache_side", 256)

    def build(dst: Path):
        items = discover_dataset()
        splits = make_splits(items, cfg["seed"], max_per_class_train=cfg["max_per_class_train"],
                             max_classes=cfg.get("max_classes"))
        dst.mkdir(parents=True, exist_ok=True)
        jobs, manifest = [], {}
        for sp, d in splits.items():
            manifest[sp] = {}
            for ci, (c, ps) in enumerate(sorted(d.items())):
                rel = []
                for j, p in enumerate(ps):
                    r = f"{sp}/{ci:05d}_{j:03d}.jpg"
                    jobs.append((p, str(dst / r), side))
                    rel.append(r)
                manifest[sp][c] = rel
            (dst / sp).mkdir(exist_ok=True)
        log(f"Resizing {len(jobs)} images to {side}px with {cpu_workers()} processes ...")
        t0 = time.time()
        with Pool(cpu_workers()) as pool:
            ok = pool.map(_resize_job, jobs, chunksize=64)
        log(f"  done in {time.time() - t0:.0f}s, {sum(ok)}/{len(ok)} ok")
        if sum(ok) < 0.5 * len(ok):
            raise RuntimeError(f"Only {sum(ok)}/{len(ok)} images could be decoded; check the dataset files.")
        # drop images that failed to decode
        bad = {j[1] for j, k in zip(jobs, ok) if not k}
        for sp in manifest:
            for c in list(manifest[sp]):
                manifest[sp][c] = [r for r in manifest[sp][c] if str(dst / r) not in bad]
        (dst / "manifest.json").write_text(_json.dumps(manifest))

    root = cached_dir(name, tmp_root / name, build)
    manifest = _json.loads((root / "manifest.json").read_text())
    min_eval = 6
    splits = {sp: {c: [str(root / r) for r in rs] for c, rs in d.items()
                   if len(rs) >= (2 if sp == "train" else min_eval)} for sp, d in manifest.items()}
    for k, v in splits.items():
        log(f"  {k}: {len(v)} classes, {sum(map(len, v.values()))} images (cached {side}px)")
    return splits


# ============================================================================ datasets
def build_transforms():
    import torchvision.transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.15, p=1.0)], p=0.3),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.ToTensor(), norm,
        T.RandomErasing(p=0.25, scale=(0.02, 0.2)),  # simulates partial occlusion
    ])  # no horizontal flip: packs carry text
    # Eval matches deployment (shelftwin/vision/identity.py): squash-resize to 224x224
    test = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])
    return train, test


class ImageListDataset:
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = list(paths), list(labels), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        import torch
        try:
            with Image.open(self.paths[i]) as im:
                x = self.tf(im.convert("RGB"))
        except Exception:  # unreadable image: return a blank one rather than crash the epoch
            x = torch.zeros(3, 224, 224)
        return x, self.labels[i]


class PKSampler:
    """Yields batches of P classes x K images (with replacement when a class has < K)."""

    def __init__(self, labels, p: int, k: int, n_batches: int, seed: int = 0):
        self.by = defaultdict(list)
        for i, l in enumerate(labels):
            self.by[l].append(i)
        self.classes = list(self.by)
        self.p, self.k, self.n_batches = min(p, len(self.classes)), k, n_batches
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for c in self.rng.sample(self.classes, self.p):
                idx = self.by[c]
                batch += self.rng.sample(idx, self.k) if len(idx) >= self.k else [self.rng.choice(idx) for _ in range(self.k)]
            yield batch


# ============================================================================ model
TIMM_NAME = "vit_small_patch14_dinov2.lvd142m"
TIMM_KWARGS_OPTIONS = ({"img_size": 224}, {"dynamic_img_size": True})


def _check_backbone(m) -> bool:
    import torch
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 3, 224, 224))
    return tuple(out.shape) == (2, DIM_IN)


def load_backbone(pretrained: bool = True, timm_kwargs: dict | None = None):
    """DINOv2 ViT-S/14 at 224 px. Tries timm (HF hub) with img_size=224, then with
    dynamic_img_size, then torch.hub. Each candidate must pass a 224 px forward giving
    384-d features. Returns (model, source, timm_kwargs)."""
    import torch
    try:
        import timm
        for kw in ([timm_kwargs] if timm_kwargs else TIMM_KWARGS_OPTIONS):
            try:
                m = timm.create_model(TIMM_NAME, pretrained=pretrained, num_classes=0, **kw)
                if _check_backbone(m):
                    log(f"Backbone: timm {TIMM_NAME} {kw}")
                    return m, "timm", kw
            except Exception as e:
                log(f"timm {kw} failed: {e}")
    except ImportError as e:
        log("timm unavailable:", e)
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
    if not _check_backbone(m):
        raise RuntimeError("DINOv2 backbone did not produce 384-d features at 224 px")
    log("Backbone: torch.hub dinov2_vits14")
    return m, "hub", {}


class Embedder(_Module):
    def __init__(self, backbone, dim=256):
        import torch
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(DIM_IN, dim)

    def forward(self, x, use_head=True):
        import torch
        f = self.backbone(x)
        if use_head:
            f = self.head(f)
        return torch.nn.functional.normalize(f.float(), dim=1)


class TrainForward(_Module):
    """head(backbone(x)) with autocast INSIDE forward: required for nn.DataParallel, whose
    replicas run in separate threads where an outer autocast context does not apply."""

    def __init__(self, model, amp: bool):
        super().__init__()
        self.model, self.amp = model, amp

    def forward(self, x):
        import torch
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            return self.model.head(self.model.backbone(x))


def freeze_except_last(backbone, n_blocks: int = 4):
    for p in backbone.parameters():
        p.requires_grad = False
    if n_blocks <= 0:  # head-only training (note: blocks[-0:] would select ALL blocks)
        return
    blocks = list(backbone.blocks)
    for b in blocks[-n_blocks:]:
        for p in b.parameters():
            p.requires_grad = True
    for p in backbone.norm.parameters():
        p.requires_grad = True


class ArcFace(_Module):
    def __init__(self, n_classes, dim=256, s=64.0, m=0.5):
        import torch
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(n_classes, dim))
        torch.nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def forward(self, emb, labels):
        import torch
        import torch.nn.functional as F
        cos = F.linear(F.normalize(emb.float(), dim=1), F.normalize(self.W.float(), dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1.0 - cos * cos)
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        onehot = F.one_hot(labels, cos.shape[1]).to(cos.dtype)
        logits = self.s * (onehot * phi + (1 - onehot) * cos)
        return F.cross_entropy(logits, labels)


# ============================================================================ evaluation
def embed_paths(model, paths, tf, device, use_head=True, batch=256, workers=4):
    import torch
    ds = ImageListDataset(paths, [0] * len(paths), tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for x, _ in dl:
            out.append(model(x.to(device, non_blocking=True), use_head=use_head).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 256), np.float32)


def openset_protocol(split: dict, seed: int = 0, n_gallery: int = 5, max_queries: int = 10):
    """Half the classes are 'known' (gallery), half 'new products' (must be unknown)."""
    rng = random.Random(seed)
    classes = sorted(split)
    rng.shuffle(classes)
    known = set(classes[: len(classes) // 2])
    g_paths, g_lab, q_paths, q_lab, q_known = [], [], [], [], []
    for c in classes:
        ps = split[c]
        if c in known:
            g_paths += ps[:n_gallery]
            g_lab += [c] * len(ps[:n_gallery])
            qs = ps[n_gallery:n_gallery + max_queries]
        else:
            qs = ps[:max_queries]
        q_paths += qs
        q_lab += [c] * len(qs)
        q_known += [c in known] * len(qs)
    return g_paths, g_lab, q_paths, q_lab, np.array(q_known)


def retrieval_scores(g_emb, g_lab, q_emb, q_lab):
    """Per query: top class, s1 (best class sim), s2 (best other class sim), recall@1/5, AP@R."""
    sims = q_emb @ g_emb.T
    g_lab = np.array(g_lab)
    classes = np.unique(g_lab)
    cls_idx = {c: i for i, c in enumerate(classes)}
    gi = np.array([cls_idx[c] for c in g_lab])
    # max similarity per class
    per_cls = np.full((len(q_emb), len(classes)), -2.0, np.float32)
    for j in range(len(classes)):
        per_cls[:, j] = sims[:, gi == j].max(axis=1)
    order = np.argsort(-per_cls, axis=1)
    top1 = classes[order[:, 0]]
    s1 = np.take_along_axis(per_cls, order[:, :1], 1)[:, 0]
    s2 = np.take_along_axis(per_cls, order[:, 1:2], 1)[:, 0] if len(classes) > 1 else np.zeros(len(q_emb))
    top5 = classes[order[:, :5]]
    q_lab = np.array(q_lab)
    r1 = top1 == q_lab
    r5 = (top5 == q_lab[:, None]).any(1)
    # mAP@R over individual gallery images (R = gallery images of the true class)
    ap = []
    rank = np.argsort(-sims, axis=1)
    for i in range(len(q_emb)):
        R = int((g_lab == q_lab[i]).sum())
        if R == 0:
            ap.append(np.nan)
            continue
        rel = (g_lab[rank[i, :R]] == q_lab[i]).astype(float)
        prec = np.cumsum(rel) / np.arange(1, R + 1)
        ap.append(float((prec * rel).sum() / R))
    return dict(top1=top1, s1=s1, s2=s2, r1=r1, r5=r5, ap=np.array(ap))


def summarize(sc, q_known, thr_sim=None, thr_margin=None):
    k = q_known
    out = dict(recall_at_1=float(sc["r1"][k].mean()), recall_at_5=float(sc["r5"][k].mean()),
               map_at_r=float(np.nanmean(sc["ap"][k])), n_known_queries=int(k.sum()),
               n_unknown_queries=int((~k).sum()))
    if thr_sim is not None:
        acc = (sc["s1"] >= thr_sim) & ((sc["s1"] - sc["s2"]) >= thr_margin)
        correct = acc & k & sc["r1"]
        out.update(thr_sim=float(thr_sim), thr_margin=float(thr_margin),
                   accepted_precision=float(correct.sum() / max(1, acc.sum())),
                   known_coverage=float(correct.sum() / max(1, k.sum())),
                   unknown_rejection=float((~acc[~k]).mean()) if (~k).any() else None)
    return out


def calibrate_thresholds(sc, q_known, target_precision=0.98, min_accepted=10):
    """Max coverage subject to precision >= target. If no setting reaches the target,
    fall back to the most precise setting (ties -> more coverage) instead of accepting nothing."""
    s1 = np.asarray(sc["s1"])
    lo = float(np.floor(max(-1.0, s1.min()) * 100) / 100) if len(s1) else 0.2
    best, most_precise = None, None
    for a in np.arange(min(lo, 0.95), 0.99, 0.01):
        for m in np.arange(0.0, 0.31, 0.01):
            acc = (s1 >= a) & ((s1 - sc["s2"]) >= m)
            if acc.sum() < min_accepted:
                continue
            correct = acc & q_known & sc["r1"]
            prec = correct.sum() / acc.sum()
            cov = correct.sum() / max(1, q_known.sum())
            cand = (float(a), float(m), float(cov), float(prec))
            if prec >= target_precision and (best is None or cov > best[2]):
                best = cand
            if most_precise is None or (prec, cov) > (most_precise[3], most_precise[2]):
                most_precise = cand
    if best is None:
        if most_precise is None:
            log("Too few queries to calibrate; keeping spec defaults 0.75 / 0.15.")
            return 0.75, 0.15
        log(f"No threshold reaches {target_precision:.0%} precision; using the most precise setting.")
        best = most_precise
    log(f"Calibrated: sim>={best[0]:.2f}, margin>={best[1]:.2f} -> coverage {best[2]:.3f}, precision {best[3]:.3f}")
    return best[0], best[1]


def evaluate_split(model, split, tf, device, use_head=True, seed=0, workers=4):
    g_p, g_l, q_p, q_l, q_k = openset_protocol(split, seed)
    g = embed_paths(model, g_p, tf, device, use_head, workers=workers)
    q = embed_paths(model, q_p, tf, device, use_head, workers=workers)
    sc = retrieval_scores(g, g_l, q, q_l)
    return sc, q_k


# ============================================================================ training
def train(cfg: dict) -> dict:
    import torch
    t_start = time.time()
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = cfg["workers"] if cfg.get("workers") is not None else cpu_workers()
    out_dir = Path(cfg.get("out_dir", WORK / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from kaggle_common import TMP
    if cfg.get("cache_subset", True):
        splits = load_or_build_splits(cfg, TMP)
    else:
        items = discover_dataset()
        splits = make_splits(items, cfg["seed"], max_per_class_train=cfg["max_per_class_train"],
                             max_classes=cfg.get("max_classes"))
    tf_train, tf_test = build_transforms()
    tr_classes = sorted(splits["train"])
    cidx = {c: i for i, c in enumerate(tr_classes)}
    paths = [p for c in tr_classes for p in splits["train"][c]]
    labels = [cidx[c] for c in tr_classes for _ in splits["train"][c]]

    backbone, source, timm_kwargs = load_backbone()
    model = Embedder(backbone, cfg["dim"]).to(device)
    report = dict(config=cfg, backbone_source=source, n_train_classes=len(tr_classes), n_train_images=len(paths))

    resume_path = out_dir / "embedder_resume.pt"
    state = None
    if cfg.get("resume", True) and resume_path.exists():
        try:
            state = torch.load(resume_path, map_location="cpu", weights_only=True)
            if state.get("n_classes") != len(tr_classes):
                log("Resume state is for a different class split; starting fresh.")
                state = None
        except Exception as e:
            log("Could not read resume state (", e, "); starting fresh.")
            state = None
    if state is not None and state.get("zero_shot_test") is not None:
        report["zero_shot_test"] = state["zero_shot_test"]

    # Zero-shot baseline (pretrained DINOv2 CLS, no head) on the test-open split
    if cfg.get("zero_shot_baseline", True) and "zero_shot_test" not in report:
        sc, qk = evaluate_split(model, splits["test"], tf_test, device, use_head=False, workers=nw)
        report["zero_shot_test"] = summarize(sc, qk)
        log("Zero-shot DINOv2 test:", report["zero_shot_test"])

    freeze_except_last(model.backbone, cfg["unfreeze_blocks"])
    arc = ArcFace(len(tr_classes), cfg["dim"], s=cfg["arc_scale"], m=cfg["arc_margin"]).to(device)
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        dict(params=bb_params, lr=cfg["lr_backbone"]),
        dict(params=model.head.parameters(), lr=cfg["lr_head"]),
        dict(params=arc.parameters(), lr=cfg["lr_head"], weight_decay=0.0),
    ], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    P, K = cfg["P"], cfg["K"]
    start_epoch, best_r1, step, history, done = 0, -1.0, 0, [], False
    if state is not None:
        model.backbone.load_state_dict(state["backbone"]); model.head.load_state_dict(state["head"])
        arc.load_state_dict(state["arc"]); opt.load_state_dict(state["opt"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch, best_r1, step = state["epoch"] + 1, state["best_r1"], state["step"]
        history, done, P = state["history"], state.get("done", False), state.get("P", P)
        log(f"Resuming embedder training at epoch {start_epoch} (best val R@1 {best_r1:.4f}, done={done})")
    steps_per_epoch = max(1, len(paths) // (P * K))
    total_steps = steps_per_epoch * cfg["epochs"]
    warm = min(steps_per_epoch, 500)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * max(f, 0.01)

    ds = ImageListDataset(paths, labels, tf_train)
    fwd = TrainForward(model, amp=device.type == "cuda")
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpu > 1 and cfg.get("multi_gpu", True):
        fwd = torch.nn.DataParallel(fwd)  # both T4s on Kaggle: ~1.7x throughput
        log(f"Training on {n_gpu} GPUs (DataParallel)")
    nw = cfg["workers"] if cfg.get("workers") is not None else cpu_workers()
    budget_s = cfg["time_budget_h"] * 3600
    finished_cleanly = True
    for epoch in range(start_epoch, 0 if done else cfg["epochs"]):
        sampler = PKSampler(labels, P, K, steps_per_epoch, seed=cfg["seed"] + epoch)
        dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=nw, pin_memory=True,
                                         prefetch_factor=4 if nw > 0 else None)
        model.train()
        t0, losses = time.time(), []
        for x, y in dl:
            set_lr(step)
            x, y = x.to(device, non_blocking=True), torch.as_tensor(y, device=device)
            try:
                emb = fwd(x)
                loss = arc(emb, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 5.0)
                scaler.step(opt)
                scaler.update()
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if P <= 8:
                    raise
                P = P // 2
                log(f"OOM: reducing P to {P} (batch {P * K}) and restarting the epoch")
                break
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                log(f"  ep {epoch} step {step}/{total_steps} loss {np.mean(losses[-100:]):.3f}")
            if time.time() - t_start > budget_s:
                break
        # Validation (open-set) for model selection
        sc, qk = evaluate_split(model, splits["val"], tf_test, device, use_head=True, workers=nw)
        val = summarize(sc, qk)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)) if losses else None,
                            val_r1=val["recall_at_1"], val_map_r=val["map_at_r"], sec=time.time() - t0))
        log(f"epoch {epoch}: loss {history[-1]['loss']}, val R@1 {val['recall_at_1']:.4f}, "
            f"mAP@R {val['map_at_r']:.4f}, {time.time() - t0:.0f}s")
        ck = dict(backbone=model.backbone.state_dict(), head=model.head.state_dict(), dim=cfg["dim"],
                  source=source, model_name=TIMM_NAME if source == "timm" else "dinov2_vits14",
                  timm_kwargs=timm_kwargs, img_size=224, epoch=epoch)
        if val["recall_at_1"] > best_r1:
            best_r1 = val["recall_at_1"]
            _atomic_save(ck, out_dir / "embedder_best.pt")
        save_json(history, out_dir / "embedder_history.json")
        stop_budget = time.time() - t_start > budget_s
        stop_early = (len(history) >= 4 and max(h["val_r1"] for h in history[-3:]) < best_r1 - 1e-4
                      and epoch > cfg["epochs"] // 2)
        done = stop_early or epoch == cfg["epochs"] - 1
        _atomic_save(dict(backbone=model.backbone.state_dict(), head=model.head.state_dict(), arc=arc.state_dict(),
                          opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=epoch, best_r1=best_r1, step=step,
                          history=history, done=done, P=P, n_classes=len(tr_classes),
                          zero_shot_test=report.get("zero_shot_test")), resume_path)
        if stop_budget:
            log("Time budget reached; stopping training (run again to resume).")
            finished_cleanly = False
            break
        if stop_early:
            log("Early stop: no val improvement in 3 epochs.")
            break

    # Final: load best, calibrate thresholds on val-open, report on test-open
    if not (out_dir / "embedder_best.pt").exists():
        raise RuntimeError("No embedder checkpoint was produced (training did not finish one epoch).")
    report["training_complete"] = bool(done and finished_cleanly)
    ck = torch.load(out_dir / "embedder_best.pt", map_location=device, weights_only=True)
    model.backbone.load_state_dict(ck["backbone"]); model.head.load_state_dict(ck["head"])
    sc, qk = evaluate_split(model, splits["val"], tf_test, device, workers=nw)
    a, m = calibrate_thresholds(sc, qk, cfg["target_precision"])
    sc, qk = evaluate_split(model, splits["test"], tf_test, device, workers=nw)
    report["test"] = summarize(sc, qk, a, m)
    report["thresholds"] = dict(accept_sim=a, accept_margin=m, target_precision=cfg["target_precision"])
    report["history"] = history
    report["train_minutes"] = (time.time() - t_start) / 60
    log("Test (open-set):", report["test"])
    save_json(report["thresholds"], out_dir / "identity_thresholds.json")
    save_json(report, out_dir / "embedder_report.json")
    _plot_scores(sc, qk, a, m, out_dir / "embedder_scores.png")
    return report


def _atomic_save(obj, path: Path):
    """Write to a temp file then rename, so a disconnect mid-save never corrupts a checkpoint."""
    import torch
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _plot_scores(sc, q_known, a, m, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        groups = {"correct": q_known & sc["r1"], "wrong": q_known & ~sc["r1"], "unknown (new SKU)": ~q_known}
        for name, msk in groups.items():
            ax[0].hist(sc["s1"][msk], bins=40, alpha=0.5, label=name, density=True)
            ax[1].hist((sc["s1"] - sc["s2"])[msk], bins=40, alpha=0.5, label=name, density=True)
        ax[0].axvline(a, color="k", ls="--"); ax[1].axvline(m, color="k", ls="--")
        ax[0].set_title("s1 (top similarity)"); ax[1].set_title("s1 - s2 (margin)")
        ax[0].legend()
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e:
        log("plot failed:", e)


def export_onnx(out_dir: Path, opset: int = 18):
    import torch
    try:
        ck = torch.load(out_dir / "embedder_best.pt", map_location="cpu", weights_only=True)
        if ck.get("source") == "timm":
            backbone, _, _ = load_backbone(pretrained=False, timm_kwargs=ck.get("timm_kwargs") or None)
        else:
            backbone, _, _ = load_backbone()
        model = Embedder(backbone, ck["dim"])
        model.backbone.load_state_dict(ck["backbone"]); model.head.load_state_dict(ck["head"])
        model.eval()
        kw = dict(input_names=["images"], output_names=["embeddings"], opset_version=opset,
                  dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}})
        x, dst = torch.randn(2, 3, 224, 224), str(out_dir / "embedder.onnx")
        try:  # legacy TorchScript exporter: needs only the `onnx` package
            torch.onnx.export(model, x, dst, dynamo=False, **kw)
        except TypeError:  # older torch without the `dynamo` argument
            torch.onnx.export(model, x, dst, **kw)
        except Exception as e1:  # newer torch may require the dynamo exporter (onnxscript)
            log("legacy ONNX export failed:", e1, "-> installing onnx/onnxscript and retrying")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"])
            torch.onnx.export(model, x, dst, **kw)
        log("ONNX saved:", dst)
    except Exception as e:
        log("ONNX export failed (non-fatal; the .pt checkpoint is the main output):", e)


# ============================================================================ experiments / error analysis
def _load_trained(ckpt_path: Path, device):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ck.get("source") == "timm":
        backbone, _, _ = load_backbone(pretrained=False, timm_kwargs=ck.get("timm_kwargs") or None)
    else:
        backbone, _, _ = load_backbone()
    model = Embedder(backbone, ck["dim"])
    model.backbone.load_state_dict(ck["backbone"]); model.head.load_state_dict(ck["head"])
    return model.to(device).eval()


def zero_shot_eval(timm_name: str, cfg: dict, split: str = "test") -> dict:
    """Retrieval quality of a pretrained backbone with no training (e.g. DINOv2 ViT-B vs ViT-S):
    tells whether a bigger backbone is worth its extra latency before spending GPU time."""
    import timm
    import torch
    from kaggle_common import TMP, cpu_workers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_or_build_splits(cfg, TMP)
    m = None
    for kw in TIMM_KWARGS_OPTIONS:
        try:
            m = timm.create_model(timm_name, pretrained=True, num_classes=0, **kw)
            break
        except Exception as e:
            log(f"{timm_name} {kw}: {e}")
    if m is None:
        return dict(model=timm_name, error="could not load")
    model = Embedder(m, 256).to(device).eval()
    _, tf_test = build_transforms()
    t0 = time.time()
    sc, qk = evaluate_split(model, splits[split], tf_test, device, use_head=False, workers=cpu_workers())
    out = dict(model=timm_name, params_M=round(sum(p.numel() for p in m.parameters()) / 1e6, 1), **summarize(sc, qk),
               eval_seconds=round(time.time() - t0, 1))
    log("zero-shot", out)
    return out


def analyze_errors(cfg: dict, ckpt_path: Path, out_dir: Path, n_gallery: int = 8, seed: int = 0) -> dict:
    """Where does identity fail? Most-confused SKU pairs, per-class recall spread, the
    precision / coverage / unknown-rejection trade-off, and a gallery of failures."""
    import json as _json
    from collections import Counter
    import torch
    from PIL import Image
    from kaggle_common import TMP, cpu_workers
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_or_build_splits(cfg, TMP)
    model = _load_trained(ckpt_path, device)
    _, tf_test = build_transforms()
    g_p, g_l, q_p, q_l, q_k = openset_protocol(splits["test"], seed)
    g = embed_paths(model, g_p, tf_test, device, workers=cpu_workers())
    q = embed_paths(model, q_p, tf_test, device, workers=cpu_workers())
    sc = retrieval_scores(g, g_l, q, q_l)
    q_l = np.array(q_l)
    wrong = q_k & ~sc["r1"]
    pairs = Counter(zip(q_l[wrong], sc["top1"][wrong])).most_common(15)
    per_cls = {}
    for c in np.unique(q_l[q_k]):
        m = q_k & (q_l == c)
        per_cls[c] = float(sc["r1"][m].mean())
    v = np.array(list(per_cls.values()))
    # trade-off curve: sweep the similarity threshold at the calibrated margin
    th_path = Path(ckpt_path).parent / "identity_thresholds.json"
    margin = _json.loads(th_path.read_text())["accept_margin"] if th_path.exists() else 0.15
    curve = []
    for a in np.arange(0.2, 1.0, 0.02):
        acc = (sc["s1"] >= a) & ((sc["s1"] - sc["s2"]) >= margin)
        correct = acc & q_k & sc["r1"]
        curve.append(dict(sim=float(a), precision=float(correct.sum() / max(1, acc.sum())),
                          coverage=float(correct.sum() / max(1, q_k.sum())),
                          unknown_rejection=float((~acc[~q_k]).mean()) if (~q_k).any() else None))
    rep = dict(recall_at_1=float(sc["r1"][q_k].mean()), n_known=int(q_k.sum()), n_wrong=int(wrong.sum()),
               most_confused=[dict(true=a, predicted=b, count=n) for (a, b), n in pairs],
               per_class_recall=dict(p10=float(np.percentile(v, 10)), median=float(np.median(v)),
                                     classes_below_50pct=int((v < 0.5).sum()), classes=len(v)),
               margin_used=margin, curve=curve)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot([c["coverage"] for c in curve], [c["precision"] for c in curve], "o-", ms=3)
    ax[0].set_xlabel("coverage (known products correctly accepted)"); ax[0].set_ylabel("precision of accepted")
    ax[0].axhline(0.98, color="r", ls="--"); ax[0].set_title(f"identity trade-off (margin {margin:.2f})")
    ax[1].hist(v, bins=20); ax[1].set_title("per-SKU recall@1 (known SKUs)")
    fig.tight_layout(); fig.savefig(out_dir / "identity_tradeoff.png", dpi=110); plt.close(fig)
    # failure gallery: query | predicted SKU | true SKU
    first = {}
    for p, l in zip(g_p, g_l):
        first.setdefault(l, p)
    idx = np.where(wrong)[0][:n_gallery]
    if len(idx):
        fig, axs = plt.subplots(len(idx), 3, figsize=(7, 2.3 * len(idx)), squeeze=False)
        for row, i in enumerate(idx):
            for col, (p, t) in enumerate([(q_p[i], "query"), (first.get(sc["top1"][i]), "predicted"),
                                          (first.get(q_l[i]), "true SKU")]):
                axs[row, col].axis("off")
                if p:
                    with Image.open(p) as im:
                        axs[row, col].imshow(im.convert("RGB"))
                axs[row, col].set_title(f"{t} s={sc['s1'][i]:.2f}" if col == 1 else t, fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / "identity_failures.png", dpi=90); plt.close(fig)
    (out_dir / "identity_error_analysis.json").write_text(_json.dumps(rep, indent=2, ensure_ascii=False))
    log("identity errors:", {k: rep[k] for k in ("recall_at_1", "n_wrong", "per_class_recall")})
    return rep


def calibrate_zero_shot(cfg: dict, out_dir: Path) -> dict:
    """When the experiments show fine-tuning does not help: keep the pretrained DINOv2 backbone
    (what shelftwin.vision.identity.Embedder(None) loads) and only calibrate the abstention
    thresholds on val, then report on test with the same open-set protocol."""
    import torch
    from kaggle_common import TMP, cpu_workers
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_or_build_splits(cfg, TMP)
    backbone, source, kw = load_backbone()
    model = Embedder(backbone, cfg["dim"]).to(device).eval()
    _, tf_test = build_transforms()
    nw = cfg["workers"] if cfg.get("workers") is not None else cpu_workers()
    sc, qk = evaluate_split(model, splits["val"], tf_test, device, use_head=False, workers=nw)
    a, m = calibrate_thresholds(sc, qk, cfg["target_precision"])
    sc, qk = evaluate_split(model, splits["test"], tf_test, device, use_head=False, workers=nw)
    rep = dict(mode="zero_shot", backbone_source=source, timm_kwargs=kw, test=summarize(sc, qk, a, m),
               thresholds=dict(accept_sim=a, accept_margin=m, target_precision=cfg["target_precision"]))
    save_json(rep["thresholds"], out_dir / "identity_thresholds.json")
    save_json(rep, out_dir / "embedder_report.json")
    _plot_scores(sc, qk, a, m, out_dir / "embedder_scores.png")
    log("Zero-shot identity (test):", rep["test"])
    return rep
