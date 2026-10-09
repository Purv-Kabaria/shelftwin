"""Builds the Kaggle notebooks and the all-in-one Colab notebook from tested source files.

    python notebooks/build_notebooks.py

Kaggle notebooks embed notebooks/src/*.py with %%writefile. The Colab notebook embeds the whole
code base (shelftwin/, sim/, notebooks/src/) as one base64 zip, so a single file is uploaded.
Edit the .py sources (and run pytest), then rebuild; never hand-edit the .ipynb files.
"""
from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "src"


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s: str):
    return nbf.v4.new_code_cell(s.strip())


def writefile(dst: str, src: Path):
    return code(f"%%writefile {dst}\n" + src.read_text(encoding="utf-8"))


ULTRA_CHECK = r'''
import os, sys, subprocess
def _ultra_ok():
    # importlib.metadata does not import the package, so an old version never gets cached in this kernel
    from importlib.metadata import version, PackageNotFoundError
    try:
        s = version("ultralytics")
    except PackageNotFoundError:
        return False, None
    try:
        v = tuple(int("".join(ch for ch in x if ch.isdigit()) or 0) for x in s.split(".")[:3])
    except Exception:
        return False, s
    return v >= (8, 4, 134), s
# Ultralytics >= 8.4.134 retries dense-image label assignment per image on GPU after an OOM
# instead of a very slow CPU fallback (SKU-110K has 150-700 boxes per image).
ok, ver = _ultra_ok()
print("ultralytics installed:", ver)
if not ok:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics>=8.4.134"])
    if r.returncode != 0:
        print("WARNING: could not install ultralytics>=8.4.134 (Internet on?). Trying any version.")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
'''

SETUP_KAGGLE = r'''
import os, sys
os.makedirs("/kaggle/working/src", exist_ok=True)
sys.path.insert(0, "/kaggle/working/src")
''' + ULTRA_CHECK + r'''
print(subprocess.run([sys.executable, "-c", "import ultralytics, torch; print('ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| GPUs', torch.cuda.device_count())"], capture_output=True, text=True).stdout)
'''

KAGGLE_CHECKLIST = """
**Before running (Kaggle notebook settings, right-hand panel):**
1. *Accelerator*: **{accel}** · *Internet*: **On** (phone-verified account)
2. *Add Input*: {inputs}
3. First run: set `SMOKE_TEST = True` and *Run All* (~{smoke} min). It must end with `ALL DONE`. Then set it back to `False`.
4. Real run: **Save Version → Save & Run All (Commit)**. It runs in the background and keeps `/kaggle/working` as output.

**Resuming (built in).** If the run prints `NOT FINISHED`, open the notebook again, *Add Input → Your Work → Notebooks →*
this notebook (its latest version), and *Save & Run All* again. Checkpoints, the prepared dataset cache and the
training state are restored automatically and training continues from the last epoch with the same schedule.
"""


# ============================================================================ Kaggle 01
def nb01():
    return [
        md("""
# ShelfTwin 01: pack detector pre-training on SKU-110K (resumable)

Class-agnostic **pack** detector (YOLO11s with NMS) on SKU-110K: 11,762 dense shelf photos, ~147 boxes per image.
YOLO26 isn't used because its NMS-free head is trained to output at most 300 boxes, while SKU-110K images can hold more than 500.

**Speed:** both T4s (DDP); one-time parallel pre-resize on all CPU cores (decoding 3000 px JPEGs is the bottleneck),
cached as one tar for later sessions; training stops right after an epoch checkpoint when the session budget runs out.
**Robustness:** dataset auto-discovery, a streaming fallback from the official archive, DDP→1-GPU fallback,
a stall watchdog, and W&B and other loggers disabled.
""" + KAGGLE_CHECKLIST.format(accel="GPU T4 x2", smoke=15,
                              inputs="search **SKU110K** and add `thedatasith/sku110k-annotations` (any copy with "
                                     "`annotations_*.csv` + images, or a YOLO-format copy, works)") + """
Outputs: `outputs/pack_detector_sku110k.pt` (+ `.onnx` when finished), `outputs/metrics_01.json`.
"""),
        code("""
SMOKE_TEST = False
SESSION_HOURS = 11.0     # Kaggle hard limit is 12 h; training gets what is left after prep, minus RESERVE_H
RESERVE_H = 0.6          # kept for evaluation, export and saving outputs
EPOCHS = 40              # total schedule (may span several sessions)
IMGSZ = 896              # dense shelves need resolution; 1024 = +30% time, slightly better small-object recall
MODEL = "yolo11s.pt"     # 'yolo11n.pt' trains ~2x faster (edge model)
BATCH = 16               # total over 2 GPUs; BATCH_PER_GPU is used on a single GPU
BATCH_PER_GPU = 8
"""),
        code(SETUP_KAGGLE),
        writefile("/kaggle/working/src/kaggle_common.py", SRC / "kaggle_common.py"),
        writefile("/kaggle/working/src/sku110k_prep.py", SRC / "sku110k_prep.py"),
        writefile("/kaggle/working/src/yolo_runner.py", SRC / "yolo_runner.py"),
        writefile("/kaggle/working/src/train_yolo_cli.py", SRC / "train_yolo_cli.py"),
        code("""
import kaggle_common as kc
kc.quiet_loggers()
kc.restore_previous_outputs("nb01")
kc.mark_session("nb01")
kc.log("GPUs:", kc.gpu_info(), "| CPUs:", os.cpu_count(), "| free /tmp GB:", round(kc.disk_free_gb("/tmp"), 1))
kc.print_input_tree(max_depth=2)
"""),
        code("""
import sku110k_prep as sp
name = f"sku110k_{IMGSZ}" + ("_smoke" if SMOKE_TEST else "")
limits = dict(train=64, val=16, test=16) if SMOKE_TEST else {}
src_dir = kc.cached_dir(name, kc.TMP / name, lambda d: sp.prepare_sku110k(out=d, max_side=IMGSZ, limits=limits))
data_yaml = src_dir / "data.yaml"
print(open(data_yaml).read(), sp.count_split(src_dir))
"""),
        code("""
import yolo_runner as yr
train_args = dict(
    model=MODEL, data=str(data_yaml), imgsz=640 if SMOKE_TEST else IMGSZ, epochs=2 if SMOKE_TEST else EPOCHS,
    batch=8 if SMOKE_TEST else BATCH, batch_per_gpu=4 if SMOKE_TEST else BATCH_PER_GPU,
    name="sku110k_y11s" + ("_smoke" if SMOKE_TEST else ""),
    optimizer="AdamW", lr0=0.002, cos_lr=True, warmup_epochs=1 if SMOKE_TEST else 3,
    mosaic=1.0, close_mosaic=0 if SMOKE_TEST else 10, fliplr=0.5, degrees=0.0, perspective=0.0005,
    max_det=1000, amp=True, patience=15, seed=0, deterministic=False, plots=True,
)
res = yr.train(train_args, budget_h=SESSION_HOURS - kc.elapsed_h() - RESERVE_H)
"""),
        code("""
split = "test" if res["finished"] else "val"     # quick val check while training is unfinished
metrics = yr.evaluate(res["best"], data_yaml, imgsz=640 if SMOKE_TEST else IMGSZ, split=split, max_det=1000, iou=0.6)
onnx = yr.export_onnx(res["best"], 640 if SMOKE_TEST else IMGSZ) if res["finished"] else None
out = yr.package_outputs({"pack_detector_sku110k.pt": res["best"], "pack_detector_sku110k.onnx": onnx,
                          "results_01.csv": yr.run_dir(train_args["name"]) / "results.csv"})
kc.save_json(dict(metrics=metrics, finished=res["finished"], epochs_done=res["epochs_done"], train_args=train_args,
                  smoke_test=SMOKE_TEST, hours_used=round(kc.elapsed_h(), 2)), out / "metrics_01.json")
print(metrics)
print("ALL DONE" if res["finished"] else "NOT FINISHED: attach this version's output as input and run again to resume.",
      "| session hours:", round(kc.elapsed_h(), 2))
"""),
    ]


# ============================================================================ Kaggle 02
def nb02():
    return [
        md("""
# ShelfTwin 02: pack + gap detector (resumable)

Fine-tunes the notebook-01 detector to two classes, **pack** and **gap**, on 640 px square crops. Their scale matches deployment,
where a 1080p frame is split into two ~1080 px tiles and each is run at 640.

* *Real gaps* (all splits): geometric heuristic on real photos (a wide space between neighbouring packs in a shelf row).
* *Synthetic gaps* (**train only**, spec 5.2.7): runs of packs erased and filled by inpainting, shelf-back texture, or real gap patches.

Metrics: per-class AP, and **gap recall at precision 0.9** on real test photos.
""" + KAGGLE_CHECKLIST.format(accel="GPU T4 x2", smoke=15,
                              inputs="the SKU-110K dataset **and** the output of your notebook-01 run (*Your Work → "
                                     "Notebooks → ShelfTwin 01*): its weights are used to start from, and its prepared "
                                     "dataset cache is reused, which skips about 10 min of preparation") + """
Outputs: `outputs/pack_gap.pt` (+ `.onnx` when finished), `outputs/metrics_02.json`.
"""),
        code("""
SMOKE_TEST = False
SESSION_HOURS = 11.0
RESERVE_H = 0.6
EPOCHS = 30
SRC_SIDE = 896              # must match notebook 01's IMGSZ to reuse its dataset cache
MAX_TRAIN_IMAGES = 5000     # source photos for training (x2 crops each); None = all ~8.2k
IMGSZ = 640
BATCH = 32
BATCH_PER_GPU = 16
"""),
        code(SETUP_KAGGLE),
        writefile("/kaggle/working/src/kaggle_common.py", SRC / "kaggle_common.py"),
        writefile("/kaggle/working/src/sku110k_prep.py", SRC / "sku110k_prep.py"),
        writefile("/kaggle/working/src/gap_synth.py", SRC / "gap_synth.py"),
        writefile("/kaggle/working/src/yolo_runner.py", SRC / "yolo_runner.py"),
        writefile("/kaggle/working/src/train_yolo_cli.py", SRC / "train_yolo_cli.py"),
        code("""
import kaggle_common as kc, sku110k_prep as sp, gap_synth as gs, yolo_runner as yr
kc.quiet_loggers()
kc.restore_previous_outputs("nb02")
kc.mark_session("nb02")
init = kc.find_weights(("pack_detector_sku110k.pt",)) or kc.find_weights(("best.pt",), must_contain="sku110k")
kc.log("GPUs:", kc.gpu_info(), "| CPUs:", os.cpu_count(), "| initial weights:", init or "none -> yolo11s.pt (COCO)")
"""),
        code("""
sfx = "_smoke" if SMOKE_TEST else ""
src_name = f"sku110k_{SRC_SIDE}{sfx}"
lim = dict(train=40, val=10, test=10) if SMOKE_TEST else {}
src_dir = kc.cached_dir(src_name, kc.TMP / src_name, lambda d: sp.prepare_sku110k(out=d, max_side=SRC_SIDE, limits=lim),
                        keep_cache=False)   # notebook 01's cache is reused when attached
pg_name = f"packgap_{IMGSZ}_{MAX_TRAIN_IMAGES}{sfx}"
pg_dir = kc.cached_dir(pg_name, kc.TMP / pg_name, lambda d: gs.build_pack_gap(
    src_dir, d, limits=dict(train=24) if SMOKE_TEST else dict(train=MAX_TRAIN_IMAGES), n_crops=2, out_size=IMGSZ, p_synth=0.6))
pg_yaml = pg_dir / "data.yaml"
print(open(pg_yaml).read())
"""),
        code("""
# Preview a few training crops with labels (sanity check of the synthetic gaps)
import cv2, random, matplotlib.pyplot as plt
ims = sorted((pg_dir / "images" / "train").glob("*.jpg")); random.Random(1).shuffle(ims)
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for ax, p in zip(axs, ims[:4]):
    im = cv2.imread(str(p))[:, :, ::-1].copy(); h, w = im.shape[:2]
    for line in (pg_dir / "labels" / "train" / f"{p.stem}.txt").read_text().splitlines():
        c, cx, cy, bw, bh = line.split(); cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        cv2.rectangle(im, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)), (int((cx + bw / 2) * w), int((cy + bh / 2) * h)),
                      (255, 0, 0) if c == "1" else (0, 200, 0), 2 if c == "1" else 1)
    ax.imshow(im); ax.axis("off")
plt.suptitle("green = pack, red = gap"); plt.savefig("/kaggle/working/packgap_preview.png", dpi=80); plt.show()
"""),
        code("""
train_args = dict(
    model=str(init) if init else "yolo11s.pt", data=str(pg_yaml), imgsz=IMGSZ, epochs=2 if SMOKE_TEST else EPOCHS,
    batch=8 if SMOKE_TEST else BATCH, batch_per_gpu=4 if SMOKE_TEST else BATCH_PER_GPU, name="packgap" + sfx,
    optimizer="AdamW", lr0=0.001, cos_lr=True, warmup_epochs=1,
    mosaic=0.5, close_mosaic=0 if SMOKE_TEST else 5, fliplr=0.5, degrees=0.0, hsv_v=0.5,
    max_det=1000, amp=True, patience=12, seed=0, deterministic=False,
)
res = yr.train(train_args, budget_h=SESSION_HOURS - kc.elapsed_h() - RESERVE_H)
"""),
        code("""
split = "test" if res["finished"] and (pg_dir / "images" / "test").exists() else "val"
metrics = yr.evaluate(res["best"], pg_yaml, imgsz=IMGSZ, split=split, max_det=1000, iou=0.6)
gap = gs.gap_recall_at_precision(res["best"], pg_dir, split=split, target_p=0.9, imgsz=IMGSZ,
                                 max_images=200 if SMOKE_TEST else 2000)
print("gap recall @ precision 0.9:", gap)
onnx = yr.export_onnx(res["best"], IMGSZ) if res["finished"] else None
out = yr.package_outputs({"pack_gap.pt": res["best"], "pack_gap.onnx": onnx,
                          "results_02.csv": yr.run_dir(train_args["name"]) / "results.csv",
                          "packgap_preview.png": kc.WORK / "packgap_preview.png"})
kc.save_json(dict(metrics=metrics, gap_recall_at_p90=gap, init_weights=str(init), finished=res["finished"],
                  epochs_done=res["epochs_done"], smoke_test=SMOKE_TEST, hours_used=round(kc.elapsed_h(), 2)),
             out / "metrics_02.json")
print("ALL DONE" if res["finished"] else "NOT FINISHED: attach this version's output as input and run again to resume.")
"""),
    ]


# ============================================================================ Kaggle 04
EMBED_CFG = """
CFG = dict(
    seed=0, dim=256, unfreeze_blocks=4, arc_scale=64.0, arc_margin=0.5,   # 0.5 rad = 28.6 deg
    lr_backbone=1e-5, lr_head=1e-3, weight_decay=0.05,
    P=48, K=4, epochs=20, workers=None,          # workers None = all CPU cores
    max_per_class_train=40,      # caps images per training SKU to fit the free GPU quota
    max_classes=None,            # None = all SKUs in the dataset
    cache_subset=True, cache_side=256,   # one-time parallel pre-resize of the used images (much faster loading)
    multi_gpu=True,              # DataParallel over both T4s when available
    target_precision=0.98,
    time_budget_h=TRAIN_HOURS,   # per session; training resumes from the last epoch next session
    zero_shot_baseline=True, resume=True,
)
if SMOKE_TEST:
    CFG.update(epochs=2, max_classes=80, max_per_class_train=8, P=16, time_budget_h=0.1)
"""


def nb04():
    return [
        md("""
# ShelfTwin 04: product embedder, DINOv2 ViT-S/14 + ArcFace (resumable)

256-d embedding for **catalogue retrieval with abstention**. New products are added with 3-5 photos and no retraining.

* Backbone: DINOv2 ViT-S/14 at 224 px (timm/HF Hub; torch.hub as fallback). The last 4 blocks are fine-tuned. ArcFace s = 64, m = 0.5 rad.
* **Speed:** the used images are resized once to 256 px in parallel and cached (later sessions skip the 500k-file scan);
  DataParallel over both T4s; AMP; all CPU cores for loading.
* **Open-set protocol:** SKUs are split 70 / 10 / 20 into train / val / test. Half the val and test SKUs are "new products" that must come back `unknown`.
  Thresholds are calibrated on val for ≥ 98 % accepted-match precision and reported on test, alongside a zero-shot DINOv2 baseline.
* **Resume:** state is saved every epoch (atomically); the next session continues from the last epoch.
""" + KAGGLE_CHECKLIST.format(accel="GPU T4 x2", smoke=10,
                              inputs="search **RP2K** and add `khyeh0719/rp2k-dataset` (any `<class>/<images>` dataset, or a CSV "
                                     "with image paths + labels, also works)") + """
Outputs: `outputs/embedder_best.pt`, `embedder.onnx`, `identity_thresholds.json`, `embedder_report.json`, `embedder_scores.png`.
"""),
        code("SMOKE_TEST = False\nTRAIN_HOURS = 2.5\n" + EMBED_CFG),
        code("""
import os, sys, subprocess
os.makedirs("/kaggle/working/src", exist_ok=True); sys.path.insert(0, "/kaggle/working/src")
try:
    import timm
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"])
import torch, torchvision, timm
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| timm", timm.__version__, "| GPUs", torch.cuda.device_count())
"""),
        writefile("/kaggle/working/src/kaggle_common.py", SRC / "kaggle_common.py"),
        writefile("/kaggle/working/src/embed_train.py", SRC / "embed_train.py"),
        code("""
import kaggle_common as kc, embed_train as et
kc.quiet_loggers()
kc.restore_previous_outputs("nb04")
kc.mark_session("nb04")
report = et.train(CFG)
"""),
        code("""
import json
if report.get("training_complete"):
    et.export_onnx(kc.WORK / "outputs")
print(json.dumps({k: report[k] for k in ("zero_shot_test", "test", "thresholds", "train_minutes", "training_complete")
                  if k in report}, indent=2))
from IPython.display import Image, display
p = kc.WORK / "outputs" / "embedder_scores.png"
if p.exists(): display(Image(str(p)))
print("ALL DONE" if report.get("training_complete") else "NOT FINISHED: attach this version's output and run again to resume.")
"""),
    ]


# ============================================================================ Kaggle 07
def nb07():
    files = {
        "/kaggle/working/st/shelftwin/__init__.py": ROOT / "shelftwin" / "__init__.py",
        "/kaggle/working/st/shelftwin/engine/__init__.py": ROOT / "shelftwin" / "engine" / "__init__.py",
        "/kaggle/working/st/shelftwin/engine/temporal.py": ROOT / "shelftwin" / "engine" / "temporal.py",
        "/kaggle/working/st/shelftwin/engine/baselines.py": ROOT / "shelftwin" / "engine" / "baselines.py",
        "/kaggle/working/st/sim/simulate.py": ROOT / "sim" / "simulate.py",
        "/kaggle/working/st/calibrate_engine.py": SRC / "calibrate_engine.py",
    }
    cells = [
        md("""
# ShelfTwin 07: temporal engine calibration and simulation

**CPU only: no GPU quota used** (*Accelerator: None*; Internet not needed). Uses all CPU cores in parallel.

1. Reproduces the comparison (single-frame, single-frame + mask, majority vote, ShelfTwin) on 20 simulated 8-hour runs (160 camera-hours).
2. Grid-searches `llr_empty`, `llr_stocked`, `act_threshold`, `decay_per_s` to minimise false alerts per camera-hour with recall ≥ 99 % and p95 delay ≤ 30 s.
   Tuning uses seeds 100-109; the result is reported on the untouched seeds 0-19. Finished configs are saved as they complete,
   so attaching this notebook's previous output resumes the search.
3. Scores an attached CSV of recorded outputs (`slot_id, ts, coverage, visibility, quality, needs_restock`) if present.
"""),
        code("""
SMOKE_TEST = False
import os, sys, shutil, glob
for d in ["/kaggle/working/st/shelftwin/engine", "/kaggle/working/st/sim", "/kaggle/working/outputs"]:
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, "/kaggle/working/st")
prev = glob.glob("/kaggle/input/**/engine_grid.jsonl", recursive=True)   # resume a previous search
if prev and not os.path.exists("/kaggle/working/outputs/engine_grid.jsonl"):
    shutil.copy(prev[0], "/kaggle/working/outputs/engine_grid.jsonl"); print("resuming from", prev[0])
"""),
    ]
    cells += [writefile(dst, src) for dst, src in files.items()]
    cells += [code(CALIB_BODY)]
    return cells


CALIB_BODY = r'''
import json, time
from pathlib import Path
from multiprocessing import Pool
from sim.simulate import SimConfig, run, print_table
from shelftwin.engine import EngineConfig
import calibrate_engine as ce
OUT = Path(os.environ.get("ST_WORK", "/kaggle/working")) / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
sim_cfg = SimConfig(hours=1.0) if SMOKE_TEST else SimConfig()
N_RUNS = 2 if SMOKE_TEST else 20
t0 = time.time()
base = run(N_RUNS, sim_cfg, verbose=False)
print_table(base); print(f"({time.time() - t0:.0f}s)")
grid = dict(llr_empty=[1.6], llr_stocked=[2.2], act_threshold=[3.5, 4.0], decay_per_s=[0.02]) if SMOKE_TEST else ce.GRID
t0 = time.time()
best, allres = ce.grid_search(sim_cfg, grid, seeds=ce.TUNE_SEEDS[:2] if SMOKE_TEST else ce.TUNE_SEEDS,
                              cache_path=OUT / ("engine_grid_smoke.jsonl" if SMOKE_TEST else "engine_grid.jsonl"))
print(f"{len(allres)} configs in {time.time() - t0:.0f}s on {os.cpu_count()} CPUs")
print("best on tuning seeds:", best["params"])
tuned = run(N_RUNS, sim_cfg, EngineConfig(**best["params"]), only=["ShelfTwin engine"], verbose=False)
print_table({"ShelfTwin (spec defaults)": base["ShelfTwin engine"], "ShelfTwin (tuned)": tuned["ShelfTwin engine"]})
rec = {}
for p in glob.glob("/kaggle/input/**/*.csv", recursive=True) + glob.glob("/content/inputs/**/*.csv", recursive=True):
    try:
        import pandas as pd
        if {"slot_id", "ts", "coverage", "visibility", "needs_restock"} <= set(pd.read_csv(p, nrows=1).columns):
            rec[p] = dict(default=ce.evaluate_recorded_csv(Path(p), EngineConfig()),
                          tuned=ce.evaluate_recorded_csv(Path(p), EngineConfig(**best["params"])))
            print(p, rec[p])
    except Exception as e:
        print("skip", p, e)
import matplotlib.pyplot as plt
names = list(base) + ["ShelfTwin (tuned)"]
vals = [base[n]["false_alerts_per_cam_hour"] for n in base] + [tuned["ShelfTwin engine"]["false_alerts_per_cam_hour"]]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh(names, vals); ax.set_xscale("log"); ax.set_xlabel("false alerts per camera-hour (log scale)")
for i, v in enumerate(vals): ax.text(v, i, f" {v:.2f}", va="center")
plt.tight_layout(); plt.savefig(OUT / "engine_false_alerts.png", dpi=120); plt.show()
json.dump(dict(baselines=base, tuned=tuned, best_params=best["params"], grid=grid, recorded=rec,
               sim_config=sim_cfg.__dict__, n_runs=N_RUNS), open(OUT / "engine_calibration.json", "w"), indent=2, default=str)
print("ALL DONE")
'''


# ============================================================================ Colab (all-in-one)
def code_bundle_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for base in ("shelftwin", "sim"):
            for p in sorted((ROOT / base).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".html"} \
                        and p.name != "results.json":
                    z.write(p, p.relative_to(ROOT).as_posix())
        for p in sorted(SRC.glob("*.py")):
            z.write(p, ("src/" + p.name))
        z.write(ROOT / "configs" / "engine.yaml", "configs/engine.yaml")
    return base64.b64encode(buf.getvalue()).decode()


def colab():
    b64 = code_bundle_b64()
    return [
        md("""
# ShelfTwin: the full pipeline on Google Colab (free tier, resumable)

One notebook runs everything: **engine simulation and calibration → SKU-110K data → pack+gap detector training →
product embedder training → end-to-end pipeline demo on a real shelf photo → live dashboard**.

### Free Colab reality
One T4 (not always available), sessions last at most ~12 h and often less, idle tabs disconnect after ~90 min,
~12 GB RAM, the disk is wiped every session. So:
* **Everything is saved to Google Drive** (`MyDrive/ShelfTwin`, about 4 GB): checkpoints, prepared datasets (as single tar
  files) and results.
* **After any disconnect: Runtime → Run all.** Finished stages are skipped, prepared data is restored from Drive in seconds,
  and training resumes from the last epoch with the same schedule.
* Each session trains for at most `DETECTOR_HOURS` + `EMBED_HOURS`, then runs the demo with the best weights so far,
  so you get results every session.

### Setup (once)
1. *Runtime → Change runtime type → **T4 GPU***.
2. **Kaggle token** (needed to download SKU-110K quickly and RP2K): kaggle.com → *Settings → API → Create New Token*.
   Then in Colab, the 🔑 *Secrets* sidebar → add `KAGGLE_USERNAME` and `KAGGLE_KEY` (toggle notebook access on).
   Without a token, SKU-110K is streamed from the official archive (slower) and the embedder stage is skipped
   (the demo then uses zero-shot DINOv2).
3. Optional: already trained on Kaggle? Put `pack_detector_sku110k.pt` / `pack_gap.pt` / `embedder_best.pt` /
   `identity_thresholds.json` into `MyDrive/ShelfTwin/models/` and they are used instead of training from scratch.
4. *Runtime → Run all*. Keep the tab open.
"""),
        code("""
# ================= configuration =================
SMOKE_TEST = False          # True: tiny data + 1-2 epochs everywhere (~20 min) to check the whole notebook
RUN = dict(engine=True, data=True, detector=True, embedder=True, demo=True, dashboard=True)
DETECTOR_EPOCHS = 30        # total schedule; spans sessions
DETECTOR_HOURS = 3.0        # detector training per session (resumes next session)
EMBED_HOURS = 1.0           # embedder training per session (resumes next session)
EMBED_EPOCHS = 12
MAX_TRAIN_IMAGES = 4000     # SKU-110K source photos used for detector training (x2 crops each)
TEST_IMAGES = 1000          # SKU-110K test photos kept for evaluation (all 2,936 = slower eval)
SRC_SIDE, IMGSZ = 896, 640
"""),
        code(r'''
import os, sys, time, json, subprocess, shutil
from pathlib import Path
try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/ShelfTwin"
except Exception as e:
    print("WARNING: Google Drive not mounted -> nothing persists across sessions:", e)
    BASE = "/content/ShelfTwin"
os.environ.update(ST_WORK=BASE, ST_TMP="/content/tmp", ST_INPUT_ROOT="/content/inputs")
for d in (BASE, BASE + "/models", BASE + "/outputs", "/content/tmp", "/content/inputs"):
    os.makedirs(d, exist_ok=True)
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2; df -h /content | tail -1",
                     shell=True, capture_output=True, text=True).stdout)
'''),
        code(ULTRA_CHECK + r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub", "timm", "fastapi", "uvicorn"])
'''),
        code("# ShelfTwin code bundle (shelftwin/, sim/, notebook helpers), built by notebooks/build_notebooks.py\n"
             "import base64, io, zipfile\n"
             f'CODE_B64 = "{b64}"\n'
             "zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall('/content/st')\n"
             "for p in ('/content/st', '/content/st/src'):\n"
             "    if p not in sys.path: sys.path.insert(0, p)\n"
             "import kaggle_common as kc, colab_data as cd\n"
             "kc.quiet_loggers()\n"
             "WORK, TMP, OUT = kc.WORK, kc.TMP, kc.WORK / 'outputs'\n"
             "PROGRESS = WORK / 'progress.json'\n"
             "progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}\n"
             "def save_progress(**kw):\n"
             "    progress.update(kw); PROGRESS.write_text(json.dumps(progress, indent=2, default=str))\n"
             "print('code ready | previous progress:', progress)\n"),
        md("## 1 · Temporal engine: simulation + calibration (CPU, parallel, resumable)"),
        code(r'''
if RUN["engine"] and not (OUT / "engine_calibration.json").exists():
    import glob
''' + CALIB_BODY.replace("\n", "\n    ") + r'''
    save_progress(engine="done")
else:
    print("engine stage: done earlier or disabled ->", OUT / "engine_calibration.json")
'''),
        md("## 2 · Data: SKU-110K → resized dataset → pack+gap crops (parallel, cached on Drive)"),
        code(r'''
import sku110k_prep as sp, gap_synth as gs
sfx = "_smoke" if SMOKE_TEST else ""
src_name = f"sku110k_{SRC_SIDE}_{MAX_TRAIN_IMAGES}_{TEST_IMAGES}{sfx}"
pg_name = f"packgap_{IMGSZ}_{MAX_TRAIN_IMAGES}{sfx}"
src_dir = pg_dir = None
if RUN["data"] or RUN["detector"] or RUN["demo"]:
    def build_src(d):
        cd.setup_kaggle_credentials()
        cd.kaggle_download(["thedatasith/sku110k-annotations"], "sku110k", min_free_gb=35)
        lim = dict(train=40, val=10, test=10) if SMOKE_TEST else dict(train=MAX_TRAIN_IMAGES, test=TEST_IMAGES)
        sp.prepare_sku110k(out=d, max_side=SRC_SIDE, limits=lim, allow_download=True)   # streams if no Kaggle copy
        cd.remove_download("sku110k")
    t0 = time.time()
    src_dir = kc.cached_dir(src_name, TMP / src_name, build_src)
    pg_dir = kc.cached_dir(pg_name, TMP / pg_name, lambda d: gs.build_pack_gap(src_dir, d, n_crops=2, out_size=IMGSZ, p_synth=0.6))
    print("data ready in", round(time.time() - t0), "s:", sp.count_split(src_dir), "| pack+gap:", sp.count_split(pg_dir))
    save_progress(data="done")
'''),
        md("## 3 · Pack + gap detector (YOLO11s, resumable)"),
        code(r'''
import yolo_runner as yr
DET_NAME = "packgap_colab" + ("_smoke" if SMOKE_TEST else "")
det_weights = None
pre = [WORK / "models" / f for f in ("pack_gap.pt",)]
if any(p.exists() for p in pre) and not SMOKE_TEST:
    det_weights = next(p for p in pre if p.exists())
    print("Using your Kaggle-trained pack+gap detector:", det_weights)
elif progress.get("detector") == "done" and (OUT / "pack_gap.pt").exists() and not SMOKE_TEST:
    det_weights = OUT / "pack_gap.pt"
    print("Detector finished in an earlier session:", json.loads((OUT / "metrics_detector.json").read_text())["metrics"])
elif RUN["detector"]:
    init = WORK / "models" / "pack_detector_sku110k.pt"
    train_args = dict(
        model=str(init) if init.exists() else "yolo11s.pt", data=str(pg_dir / "data.yaml"), imgsz=IMGSZ,
        epochs=2 if SMOKE_TEST else DETECTOR_EPOCHS, batch=16, batch_per_gpu=16, name=DET_NAME,
        optimizer="AdamW", lr0=0.001 if init.exists() else 0.002, cos_lr=True, warmup_epochs=1 if SMOKE_TEST else 2,
        mosaic=0.5, close_mosaic=0 if SMOKE_TEST else 5, fliplr=0.5, degrees=0.0, hsv_v=0.5,
        max_det=1000, amp=True, patience=10, seed=0, deterministic=False, workers=os.cpu_count(),
    )
    print("initial weights:", train_args["model"])
    res = yr.train(train_args, budget_h=0.08 if SMOKE_TEST else DETECTOR_HOURS, multi_gpu=False)
    det_weights = res["best"]
    split = "test" if res["finished"] else "val"
    metrics = yr.evaluate(det_weights, pg_dir / "data.yaml", imgsz=IMGSZ, split=split, max_det=1000, iou=0.6)
    gap = gs.gap_recall_at_precision(det_weights, pg_dir, split=split, target_p=0.9, imgsz=IMGSZ,
                                     max_images=100 if SMOKE_TEST else 600)
    if res["finished"]:
        yr.export_onnx(det_weights, IMGSZ)
    kc.save_json(dict(metrics=metrics, gap_recall_at_p90=gap, finished=res["finished"], epochs_done=res["epochs_done"]),
                 OUT / "metrics_detector.json")
    shutil.copy(det_weights, OUT / "pack_gap.pt")
    save_progress(detector="done" if res["finished"] else f"{res['epochs_done']} epochs")
    print(metrics, "\ngap recall @ P0.9:", gap)
else:
    p = yr.run_dir(DET_NAME) / "weights" / "best.pt"
    det_weights = p if p.exists() else None
print("detector weights:", det_weights)
'''),
        md("## 4 · Product embedder: DINOv2 + ArcFace on RP2K (resumable)"),
        code("TRAIN_HOURS = EMBED_HOURS\n" + EMBED_CFG + r'''
CFG.update(epochs=2 if SMOKE_TEST else EMBED_EPOCHS, out_dir=str(OUT), P=32)   # P=32 fits a single T4 comfortably
import embed_train as et
emb_ckpt = WORK / "models" / "embedder_best.pt"
if emb_ckpt.exists() and not SMOKE_TEST:
    print("Using your Kaggle-trained embedder:", emb_ckpt)
elif progress.get("embedder") == "done" and (OUT / "embedder_best.pt").exists() and not SMOKE_TEST:
    emb_ckpt = OUT / "embedder_best.pt"
    print("Embedder finished in an earlier session:", json.loads((OUT / "embedder_report.json").read_text()).get("test"))
elif RUN["embedder"]:
    have_cache = kc.find_cached(et.subset_key(CFG) + ".tar") is not None
    if not have_cache:
        cd.setup_kaggle_credentials()
        cd.kaggle_download(["khyeh0719/rp2k-dataset", "khyeh0719/guie-rp2k"], "rp2k", min_free_gb=20)
    try:
        report = et.train(CFG)
        emb_ckpt = OUT / "embedder_best.pt"
        if report.get("training_complete"):
            et.export_onnx(OUT)
            cd.remove_download("rp2k")
        print(json.dumps({k: report.get(k) for k in ("zero_shot_test", "test", "thresholds", "training_complete")}, indent=2))
        save_progress(embedder="done" if report.get("training_complete") else "partial")
    except FileNotFoundError as e:
        print("Embedder skipped (no RP2K; add a Kaggle token):", e)
        emb_ckpt = None
else:
    emb_ckpt = OUT / "embedder_best.pt" if (OUT / "embedder_best.pt").exists() else None
print("embedder checkpoint:", emb_ckpt or "none -> zero-shot DINOv2")
'''),
        md("""
## 5 · End-to-end pipeline demo on a real shelf photo

All stages run on every frame with **your trained models**: quality gate → change gate → person mask → pack+gap
detector → geometry/coverage → identity (embedder + catalogue, abstains) → planogram alignment → temporal engine → tasks.
The scripted 8-minute episode: slot **A** runs out and is refilled, a shopper blocks **B**, glare, and products **C/D** are
swapped. The shopper is a drawn silhouette, so its mask is known (oracle with a 5 % miss rate).
"""),
        code(r'''
if RUN["demo"] and det_weights is not None:
    import cv2, numpy as np
    import e2e_demo as ed
    from shelftwin.vision.detector import PackGapDetector, DetectorConfig
    from shelftwin.vision.identity import Embedder
    det = PackGapDetector(DetectorConfig(weights=str(det_weights), imgsz=IMGSZ, device=0, half=True, conf=0.25))
    emb = Embedder(str(emb_ckpt) if emb_ckpt else None)
    th_path = next((p for p in (WORK / "models" / "identity_thresholds.json", OUT / "identity_thresholds.json") if p.exists()), None)
    th = json.loads(th_path.read_text()) if th_path else None
    # dense real test photos first
    lbl = src_dir / "labels" / "test"
    cands = sorted(lbl.glob("*.txt"), key=lambda p: -len(p.read_text().splitlines()))
    cands = [p for p in cands if 60 <= len(p.read_text().splitlines()) <= 260][:8] or cands[:8]
    demo = None
    for p in cands:
        img = cv2.imread(str(src_dir / "images" / "test" / (p.stem + ".jpg")))
        try:
            demo = ed.run_demo(ed.square_crop(img), det, embedder=emb, thresholds=th, seed=0,
                               video_path="/content/demo_raw.mp4", n_frames=60 if SMOKE_TEST else ed.N_FRAMES)
            break
        except RuntimeError as e:
            print("skip", p.stem, e)
    if demo:
        s = demo["summary"]
        print(json.dumps({k: v for k, v in s.items() if k != "tasks"}, indent=2))
        for t in s["tasks"]: print("  task:", t)
        ed.plot_trace(demo, str(OUT / "e2e_trace.png"))
        kc.save_json(s, OUT / "e2e_summary.json")
        from IPython.display import Image, HTML, display
        display(Image(str(OUT / "e2e_trace.png")))
        mp4 = OUT / "e2e_demo.mp4"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/content/demo_raw.mp4", "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "28", str(mp4)])
        if r.returncode == 0:
            display(HTML(f'<video controls width="640" src="data:video/mp4;base64,{base64.b64encode(mp4.read_bytes()).decode()}"></video>'))
        save_progress(demo="done")
else:
    print("demo skipped (no detector weights yet or disabled)")
'''),
        md("## 6 · Live dashboard (simulated store) inside Colab"),
        code(r'''
if RUN["dashboard"]:
    import threading, uvicorn, yaml
    from shelftwin.api.server import create_app
    from shelftwin.engine import EngineConfig
    from shelftwin.twin import Twin
    from sim.live import LiveStore
    cal = OUT / "engine_calibration.json"
    ecfg = EngineConfig(**json.loads(cal.read_text())["best_params"]) if cal.exists() else EngineConfig()
    live = LiveStore()
    twin = Twin(live.infos, engine_cfg=ecfg)
    app = create_app(twin, on_done=live.refill)
    def sim_loop(speed=20.0):
        while True:
            t0 = time.time()
            for sid, ts, cov, vis, q in live.step():
                twin.observe(sid, ts, cov, vis, q)
            twin.tick(live.t)
            time.sleep(max(0.0, live.dt / speed - (time.time() - t0)))
    threading.Thread(target=sim_loop, daemon=True).start()
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(3)
    try:
        from google.colab import output
        output.serve_kernel_port_as_window(8000)   # opens the dashboard in a new tab
    except Exception as e:
        print("Open http://localhost:8000 (not running in Colab?)", e)
'''),
        code(r'''
print(json.dumps(progress, indent=2))
unfinished = [k for k in ("detector", "embedder") if RUN.get(k) and progress.get(k) not in ("done", None)]
print("ALL DONE" if not unfinished else f"Training continues next session for {unfinished}: Runtime -> Run all.")
print("Results and weights are in Google Drive:", OUT)
'''),
    ]


# ============================================================================ Kaggle: final all-in-one
FINAL_CONFIG = r'''
# ======================= configuration (edit here) =======================
SMOKE = False               # True: tiny data, 1-2 epochs everywhere (~30 min) to check the whole notebook first
SESSION_HOURS = 11.3        # Kaggle hard limit is 12 h
RESERVE_H = 0.4             # kept for saving outputs at the end of a session

CFG = dict(
    SMOKE=SMOKE, SFX="_smoke" if SMOKE else "", SESSION_HOURS=SESSION_HOURS, RESERVE_H=RESERVE_H,
    SRC_SIDE=1024,                                    # SKU-110K images resized once to this long side
    EXP_SUBSET=dict(train=1500, val=300, test=500),   # small subset for experiments
    FULL_SUBSET={},                                   # {} = full SKU-110K for final training
    # ---- detector experiments: one run per GPU in parallel, short schedule on the subset
    # Session-1 EDA: 40.7% of packs are < 16 px at 640, 17.7% at 896, 11.7% at 1024 -> 640 dropped, 1024 tested.
    DET_EXP_EPOCHS=10, SPEED_TOL=0.01,                # within 0.01 mAP50-95 of the best, prefer the faster model
    DET_RUNS=[                                        # longest first: balances the two GPUs
        dict(name="d_y11s_1024", model="yolo11s.pt", imgsz=1024, mosaic=1.0, batch=6, batch_per_gpu=6),
        dict(name="d_y11s_896", model="yolo11s.pt", imgsz=896, mosaic=1.0, batch=8, batch_per_gpu=8),
        dict(name="d_y11s_896_nomosaic", model="yolo11s.pt", imgsz=896, mosaic=0.0, batch=8, batch_per_gpu=8),
        dict(name="d_y11n_1024", model="yolo11n.pt", imgsz=1024, mosaic=1.0, batch=12, batch_per_gpu=12),
    ],
    # ---- gap-label experiments (real geometric gaps vs synthetic erasing, fill styles)
    GAP_RUNS=[
        dict(name="real_only", p_synth=0.0, fill=[0.35, 0.45, 0.20]),
        dict(name="mixed", p_synth=0.6, fill=[0.35, 0.45, 0.20]),
        dict(name="shelfback_only", p_synth=0.6, fill=[0.0, 1.0, 0.0]),
    ],
    GAP_EXP_TRAIN=1200, GAP_EXP_EPOCHS=8, GAP_FULL_TRAIN=6000,
    DEFAULT_GAP=dict(p_synth=0.6, fill=[0.35, 0.45, 0.20]),   # pre-built in the CPU session; rebuilt only if the experiment disagrees
    # ---- embedder experiments
    EMBED_BASE=dict(seed=0, dim=256, unfreeze_blocks=4, arc_scale=64.0, arc_margin=0.5, lr_backbone=1e-5, lr_head=1e-3,
                    weight_decay=0.05, P=48, K=4, workers=None, cache_subset=True, cache_side=256, multi_gpu=True,
                    target_precision=0.98, zero_shot_baseline=True, resume=True, time_budget_h=1.0),
    EMBED_EXP=dict(max_classes=600, max_per_class_train=20, epochs=4),
    EMBED_FINAL=dict(max_classes=None, max_per_class_train=40, epochs=12),
    EMBED_ZERO_SHOT=["vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m"],
    EMBED_RUNS=[
        dict(name="head_only", unfreeze_blocks=0, lr_head=2e-3),
        dict(name="last2", unfreeze_blocks=2),
        dict(name="last4", unfreeze_blocks=4),
    ],
    # ---- final training (uses the experiment decisions)
    # Deadline plan: everything left fits in ONE GPU session (~8 h). Final detectors are time-boxed:
    # Ultralytics re-plans epochs and the LR schedule to finish in *_TIME_H hours.
    DET_FINAL_EPOCHS=40, DET_FINAL_TIME_H=3.0, DET_FINAL_MAX_H=3.4,
    PG_FINAL_EPOCHS=25, PG_FINAL_TIME_H=1.5, PG_FINAL_MAX_H=1.9,
    EMB_FINAL_MAX_H=0.9,
)
if SMOKE:
    CFG.update(EXP_SUBSET=dict(train=40, val=10, test=10), FULL_SUBSET=dict(train=60, val=10, test=10),
               DET_EXP_EPOCHS=2, DET_RUNS=CFG["DET_RUNS"][:2], GAP_RUNS=CFG["GAP_RUNS"][:2], GAP_EXP_TRAIN=30,
               GAP_EXP_EPOCHS=2, GAP_FULL_TRAIN=40, EMBED_EXP=dict(max_classes=60, max_per_class_train=6, epochs=2),
               EMBED_FINAL=dict(max_classes=80, max_per_class_train=6, epochs=2), EMBED_ZERO_SHOT=CFG["EMBED_ZERO_SHOT"][:1],
               EMBED_RUNS=CFG["EMBED_RUNS"][:2], DET_FINAL_EPOCHS=2, PG_FINAL_EPOCHS=2,
               DET_FINAL_TIME_H=None, PG_FINAL_TIME_H=None)
    CFG["EMBED_BASE"].update(P=16)
'''


def kaggle_final():
    b64 = code_bundle_b64()
    show = r'''
from IPython.display import Image, Markdown, display, HTML
def show(*names):
    for n in names:
        p = OUT / n
        if p.exists():
            display(Image(str(p))) if p.suffix == ".png" else print(p.read_text()[:3000])
'''
    def phase_cell(name, fn, gpu, extra=""):
        return code(f'fp.run_phase(prog, "{name}", fp.{fn}, CFG, needs_gpu={gpu})\n' + extra)
    return [
        md("""
# ShelfTwin: complete Kaggle pipeline (EDA → experiments → final training → evaluation → demo)

One notebook does the whole project within the free **30 GPU-hours/week**. It works in **phases across several
sessions**. Each run does what it can, saves everything to its output, and the next run continues from there.

| Session | Accelerator | What runs | GPU quota |
|---|---|---|---|
| 1 | **None (CPU)** | data preparation (all of it, cached), EDA, engine simulation and error analysis | **0 h** |
| 2 | **GPU T4 x2** | everything else in one session (deadline plan): experiments (4 detector configs, 3 gap-label strategies, 3 embedder strategies + zero-shot ViT-S vs ViT-B, each with error analysis) -> final detector (3 h, time-boxed) -> final pack+gap detector (1.5 h) -> final embedder -> evaluation, demo video, report | ~8 h |
| 3 | GPU T4 x2 (only if needed) | anything session 2 did not finish (it resumes) | as needed |

Total: about 8 GPU h after the free CPU session. Phases are idempotent: finished ones are skipped,
and interrupted training resumes from its last epoch.

### Every session
1. *Add Input* → datasets: **`thedatasith/sku110k-annotations`** (SKU-110K) and **`khyeh0719/rp2k-dataset`** (RP2K).
2. From session 2 on, also *Add Input → Your Work → Notebooks →* **this notebook** (its latest version). That's how
   caches, checkpoints and decisions carry over.
3. Pick the accelerator from the table, set *Internet: On*, then **Save Version → Save & Run All (Commit)**.
4. The last cell prints what is done and what to run next.

First time? Set `SMOKE = True` and *Run All* once with a GPU (~30 min) to check everything end to end.
Then set it back to `False` and start at session 1.
"""),
        code(FINAL_CONFIG),
        code(ULTRA_CHECK + r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)
'''),
        code("# ShelfTwin code (shelftwin/, sim/, helpers), built from the tested repository by build_notebooks.py\n"
             "import base64, io, zipfile, os, sys, json\n"
             f'CODE_B64 = "{b64}"\n'
             "zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall('/kaggle/working/st')\n"
             "for p in ('/kaggle/working/st', '/kaggle/working/st/src'):\n"
             "    if p not in sys.path: sys.path.insert(0, p)\n"
             "import kaggle_common as kc, final_pipeline as fp\n"
             "kc.quiet_loggers()\n"
             "TAG = 'final' + CFG['SFX']   # smoke-test outputs are never restored into a real run\n"
             "kc.restore_previous_outputs(TAG, subdirs=('runs', 'outputs', 'cache', 'exp_embed'))\n"
             "kc.mark_session(TAG)\n"
             "OUT = fp.OUT; OUT.mkdir(parents=True, exist_ok=True)\n"
             "prog = fp.Progress(CFG['SFX'])\n"
             "kc.log('GPUs:', kc.gpu_info() or 'none (CPU session)', '| CPUs:', os.cpu_count(),\n"
             "       '| free disk GB:', round(kc.disk_free_gb('/kaggle/working'), 1))\n"
             "kc.log('progress so far:', prog.summary())\n"
             "kc.print_input_tree(max_depth=2)\n" + show),
        md("## Phase 1 · Data preparation (CPU, parallel, cached)\nResizes SKU-110K once (all CPU cores), builds the pack+gap crops for every experiment, and pre-resizes the RP2K subsets. All of it is cached in the output, so later sessions restore it in seconds."),
        phase_cell("data", "phase_data", False, "print(prog.d.get('data'))"),
        md("## Phase 2 · Exploratory data analysis (CPU)\nBoxes per image (→ `max_det`), pack size in pixels at each training resolution (→ `imgsz`), real-gap statistics (→ gap labels), RP2K class balance."),
        phase_cell("eda", "phase_eda", False,
                   "show('eda/eda_sku110k.png', 'eda/eda_sku110k_samples.png', 'eda/eda_rp2k.png')\n"
                   "for k, v in (prog.d.get('eda') or {}).items():\n    if isinstance(v, list): print(k, *v, sep='\\n  ')"),
        md("## Phase 3 · Temporal engine: sensitivity, error analysis, calibration (CPU)\nHow false alerts grow as each noise source gets worse; which noise causes the remaining false alerts; parameters tuned on held-out seeds."),
        phase_cell("engine", "phase_engine", False, "show('engine_analysis.png')\nprint(prog.d.get('engine'))"),
        md("## Phase 4 · Detector experiments (GPU, one run per GPU in parallel)\nModel size × resolution × mosaic on a 1,500-image subset, 15 epochs each. The winner is the best mAP50-95, preferring the faster model within 0.01. Then error analysis on the winner: recall by pack size and shelf density, counting error, false-positive types, worst images, NMS/`max_det` sweep, and tiled vs full-frame inference."),
        phase_cell("exp_detector", "phase_exp_detector", True,
                   "show('exp_detector_worst.png')\n"
                   "p = OUT / 'exp_detector.json'\n"
                   "if p.exists():\n    r = json.loads(p.read_text())\n"
                   "    print(json.dumps(dict(winner=r['winner'], error_analysis=r['error_analysis'], sweep=r['eval_sweep'],\n"
                   "                          tiled=r['tiled_vs_full_at_640']), indent=1, default=str)[:4000])"),
        md("## Phase 5 · Gap-label experiments (GPU, parallel)\nReal geometric gaps only vs mixed vs synthetic-heavy vs shelf-back-only fill. Evaluated on **real** test photos only: gap AP50 and gap recall at precision 0.9."),
        phase_cell("exp_gap", "phase_exp_gap", True, "show('exp_gap.json')"),
        md("## Phase 6 · Embedder experiments (GPU)\nZero-shot DINOv2 ViT-S vs ViT-B, then ArcFace fine-tuning (head only, last 2 blocks, last 4 blocks, smaller margin) on a 600-SKU subset. Open-set evaluation, then error analysis on the winner: most-confused SKU pairs, precision/coverage trade-off, failure gallery."),
        phase_cell("exp_embed", "phase_exp_embed", True,
                   "show('exp_embed_errors/identity_tradeoff.png', 'exp_embed_errors/identity_failures.png')\n"
                   "p = OUT / 'exp_embed.json'\n"
                   "if p.exists():\n    r = json.loads(p.read_text())\n"
                   "    print(json.dumps(dict(zero_shot=r['zero_shot'], winner=r['winner'],\n"
                   "                          most_confused=r['errors']['most_confused'][:8]), indent=1, ensure_ascii=False)[:4000])"),
        md("## Phase 7 · Final detector on full SKU-110K (GPU, DDP on both T4s, resumable)\nUses `decision_detector.json` from phase 4."),
        phase_cell("final_detector", "phase_final_detector", True),
        md("## Phase 8 · Final pack + gap detector (GPU, resumable)\nStarts from the phase-7 weights; gap labels built the way phase 5 decided."),
        phase_cell("final_packgap", "phase_final_packgap", True),
        md("## Phase 9 · Final embedder on full RP2K (GPU, both T4s, resumable)\nUses `decision_embedder.json` from phase 6."),
        phase_cell("final_embed", "phase_final_embed", True,
                   "show('final_embed_errors/identity_tradeoff.png')\nprint(prog.d.get('final_embed'))"),
        md("## Phase 10 · Final evaluation, error analysis, end-to-end demo, report (GPU)"),
        phase_cell("final_eval", "phase_final_eval", True,
                   "show('final_detector_worst.png', 'e2e_trace.png')\n"
                   "v = OUT / 'e2e_demo.mp4'\n"
                   "if v.exists():\n"
                   "    display(HTML(f'<video controls width=\"640\" src=\"data:video/mp4;base64,{base64.b64encode(v.read_bytes()).decode()}\"></video>'))\n"
                   "if (OUT / 'REPORT.md').exists(): display(Markdown((OUT / 'REPORT.md').read_text(encoding='utf-8')))"),
        code(r'''
order = ["data", "eda", "engine", "exp_detector", "exp_gap", "exp_embed", "final_detector", "final_packgap", "final_embed", "final_eval"]
st = fp.Progress(CFG['SFX']).summary()
for k in order:
    print(f"{k:16s} {st.get(k, 'not started')}")
todo = [k for k in order if st.get(k) != "done"]
gpu_needed = [k for k in todo if k not in ("data", "eda", "engine")]
if not todo:
    print("\nALL DONE. Models: outputs/pack_gap.pt, outputs/pack_detector_sku110k.pt, outputs/embedder/; report: outputs/REPORT.md")
else:
    cpu_left = [k for k in todo if k in ("data", "eda", "engine")]
    nxt = "None (CPU) - free, no GPU quota" if cpu_left else "GPU T4 x2"
    print(f"\nNEXT SESSION: attach this notebook's latest version as input; accelerator = {nxt}.")
    print("Remaining phases:", todo)
'''),
    ]


def save(cells, name, gpu: bool, colab: bool = False):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    if colab:
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "T4"}}
    else:
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isInternetEnabled": True,
                                  "isGpuEnabled": gpu}}
    nbf.validate(nb)
    path = HERE / name
    nbf.write(nb, path)
    print("wrote", path, f"({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    save(nb01(), "01_sku110k_pretrain.ipynb", True)
    save(nb02(), "02_pack_gap_finetune.ipynb", True)
    save(nb04(), "04_embedder_arcface.ipynb", True)
    save(nb07(), "07_engine_calibration.ipynb", False)
    save(colab(), "ShelfTwin_Colab_full.ipynb", True, colab=True)
    save(kaggle_final(), "ShelfTwin_Kaggle_final.ipynb", True)
