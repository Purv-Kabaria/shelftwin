"""CPU phases of the final Kaggle notebook on tiny fake inputs (no downloads, no GPU)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "src"))

from test_notebook_helpers import fake_kaggle, _shelf_image  # noqa: E402,F401
from test_experiments import _write_sku  # noqa: E402


def _cfg():
    base = dict(seed=0, dim=256, unfreeze_blocks=4, arc_scale=64.0, arc_margin=0.5, lr_backbone=1e-5, lr_head=1e-3,
                weight_decay=0.05, P=16, K=4, workers=0, cache_subset=True, cache_side=64, multi_gpu=False,
                target_precision=0.98, zero_shot_baseline=True, resume=True, time_budget_h=1.0)
    return dict(SMOKE=True, SFX="_smoke", SESSION_HOURS=1.0, RESERVE_H=0.0, SRC_SIDE=320,
                EXP_SUBSET=dict(train=10, val=3, test=3), FULL_SUBSET=dict(train=20, val=5, test=5),
                GAP_RUNS=[dict(name="real_only", p_synth=0.0, fill=[0.35, 0.45, 0.2]),
                          dict(name="mixed", p_synth=0.6, fill=[0.35, 0.45, 0.2])],
                GAP_EXP_TRAIN=8, GAP_FULL_TRAIN=15, DEFAULT_GAP=dict(p_synth=0.6, fill=[0.35, 0.45, 0.2]),
                EMBED_BASE=base, EMBED_EXP=dict(max_classes=110, max_per_class_train=6, epochs=1),
                EMBED_FINAL=dict(max_classes=None, max_per_class_train=8, epochs=1))


def _write_rp2k(inp):
    from PIL import Image
    root = inp / "rp2k-dataset" / "all" / "train"
    rng = np.random.default_rng(0)
    for c in range(110):
        d = root / f"prod_{c}"
        d.mkdir(parents=True)
        col = tuple(int(x) for x in rng.integers(0, 255, 3))
        for i in range(7):
            Image.new("RGB", (40, 60), col).save(d / f"{i}.jpg")


def test_cpu_phases_and_gpu_deferral(fake_kaggle):
    tmp, inp = fake_kaggle
    _write_sku(inp, n_train=25)
    _write_rp2k(inp)
    import final_pipeline as fp
    import kaggle_common as kc
    kc.gpu_info = lambda: []          # CPU session
    cfg = _cfg()
    prog = fp.Progress(cfg["SFX"])
    assert fp.run_phase(prog, "data", fp.phase_data, cfg, needs_gpu=False)
    info = prog.d["data"]
    assert info["sku_exp"]["train"] == 10 and info["sku_full"]["train"] == 20 and info["rp2k"]
    # caches written for the next session
    names = {p.name for p in (kc.WORK / "cache").glob("*.tar")}
    assert {"sku110k_exp_smoke.tar", "sku110k_full_smoke.tar", "pg_full_default_smoke.tar"} <= names
    assert any(n.startswith("rp2k_subset_") for n in names)
    assert fp.run_phase(prog, "eda", fp.phase_eda, cfg, needs_gpu=False)
    assert (fp.OUT / "eda" / "eda_sku110k.png").exists() and (fp.OUT / "eda" / "eda_rp2k.png").exists()
    assert fp.run_phase(prog, "engine", fp.phase_engine, cfg, needs_gpu=False)
    assert (fp.OUT / "engine_analysis.png").exists()
    assert json.loads((fp.OUT / "decision_engine_smoke.json").read_text())["decision"]
    assert not (fp.OUT / "decision_engine.json").exists() and (fp.OUT / "progress_final_smoke.json").exists()
    # GPU phases are deferred, not failed
    assert not fp.run_phase(prog, "exp_detector", fp.phase_exp_detector, cfg, needs_gpu=True)
    assert prog.d["exp_detector"]["status"] == "waiting_for_gpu"
    # second run of a done phase is a no-op
    assert fp.run_phase(prog, "data", lambda c, b: (_ for _ in ()).throw(AssertionError("re-ran")), cfg, False)
    # a new session restores data from cache instead of rebuilding
    import shutil
    shutil.rmtree(kc.TMP)
    d = fp.sku_full(cfg)
    assert (d / ".complete").exists() and (d / "data.yaml").exists()
    fp.write_report(cfg, dict(note="x"))
    assert "engine decision" in (fp.OUT / "REPORT.md").read_text(encoding="utf-8")


def _write_sku_yolo(inp, n_train=25):
    """A YOLO-format SKU copy (no annotation CSVs), like the Kaggle dataset actually attached."""
    import cv2
    ds = inp / "sku-yolo" / "SKU110K"
    for split, n in (("train", n_train), ("val", 5), ("test", 5)):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for i in range(n):
            img, boxes = _shelf_image(rng=np.random.default_rng(i), skip={(1, 1 + i % 6)})
            cv2.imwrite(str(ds / "images" / split / f"{split}_{i}.jpg"), img)
            (ds / "labels" / split / f"{split}_{i}.txt").write_text("\n".join(
                f"0 {(b[0] + b[2]) / 2 / 640:.6f} {(b[1] + b[3]) / 2 / 480:.6f} {(b[2] - b[0]) / 640:.6f} {(b[3] - b[1]) / 480:.6f}"
                for b in boxes) + "\n")


def test_yolo_format_eda_and_subset_from_full(fake_kaggle):
    tmp, inp = fake_kaggle
    _write_sku_yolo(inp)
    _write_rp2k(inp)
    import final_pipeline as fp
    import kaggle_common as kc
    import sku110k_prep as sp
    kc.gpu_info = lambda: []
    cfg = _cfg()
    assert sp.find_csv_source() is None and sp.find_yolo_source() is not None
    prog = fp.Progress(cfg["SFX"])
    assert fp.run_phase(prog, "data", fp.phase_data, cfg, needs_gpu=False)
    exp, full = kc.TMP / "sku110k_exp_smoke", kc.TMP / "sku110k_full_smoke"
    exp_names = {p.name for p in (exp / "images" / "train").glob("*.jpg")}
    full_names = {p.name for p in (full / "images" / "train").glob("*.jpg")}
    assert len(exp_names) == 10 and exp_names <= full_names        # subset cut from the full set
    assert len(list((exp / "labels" / "train").glob("*.txt"))) == 10
    assert fp.run_phase(prog, "eda", fp.phase_eda, cfg, needs_gpu=False)
    rep = json.loads((fp.OUT / "eda" / "eda_sku110k.json").read_text())
    assert rep["source"].startswith("YOLO labels") and rep["boxes_per_image"]["max"] == 23
    assert rep["real_gaps_per_image"]["share_with_gap"] == 1.0
    assert (fp.OUT / "eda" / "eda_sku110k_samples.png").exists()
    # the RP2K directory scan happened once and is cached for every later phase/session
    assert (kc.WORK / "cache" / "rp2k_index.json.gz").exists()
    import embed_train as et
    calls = []
    orig = et.discover_folder_dataset
    et.discover_folder_dataset = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        assert len(et.discover_dataset()) == 770 and calls == []    # served from the cached index
    finally:
        et.discover_folder_dataset = orig
