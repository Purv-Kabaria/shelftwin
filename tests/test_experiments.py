"""Light tests for the experiment-phase helpers (EDA, error analysis, decisions, sensitivity)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "src"))

from test_notebook_helpers import fake_kaggle, _shelf_image  # noqa: E402,F401


def _write_sku(inp, n_train=30):
    import cv2
    ds = inp / "sku" / "SKU110K_fixed"
    (ds / "images").mkdir(parents=True)
    (ds / "annotations").mkdir(parents=True)
    for split, n in (("train", n_train), ("val", 5), ("test", 5)):
        rows = []
        for i in range(n):
            img, boxes = _shelf_image(rng=np.random.default_rng(i), skip={(1, 1 + i % 6)})  # interior gaps
            name = f"{split}_{i}.jpg"
            cv2.imwrite(str(ds / "images" / name), img)
            rows += [f"{name},{b[0]},{b[1]},{b[2]},{b[3]},object,640,480" for b in boxes]
        (ds / "annotations" / f"annotations_{split}.csv").write_text("\n".join(rows) + "\n")
    return ds


def test_sku_eda(fake_kaggle):
    tmp, inp = fake_kaggle
    _write_sku(inp)
    import eda
    import sku110k_prep as sp
    by_split, _ = sp.find_csv_source(inp)
    rep = eda.sku_eda(by_split, tmp / "eda", gap_sample=20)
    assert rep["boxes_per_image"]["max"] == 23 and rep["images_over_300_boxes"] == 0
    assert rep["real_gaps_per_image"]["share_with_gap"] == 1.0  # one pack removed per image
    assert rep["rows_per_image"]["median"] == 3
    assert set(rep["box_short_side_after_resize"]) == {640, 896, 1024}
    assert (tmp / "eda" / "eda_sku110k.png").exists() and rep["recommendations"]
    eda.show_sku_samples(by_split, tmp / "eda", n=2)
    assert (tmp / "eda" / "eda_sku110k_samples.png").exists()


def test_rp2k_eda(fake_kaggle, tmp_path):
    from PIL import Image
    import eda
    items = []
    for c in range(30):
        for i in range(2 + c % 10):
            p = tmp_path / f"{c}_{i}.jpg"
            Image.new("RGB", (60 + c, 90)).save(p)
            items.append((str(p), f"c{c}"))
    rep = eda.rp2k_eda(items, tmp_path / "out", sample=50)
    assert rep["classes"] == 30 and rep["per_class"]["min"] == 2 and 0 <= rep["gini"] < 1


def test_match_and_summary():
    import det_analysis as da
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], float)
    pred = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [60, 0, 70, 10]], float)
    conf = np.array([0.9, 0.8, 0.7])
    tp, matched = da.match(pred, conf, gt)
    assert tp.tolist() == [True, False, False] and matched.tolist() == [True, False, False]
    rec = dict(image="x", w=100, h=100, gt=gt, pred=pred, conf=conf, tp=tp, matched=matched,
               gt_rel_size=np.full(3, 0.1))
    s = da.summarize([rec])
    assert abs(s["recall"] - 1 / 3) < 1e-9 and s["false_positive_types"] == dict(duplicate_or_misaligned=1, background=1)
    assert s["count_error"]["bias"] == 0.0


def test_decisions_roundtrip(fake_kaggle):
    tmp, inp = fake_kaggle
    import kaggle_common as kc
    assert kc.load_decision("detector") == {}
    prev = inp / "exp-output" / "outputs"
    prev.mkdir(parents=True)
    (prev / "decision_detector.json").write_text(json.dumps(dict(decision=dict(imgsz=896, model="yolo11s.pt"))))
    assert kc.load_decision("detector")["imgsz"] == 896
    kc.save_decision("gap", dict(p_synth=0.6), dict(table=[]))
    assert kc.load_decision("gap") == dict(p_synth=0.6)


def test_pick_prefers_faster_within_tolerance():
    import ablation as ab
    rows = [dict(name="a", mAP50_95=0.60, latency_ms_t4_fp16=20.0),
            dict(name="b", mAP50_95=0.605, latency_ms_t4_fp16=40.0),
            dict(name="c", mAP50_95=0.55, latency_ms_t4_fp16=5.0)]
    assert ab.pick(rows)["name"] == "a"
    assert ab.pick(rows, tol=0.001)["name"] == "b"
    assert "mAP50_95" in ab.table(rows, ["name", "mAP50_95"])


def test_engine_sensitivity_tiny():
    import calibrate_engine as ce
    from sim.simulate import SimConfig
    rows = ce.sensitivity(SimConfig(hours=0.5, n_slots=6), dict(p_mask_miss=[0.0, 0.3]), seeds=(0,), workers=1)
    assert len(rows) == 2 and "ShelfTwin engine" in rows[0]
    att = ce.attribute_false_alerts(sim_cfg=SimConfig(hours=0.5, n_slots=6), seeds=(0,))
    assert "cause_counts" in att and att["n_false"] >= 0
