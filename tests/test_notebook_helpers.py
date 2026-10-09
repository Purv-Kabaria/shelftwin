"""Light tests of the Kaggle notebook helpers on tiny synthetic data (no GPU, no downloads)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "notebooks" / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture()
def fake_kaggle(tmp_path, monkeypatch):
    inp = tmp_path / "input"
    monkeypatch.setenv("ST_INPUT_ROOT", str(inp))
    monkeypatch.setenv("ST_WORK", str(tmp_path / "working"))
    monkeypatch.setenv("ST_TMP", str(tmp_path / "tmp"))
    for m in ["kaggle_common", "sku110k_prep", "gap_synth", "embed_train", "final_pipeline", "eda", "det_analysis",
              "ablation", "yolo_runner", "calibrate_engine", "colab_data", "e2e_demo"]:
        sys.modules.pop(m, None)
    return tmp_path, inp


def _shelf_image(w=640, h=480, rows=3, per_row=8, rng=None, skip=()):
    rng = rng or np.random.default_rng(0)
    img = np.full((h, w, 3), 200, np.uint8)
    boxes = []
    rh = h // rows
    pw = w // per_row
    for r in range(rows):
        cv2.rectangle(img, (0, (r + 1) * rh - 6), (w, (r + 1) * rh), (90, 90, 90), -1)
        for k in range(per_row):
            if (r, k) in skip:
                continue
            x1, y1, x2, y2 = k * pw + 4, r * rh + 15, (k + 1) * pw - 4, (r + 1) * rh - 8
            img[y1:y2, x1:x2] = rng.integers(0, 225, 3)  # no saturated (glare-like) packs
            boxes.append([x1, y1, x2, y2])
    return img, boxes


def test_sku_csv_flow(fake_kaggle):
    tmp, inp = fake_kaggle
    ds = inp / "sku110k-annotations" / "SKU110K_fixed"
    (ds / "images").mkdir(parents=True)
    (ds / "annotations").mkdir(parents=True)
    rows = {"train": [], "val": [], "test": []}
    for split, n in (("train", 12), ("val", 3), ("test", 3)):
        for i in range(n):
            img, boxes = _shelf_image(rng=np.random.default_rng(i))
            name = f"{split}_{i}.jpg"
            cv2.imwrite(str(ds / "images" / name), img)
            for b in boxes:
                rows[split].append(f"{name},{b[0]},{b[1]},{b[2]},{b[3]},object,640,480")
    for split, r in rows.items():
        (ds / "annotations" / f"annotations_{split}.csv").write_text("\n".join(r) + "\n")
    import sku110k_prep as sp
    y = sp.prepare_sku110k(out=tmp / "tmp" / "sku", max_side=320, workers=1)
    assert sp.count_split(tmp / "tmp" / "sku") == {"train": 12, "val": 3, "test": 3}
    im = cv2.imread(str(tmp / "tmp" / "sku" / "images" / "train" / "train_0.jpg"))
    assert max(im.shape[:2]) == 320
    lines = (tmp / "tmp" / "sku" / "labels" / "train" / "train_0.txt").read_text().split("\n")
    assert len([l for l in lines if l]) == 24
    vals = np.array([[float(v) for v in l.split()[1:]] for l in lines if l])
    assert (vals >= 0).all() and (vals <= 1).all()
    assert "names:" in y.read_text()


def test_csv_header_and_degenerate_boxes(tmp_path):
    sys.path.insert(0, str(SRC))
    import sku110k_prep as sp
    p = tmp_path / "a.csv"
    p.write_text("image_name,x1,y1,x2,y2,class,image_width,image_height\n"
                 "a.jpg,10,10,50,60,object,100,100\na.jpg,5,5,5,9,object,100,100\na.jpg,-5,90,30,140,object,100,100\n")
    ann = sp.read_sku_csv(p)
    lines = sp.boxes_to_yolo(ann["a.jpg"]["boxes"], 100, 100)
    assert len(lines) == 2  # degenerate dropped, out-of-image clipped


def test_gap_synthesis(fake_kaggle):
    tmp, _ = fake_kaggle
    import gap_synth as gs
    src = tmp / "src"
    for split in ("train", "val"):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        for i in range(3):
            img, boxes = _shelf_image(rng=np.random.default_rng(i), skip={(1, 3)})  # one real gap
            cv2.imwrite(str(src / "images" / split / f"{split}{i}.jpg"), img)
            gs.write_yolo(src / "labels" / split / f"{split}{i}.txt", np.array(boxes, float),
                          np.zeros(len(boxes), int), 640, 480)
    # real-gap heuristic finds the missing pack
    img, boxes = _shelf_image(skip={(1, 3)})
    b = np.array(boxes, np.float32)
    gaps = gs.real_gaps(b, gs.cluster_rows(b))
    assert len(gaps) == 1 and 3 * 80 <= gaps[0][0] + 10 and gaps[0][2] <= 4 * 80 + 10
    y = gs.build_pack_gap(src, tmp / "pg", n_crops=2, out_size=160, p_synth=1.0, workers=1)
    labels = list((tmp / "pg" / "labels" / "train").glob("*.txt"))
    assert len(labels) == 6
    cls = [int(l.split()[0]) for f in labels for l in f.read_text().splitlines()]
    assert 1 in cls and 0 in cls
    assert "1: gap" in y.read_text()


def test_embed_metrics_and_discovery(fake_kaggle):
    tmp, inp = fake_kaggle
    root = inp / "rp2k-dataset" / "all" / "train"
    for c in range(120):
        d = root / f"cls{c}"
        d.mkdir(parents=True)
        for i in range(2):
            (d / f"{i}.jpg").write_bytes(b"x")
    (inp / "rp2k-dataset" / "all" / "test" / "cls0").mkdir(parents=True)
    (inp / "rp2k-dataset" / "all" / "test" / "cls0" / "9.jpg").write_bytes(b"x")
    import embed_train as et
    items = et.discover_folder_dataset(inp, min_classes=100)
    assert len(items) == 241
    # metrics on perfect embeddings
    g = np.eye(4, dtype=np.float32)
    q = np.array([[1, 0, 0, 0], [0, 0.9, 0.1, 0], [0, 0, 0, 1]], np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sc = et.retrieval_scores(g, ["a", "b", "c", "d"], q, ["a", "b", "d"])
    assert sc["r1"].all() and np.allclose(sc["ap"], 1.0)
    known = np.array([True, True, True])
    s = et.summarize(sc, known, 0.5, 0.1)
    assert s["accepted_precision"] == 1.0


def test_pk_sampler():
    import embed_train as et
    labels = [0] * 10 + [1] * 2 + [2] * 5
    s = et.PKSampler(labels, p=2, k=4, n_batches=5)
    batches = list(s)
    assert len(batches) == 5 and all(len(b) == 8 for b in batches)
    for b in batches:
        assert len({labels[i] for i in b}) == 2


def test_run_streaming_progress_and_watchdog(fake_kaggle, capsys):
    tmp, _ = fake_kaggle
    import kaggle_common as kc
    child = ("import sys, time\n"
             "for i in range(50):\n"
             "    sys.stdout.write('\\rbar %d/49' % i); sys.stdout.flush(); time.sleep(0.005)\n"
             "print()\nprint('epoch done')\nsys.exit(3)\n")
    rc = kc.run_streaming([sys.executable, "-c", child], tmp / "a.log", stall_min=1)
    out = capsys.readouterr().out
    assert rc == 3
    assert "bar 49/49" in out and "bar 10/49" not in out  # only the final bar state is printed
    assert "epoch done" in out
    assert (tmp / "a.log").stat().st_size > 400  # full raw output kept in the log
    hang = "import time\nprint('start', flush=True)\ntime.sleep(60)\n"
    rc = kc.run_streaming([sys.executable, "-c", hang], tmp / "b.log", stall_min=0.02)  # 1.2 s
    assert rc == -9


def test_cached_dir_and_restore(fake_kaggle):
    tmp, inp = fake_kaggle
    import kaggle_common as kc
    calls = []

    def build(d):
        calls.append(1)
        d.mkdir(parents=True)
        (d / "a.txt").write_text("x")

    out = kc.cached_dir("ds", tmp / "tmp" / "ds", build)
    assert (out / "a.txt").exists() and calls == [1]
    assert (kc.WORK / "cache" / "ds.tar").exists()
    import shutil
    shutil.rmtree(out)                       # new session: local disk wiped, cache survives
    out = kc.cached_dir("ds", tmp / "tmp" / "ds", build)
    assert (out / "a.txt").exists() and calls == [1]  # restored, not rebuilt
    # Kaggle-style resume: previous version output attached under /kaggle/input
    prev = inp / "shelftwin-01-output"
    (prev / "runs" / "r" / "weights").mkdir(parents=True)
    (prev / "runs" / "r" / "weights" / "last.pt").write_text("ck")
    (prev / "st_session_nb01.json").write_text('{"sessions": 1}')
    other = inp / "other-notebook"
    (other / "runs" / "zzz").mkdir(parents=True)
    (other / "st_session_nb02.json").write_text("{}")
    got = kc.restore_previous_outputs("nb01")
    assert "runs" in got and (kc.WORK / "runs" / "r" / "weights" / "last.pt").exists()
    assert not (kc.WORK / "runs" / "zzz").exists()
    kc.mark_session("nb01")
    import json
    assert json.loads((kc.WORK / "st_session_nb01.json").read_text())["sessions"] == 2
