import numpy as np
import pytest

from shelftwin.gate import ChangeGate, QualityGate
from shelftwin.vision import (BayGeometry, Catalogue, Slot, assign_to_slots, cluster_rows, make_tiles,
                              merge_tile_detections, normalise_tokens, ocr_tiebreak, slot_coverage,
                              slot_visibility)


def test_homography_roundtrip():
    g = BayGeometry([[100, 50], [900, 80], [880, 600], [120, 580]], (120.0, 80.0))
    pts = np.array([[200.0, 200.0], [700.0, 400.0]])
    assert np.allclose(g.to_px(g.to_cm(pts)), pts, atol=1e-6)
    assert np.allclose(g.to_cm([[100, 50]]), [[0, 0]], atol=1e-6)


def test_coverage_and_assignment():
    s1 = Slot("a", 1, 0, 20, 0, 30)
    s2 = Slot("b", 1, 20, 20, 0, 30)
    boxes = [(0, 5, 10, 30), (5, 5, 15, 30), (22, 5, 30, 30)]  # two overlapping in a, one in b
    asg = assign_to_slots(boxes, [s1, s2])
    assert asg == {"a": [0, 1], "b": [2]}
    assert slot_coverage([boxes[i] for i in asg["a"]], s1) == pytest.approx(15 * 25 / (20 * 30), abs=0.02)
    assert slot_coverage([], s1) == 0.0


def test_cluster_rows():
    lab = cluster_rows(np.array([100, 102, 98, 300, 305]), 50)
    assert len(set(lab[:3])) == 1 and len(set(lab[3:])) == 1 and lab[0] != lab[3]


def test_visibility():
    mask = np.zeros((100, 100), np.uint8)
    mask[:, :50] = 1
    v = slot_visibility(mask, {"l": np.array([[0, 0], [49, 0], [49, 99], [0, 99]]),
                               "r": np.array([[60, 0], [99, 0], [99, 99], [60, 99]])})
    assert v["l"] < 0.05 and v["r"] == 1.0


def test_tiles_cover_frame_and_merge():
    tiles = make_tiles(1920, 1080, 2, 0.10)
    assert tiles[0][0] == 0 and tiles[-1][2] == 1920
    assert tiles[0][2] > tiles[1][0]  # overlap
    dets = [dict(bbox_xyxy=[900, 10, 1000, 100], cls="pack", conf=0.9),
            dict(bbox_xyxy=[902, 12, 1001, 99], cls="pack", conf=0.8),
            dict(bbox_xyxy=[10, 10, 50, 50], cls="gap", conf=0.7)]
    out = merge_tile_detections(dets)
    assert len(out) == 2


def test_quality_gate():
    rng = np.random.default_rng(0)
    sharp = (rng.random((240, 320, 3)) * 200).astype(np.uint8)  # no glare
    import cv2
    blurry = cv2.GaussianBlur(sharp, (31, 31), 10)
    qg = QualityGate()
    qg.calibrate([sharp] * 3)
    assert qg.score(sharp)["q"] > 0.9
    assert qg.score(blurry)["q"] < 0.5
    white = np.full((100, 100, 3), 255, np.uint8)
    assert qg.score(white)["q_glare"] == 0.0


def test_change_gate():
    g = ChangeGate(force_every_s=60)
    a = np.zeros((200, 200, 3), np.uint8)
    b = a.copy()
    b[50:150, 50:150] = 255
    assert g.changed_bays(a, 0, {"x": (0, 0, 200, 200)}) == ["x"]  # first frame: full pass
    assert g.changed_bays(a, 2, {"x": (0, 0, 200, 200)}) == []
    assert g.changed_bays(b, 4, {"x": (0, 0, 200, 200)}) == ["x"]


def test_catalogue_abstains():
    cat = Catalogue(4)
    cat.add("A", np.array([[1, 0, 0, 0], [0.95, 0.05, 0, 0]], np.float32))
    cat.add("B", np.array([[0, 1, 0, 0]], np.float32))
    r = cat.identify(np.array([[1, 0.02, 0, 0], [0.7, 0.7, 0, 0], [0, 0, 1, 0]], np.float32))
    assert r[0]["sku_id"] == "A" and r[0]["accepted"]
    assert r[1]["sku_id"] == "unknown"  # look-alike: margin too small
    assert r[2]["sku_id"] == "unknown"  # new product: similarity too low


def test_catalogue_save_load(tmp_path):
    cat = Catalogue(4)
    cat.add("A", np.eye(4, dtype=np.float32)[:2])
    cat.save(tmp_path / "cat")
    c2 = Catalogue.load(tmp_path / "cat")
    assert c2.labels == ["A", "A"]


def test_ocr_tokens():
    assert {"maggi", "70g"} <= normalise_tokens("MAGGI Masala 70 gm")
    best, _ = ocr_tiebreak("Maggi 70 g masala", [("m70", "Maggi Masala 70g"), ("m140", "Maggi Masala 140g")])
    assert best == "m70"
