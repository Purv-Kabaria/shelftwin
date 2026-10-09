"""End-to-end demo logic with test doubles (no models): stub detector reads the scripted
scene, stub embedder is a colour histogram. Verifies the whole pipeline story."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "src"))

from test_notebook_helpers import _shelf_image  # noqa: E402


class StubDetector:
    """Returns the base boxes that are visible in the current scripted frame."""

    def __init__(self, base_boxes):
        self.base = [list(map(float, b)) for b in base_boxes]
        self.scene = None
        self.rng = np.random.default_rng(0)

    def __call__(self, img):
        out = []
        for b in self.base:
            x1, y1, x2, y2 = (int(v) for v in b)
            if self.scene is not None:
                sid = self._sid(b)
                tr = self.scene.truth
                if sid == self.scene.roles["A"] and tr.get("a_empty"):
                    continue
                m = self.scene.mask
                if m is not None and m[y1:y2, x1:x2].mean() > 0.3:
                    continue  # hidden by the shopper (whatever the mask reported)
                if self.rng.random() < 0.03:
                    continue  # random detector miss
            j = self.rng.normal(0, 1.0, 4)
            out.append(dict(bbox_xyxy=[b[i] + j[i] for i in range(4)], cls="pack", conf=0.9))
        return out

    def _sid(self, b):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        for sid, s in self.scene.slots.items():
            x1, y1, x2, y2 = self.scene.px(sid)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return sid
        return None


def hist_embedder(crops):
    """Centred mean colour: distinct packs point in distinct directions, robust to jitter."""
    out = []
    for c in crops:
        h, w = c.shape[:2]
        c = c[h // 4: h - h // 4, w // 4: w - w // 4]  # centre: ignore shelf background in the box
        v = c.reshape(-1, 3).mean(0) - 112.0
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(out, np.float32)


def test_e2e_story(tmp_path):
    import e2e_demo as ed
    img, boxes = _shelf_image(w=800, h=600, rows=3, per_row=8, rng=np.random.default_rng(5))
    det = StubDetector(boxes)
    res = ed.run_demo(img, det, embedder=hist_embedder, thresholds=dict(accept_sim=0.9, accept_margin=0.05),
                      on_scene=lambda s: setattr(det, "scene", s), log=lambda *a: None,
                      video_path=str(tmp_path / "demo.mp4"))
    s = res["summary"]
    assert s["engine_correct_alerts"] == 1, s
    assert s["engine_false_alerts"] == 0, s
    assert s["single_frame_alerts"] > 5 * max(1, s["engine_alerts"]), s
    assert s["detection_delay_s"] is not None and s["detection_delay_s"] <= 20, s
    assert s["refill_verified"], s
    assert s["displacement_task"], s
    ed.plot_trace(res, str(tmp_path / "trace.png"))
    assert (tmp_path / "trace.png").stat().st_size > 10_000
    assert (tmp_path / "demo.mp4").exists()


def test_e2e_with_annotated_layout(tmp_path):
    """Slots from the annotated layout; the (weak) detector still runs on every frame."""
    import e2e_demo as ed
    img, boxes = _shelf_image(w=800, h=600, rows=3, per_row=8, rng=np.random.default_rng(5))
    det = StubDetector(boxes)
    res = ed.run_demo(img, det, embedder=hist_embedder, thresholds=dict(accept_sim=0.9, accept_margin=0.05),
                      on_scene=lambda s: setattr(det, "scene", s), log=lambda *a: None,
                      layout_boxes=np.array(boxes, float), n_frames=60)
    assert res["summary"]["n_slots"] == 24
    cb = ed.crop_boxes(np.array([[0, 0, 10, 10], [300, 200, 340, 260]], float), (600, 800), frac=0.75, size=960)
    assert len(cb) == 1 and np.allclose(cb[0], (np.array([300, 200, 340, 260]) - [175, 75, 175, 75]) * (960 / 450))
