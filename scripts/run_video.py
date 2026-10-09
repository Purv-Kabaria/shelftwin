"""Run ShelfTwin on a video file / RTSP stream / folder of images, with the dashboard.

    python scripts/run_video.py --source shelf.mp4 --detector models/pack_gap.pt --fps 1
    python scripts/run_video.py --source rtsp://... --detector models/pack_gap.pt --layout configs/planogram_demo.csv

Without --layout, slots are drafted from the first frame (each shelf row is split into
slots at the detected pack positions) so the pipeline runs on any shelf footage.
Models are the outputs of the Kaggle notebooks (see notebooks/README.md).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelftwin.pipeline import BayPipeline, PipelineConfig  # noqa: E402
from shelftwin.planogram import load_planogram  # noqa: E402
from shelftwin.store import TwinStore  # noqa: E402
from shelftwin.tasks import SlotInfo  # noqa: E402
from shelftwin.twin import Twin  # noqa: E402
from shelftwin.vision.detector import DetectorConfig, PackGapDetector  # noqa: E402
from shelftwin.vision.geometry import BayGeometry, Slot, cluster_rows  # noqa: E402


def frames(source: str, fps: float):
    p = Path(source)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is not None:
                yield i / fps, img, f.name
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    live = source.startswith(("rtsp://", "http://", "https://")) or source.isdigit()
    step = max(1, int(round(vfps / fps)))
    i = 0
    t0 = time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            ts = (time.time() - t0) if live else i / vfps
            yield ts, img, f"f{i}"
        i += 1
    cap.release()


def draft_slots(dets: list[dict], geo: BayGeometry, bay: str) -> list[Slot]:
    packs = [d for d in dets if d["cls"] == "pack"]
    if not packs:
        raise SystemExit("No packs detected in the first frame; cannot draft slots. Provide --layout.")
    boxes = np.array([geo.box_to_cm(d["bbox_xyxy"]) for d in packs])
    med_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    rows = cluster_rows(boxes[:, 3], 0.5 * med_h)
    slots = []
    for r in np.unique(rows):
        rb = boxes[rows == r]
        rb = rb[np.argsort(rb[:, 0])]
        top, bottom = float(rb[:, 1].min()), float(rb[:, 3].max())
        for k, b in enumerate(rb, 1):
            slots.append(Slot(f"{bay}-S{r + 1}-P{k}", int(r) + 1, float(b[0]), float(b[2] - b[0]), top, bottom - top))
    return slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--detector", default="models/pack_gap.pt")
    ap.add_argument("--mask", default="yolo11n-seg.pt", help="person/cart seg weights; 'none' to disable")
    ap.add_argument("--layout", default=None, help="planogram CSV/JSON (bay, shelf, x_cm, width_cm, sku_id, facings)")
    ap.add_argument("--corners", default=None, help="TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy of the bay face in pixels")
    ap.add_argument("--size-cm", default=None, help="W,H of the bay face in cm (with --corners)")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--bay", default="B1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-server", action="store_true")
    a = ap.parse_args()

    detector = PackGapDetector(DetectorConfig(weights=a.detector, device=a.device))
    masker = None
    if a.mask.lower() != "none":
        from shelftwin.vision.visibility import VisibilityMasker
        masker = VisibilityMasker(a.mask, device=a.device)

    it = frames(a.source, a.fps)
    ts0, first, fid0 = next(it)
    h, w = first.shape[:2]
    if a.corners and a.size_cm:
        c = [float(v) for v in a.corners.split(",")]
        geo = BayGeometry(np.array(c).reshape(4, 2), tuple(float(v) for v in a.size_cm.split(",")))
    else:
        geo = BayGeometry.identity(w, h, px_per_cm=10.0)

    if a.layout:
        lay = load_planogram(a.layout)[a.bay]
        shelf_h = geo.size_cm[1] / max(1, max(s.shelf for s in lay.slots))
        slots = [Slot(s.slot_id, s.shelf, s.x_cm, s.width_cm, (s.shelf - 1) * shelf_h, shelf_h, s.sku_id, s.facings)
                 for s in lay.slots]
    else:
        slots = draft_slots(detector(first), geo, a.bay)
    print(f"{len(slots)} slots")

    infos = {s.slot_id: SlotInfo(s.slot_id, bay=a.bay, shelf=s.shelf, sku_id=s.sku_id or "",
                                 sku_name=s.sku_id or s.slot_id) for s in slots}
    twin = Twin(infos, store=TwinStore(":memory:"))
    pipe = BayPipeline(twin, geo, slots, detector, masker, cfg=PipelineConfig(bay_id=a.bay))

    if not a.no_server:
        import uvicorn
        from shelftwin.api.server import create_app
        app = create_app(twin)
        threading.Thread(target=lambda: uvicorn.run(app, port=a.port, log_level="warning"), daemon=True).start()
        print(f"dashboard: http://127.0.0.1:{a.port}")

    out = pipe.process(first, ts0, fid0)
    for ts, img, fid in it:
        out = pipe.process(img, ts, fid)
        acts = [s for s, d in out["decisions"].items() if d == "ACT"]
        if acts:
            print(f"t={ts:7.1f}s ACT {acts}")
    print("done; tasks:", [t.to_dict()["kind"] for t in twin.service.tasks.values()])
    if not a.no_server:
        print("source finished; dashboard still running (Ctrl+C to quit)")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
