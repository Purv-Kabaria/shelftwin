"""Pack and gap detector (spec Section 3.4). Wraps an Ultralytics YOLO model.

Weights come from notebooks 01/02 (best.pt or exported .onnx). Class names are read
from the model; a 1-class SKU-110K model only produces `pack`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tiling import make_tiles, merge_tile_detections


@dataclass
class DetectorConfig:
    weights: str = "models/pack_gap.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.55
    max_det: int = 1000
    tiles_x: int = 2
    overlap: float = 0.10
    device: str | int | None = None
    half: bool = False


class PackGapDetector:
    def __init__(self, cfg: DetectorConfig | None = None, model=None):
        self.cfg = cfg or DetectorConfig()
        if model is None:
            from ultralytics import YOLO  # heavy import, deferred
            model = YOLO(self.cfg.weights)
        self.model = model
        names = getattr(model, "names", None) or {0: "pack"}
        self.names = {int(k): (v if v != "object" else "pack") for k, v in dict(names).items()}

    def __call__(self, bgr: np.ndarray) -> list[dict]:
        h, w = bgr.shape[:2]
        n_x = self.cfg.tiles_x if w > 1.3 * h else 1
        tiles = make_tiles(w, h, n_x, self.cfg.overlap)
        crops = [bgr[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
        results = self.model.predict(crops, imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou,
                                     max_det=self.cfg.max_det, device=self.cfg.device, half=self.cfg.half,
                                     verbose=False)
        dets = []
        for (x1, y1, _, _), r in zip(tiles, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, conf, cls):
                dets.append(dict(bbox_xyxy=[b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1],
                                 cls=self.names.get(int(k), str(k)), conf=float(c)))
        return merge_tile_detections(dets, self.cfg.iou) if n_x > 1 else dets
