"""Run one Ultralytics training from a JSON config (called as a subprocess)."""
import json
import sys

from kaggle_common import quiet_loggers

if __name__ == "__main__":
    quiet_loggers()
    cfg = json.load(open(sys.argv[1]))
    from ultralytics import YOLO
    dev = cfg.get("device")
    if isinstance(dev, str) and "," in dev:
        cfg["device"] = [int(x) for x in dev.split(",")]
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)
