"""Vision modules. Heavy deps (ultralytics, torch, faiss, paddleocr) load lazily."""
from .tiling import make_tiles, merge_tile_detections, nms, iou_matrix
from .geometry import Slot, BayGeometry, assign_to_slots, slot_coverage, cluster_rows, DriftChecker
from .visibility import slot_visibility, redact
from .identity import Catalogue, IdentityConfig, UNKNOWN_SKU
from .ocr import normalise_tokens, ocr_tiebreak

__all__ = ["make_tiles", "merge_tile_detections", "nms", "iou_matrix", "Slot", "BayGeometry",
           "assign_to_slots", "slot_coverage", "cluster_rows", "DriftChecker", "slot_visibility",
           "redact", "Catalogue", "IdentityConfig", "UNKNOWN_SKU", "normalise_tokens", "ocr_tiebreak"]
