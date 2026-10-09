"""Layout versions: observed / draft / approved kept separate (design rule 5).

Imported planograms are CSV/JSON rows: bay, shelf, x_cm, width_cm, sku_id, facings.
Only an APPROVED layout drives alerts.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

OBSERVED, DRAFT, APPROVED = "observed", "draft", "approved"


@dataclass
class SlotSpec:
    slot_id: str
    bay: str
    shelf: int
    x_cm: float
    width_cm: float
    sku_id: str
    facings: int = 1
    depth_units: int = 4

    @property
    def centre_cm(self) -> float:
        return self.x_cm + self.width_cm / 2


@dataclass
class Layout:
    bay: str
    version: int
    status: str
    slots: list[SlotSpec] = field(default_factory=list)
    approved_by: str | None = None
    approved_at: str | None = None

    def rows(self) -> dict[int, list[SlotSpec]]:
        out: dict[int, list[SlotSpec]] = {}
        for s in sorted(self.slots, key=lambda s: (s.shelf, s.x_cm)):
            out.setdefault(s.shelf, []).append(s)
        return out

    def approve(self, by: str) -> "Layout":
        return Layout(self.bay, self.version, APPROVED, list(self.slots), by,
                      datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def slot_id_for(bay: str, shelf: int, idx: int) -> str:
    return f"{bay}-S{shelf}-P{idx}"


def load_planogram(path: str | Path, status: str = DRAFT) -> dict[str, Layout]:
    """Import a planogram file; imported layouts start as DRAFT until a manager approves."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text())
    else:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    by_bay: dict[str, list[dict]] = {}
    for r in rows:
        by_bay.setdefault(str(r["bay"]), []).append(r)
    layouts = {}
    for bay, rs in by_bay.items():
        rs.sort(key=lambda r: (int(r["shelf"]), float(r["x_cm"])))
        slots, counters = [], {}
        for r in rs:
            sh = int(r["shelf"])
            counters[sh] = counters.get(sh, 0) + 1
            slots.append(SlotSpec(r.get("slot_id") or slot_id_for(bay, sh, counters[sh]), bay, sh,
                                  float(r["x_cm"]), float(r["width_cm"]), str(r["sku_id"]),
                                  int(r.get("facings") or 1), int(r.get("depth_units") or 4)))
        layouts[bay] = Layout(bay, 1, status, slots)
    return layouts


def draft_from_observations(bay: str, shelf_obs: dict[int, list[tuple[float, float, str]]],
                            min_support: float = 0.7) -> Layout:
    """Draft a layout from stable observations: shelf -> [(x_cm, width_cm, sku_id)] where each
    entry is a pack position that held the same identity in >= min_support of clean views.
    The caller is responsible for computing that support over the first days."""
    slots = []
    for sh, packs in sorted(shelf_obs.items()):
        packs = sorted(packs)
        merged: list[list] = []
        for x, w, sku in packs:  # merge neighbouring facings of the same SKU into one slot
            if merged and merged[-1][2] == sku and x <= merged[-1][0] + merged[-1][1] + 0.25 * w:
                merged[-1][1] = x + w - merged[-1][0]
                merged[-1][3] += 1
            else:
                merged.append([x, w, sku, 1])
        for i, (x, w, sku, fac) in enumerate(merged, 1):
            if sku == "unknown":
                continue
            slots.append(SlotSpec(slot_id_for(bay, sh, i), bay, sh, x, w, sku, fac))
    return Layout(bay, 1, DRAFT, slots)
