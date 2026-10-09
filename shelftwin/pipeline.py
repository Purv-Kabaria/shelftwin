"""End-to-end vision pipeline for one camera / bay (spec Section 2.1, stages 1-7).

frame -> quality + change gate -> visibility mask -> pack/gap detector -> geometry
      -> coverage per slot -> (identity + planogram alignment) -> twin (engine + tasks)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gate import ChangeGate, QualityGate
from .planogram import AlignConfig, ExpectedSlot, ObservedPack, align_row, bay_layout_changed, slot_verdicts
from .twin import Twin
from .vision.geometry import BayGeometry, Slot, assign_to_slots, slot_coverage
from .vision.visibility import redact, slot_visibility


@dataclass
class PipelineConfig:
    bay_id: str = "B1"
    min_quality_to_run: float = 0.2      # below this, skip heavy models (engine still gets q)
    identity_every_n: int = 5            # run identity/alignment every N processed frames


class BayPipeline:
    def __init__(self, twin: Twin, geometry: BayGeometry, slots: list[Slot], detector, masker=None,
                 embedder=None, catalogue=None, cfg: PipelineConfig | None = None, sku_zones: dict | None = None):
        self.twin, self.geo, self.slots = twin, geometry, slots
        self.detector, self.masker = detector, masker
        self.embedder, self.catalogue = embedder, catalogue
        self.cfg = cfg or PipelineConfig()
        self.quality = QualityGate()
        self.change = ChangeGate()
        self.sku_zones = sku_zones or {}
        self._last_obs: dict[str, tuple[float, float]] = {}
        self._n = 0
        self.slot_polys = {s.slot_id: self.geo.slot_polygon_px(s) for s in slots}

    def process(self, bgr: np.ndarray, ts: float, frame_id: str = "") -> dict:
        h, w = bgr.shape[:2]
        qinfo = self.quality.score(bgr)
        q = qinfo["q"]
        changed = self.change.changed_bays(bgr, ts, {self.cfg.bay_id: (0, 0, w, h)})
        out = dict(ts=ts, q=q, changed=bool(changed), decisions={})

        # Re-check mode: while any slot's evidence is building, analyse every frame so a
        # decision rests on fresh, independent views (not on one cached frame).
        recheck = self.twin.pending()
        out["recheck"] = recheck
        if self._last_obs and (q < self.cfg.min_quality_to_run or (not changed and not recheck)):
            # Unchanged scene or unusable frame: advance time, add NO evidence
            cfg = self.twin.engine.cfg
            for sid, (cov, vis) in self._last_obs.items():
                clean = vis >= cfg.min_visibility and q >= cfg.min_quality
                out["decisions"][sid] = self.twin.touch(sid, ts, clean)
            self.twin.tick(ts)
            return out

        mask = self.masker.mask(bgr) if self.masker is not None else np.zeros((h, w), np.uint8)
        vis = slot_visibility(mask, self.slot_polys)
        dets = self.detector(bgr)
        packs = [d for d in dets if d["cls"] == "pack"]
        boxes_cm = [self.geo.box_to_cm(d["bbox_xyxy"]) for d in packs]
        assign = assign_to_slots(boxes_cm, self.slots)

        for s in self.slots:
            idx = assign[s.slot_id]
            cov = slot_coverage([boxes_cm[i] for i in idx], s)
            v = vis.get(s.slot_id, 1.0)
            self._last_obs[s.slot_id] = (cov, v)
            out["decisions"][s.slot_id] = self.twin.observe(s.slot_id, ts, cov, v, q, frame_id)

        self._n += 1
        if self.embedder is not None and self.catalogue is not None and self._n % self.cfg.identity_every_n == 0:
            out["alignment"] = self._identity_and_alignment(redact(bgr, mask), packs, boxes_cm, assign, ts, vis)
        self.twin.tick(ts)
        return out

    def _identity_and_alignment(self, bgr, packs, boxes_cm, assign, ts, vis) -> dict:
        crops, owners = [], []
        for s in self.slots:
            if vis.get(s.slot_id, 1.0) < 0.6:
                continue  # hidden slots never produce identity evidence
            for i in assign[s.slot_id]:
                x1, y1, x2, y2 = (int(round(v)) for v in packs[i]["bbox_xyxy"])
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    crops.append(bgr[max(0, y1):y2, max(0, x1):x2])
                    owners.append(i)
        ids = self.catalogue.identify(self.embedder(crops)) if crops else []
        by_pack = dict(zip(owners, ids))
        rows: dict[int, tuple[list, list]] = {}
        for s in self.slots:
            exp, obs = rows.setdefault(s.shelf, ([], []))
            exp.append(ExpectedSlot(s.slot_id, s.x_cm + s.width_cm / 2, s.sku_id or ""))
            for i in assign[s.slot_id]:
                if i in by_pack:
                    x1, _, x2, _ = boxes_cm[i]
                    obs.append(ObservedPack((x1 + x2) / 2, by_pack[i]["sku_id"], by_pack[i]["s1"]))
        outcomes = [align_row(e, o, AlignConfig()) for e, o in rows.values()]
        changed = bay_layout_changed(outcomes)
        verdicts = {}
        flat = [o for r in outcomes for o in r]
        for sid, kind in slot_verdicts(flat).items():
            obs_sku = next((o["observed"] for o in flat if o["slot_id"] == sid and o["kind"] == kind), None)
            verdicts[sid] = dict(kind=kind, observed=obs_sku, observed_zone=self.sku_zones.get(obs_sku))
        tasks = self.twin.service.on_alignment(self.cfg.bay_id, verdicts, ts, changed)
        for t in tasks:
            self.twin.store.upsert_task(t.to_dict())
        return dict(layout_changed=changed, verdicts=verdicts)
