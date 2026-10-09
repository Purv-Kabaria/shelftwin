"""Live simulated store for the dashboard demo (no models or datasets needed).

Same noise model as sim/simulate.py (shoppers block slots, detector misreads, glare),
stepped incrementally so the twin, engine, task service and dashboard run live.
Staff pressing "Done" in the dashboard refills the slot; otherwise slots are refilled
automatically 10-40 simulated minutes after they need restock.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelftwin.tasks import SlotInfo  # noqa: E402

DEMO_SKUS = [
    ("Maggi Masala 70g", 12, 2.5), ("Parle-G 250g", 9, 3.0), ("Amul Butter 100g", 6, 4.0),
    ("Tata Salt 1kg", 4, 1.0), ("Britannia Good Day 120g", 8, 1.5), ("Haldiram Bhujia 200g", 15, 1.2),
    ("Surf Excel 1kg", 22, 0.8), ("Colgate 100g", 14, 1.0), ("Aashirvaad Atta 5kg", 30, 0.6),
    ("Fortune Oil 1L", 18, 1.4), ("Kurkure 90g", 7, 3.5), ("Lays Classic 52g", 6, 4.5),
    ("Dairy Milk 50g", 9, 3.0), ("Bournvita 500g", 25, 0.7), ("Red Label Tea 250g", 20, 0.9),
    ("Dettol Soap 75g", 8, 1.6), ("Bisleri 1L", 4, 5.0), ("Thums Up 750ml", 7, 3.2),
    ("Frooti 600ml", 6, 2.4), ("Sunfeast Yippee 70g", 6, 2.2), ("MDH Garam Masala", 16, 0.5),
    ("Everest Chilli 100g", 12, 0.5), ("Nescafe Classic 50g", 35, 0.8), ("Horlicks 500g", 28, 0.6),
]


@dataclass
class LiveSlot:
    units: int
    cap: int
    rate_per_s: float
    blocked_until: float = -1.0
    mask_miss: bool = False
    need_since: float | None = None
    refill_at: float | None = None


class LiveStore:
    def __init__(self, n_slots: int = 24, seed: int = 7, dt: float = 2.0, rate_scale: float = 4.0,
                 blocks_per_h: float = 25.0):
        self.rng = random.Random(seed)
        self.dt, self.t = dt, 0.0
        self.blocks_per_h = blocks_per_h
        self.slots: dict[str, LiveSlot] = {}
        self.infos: dict[str, SlotInfo] = {}
        per_shelf = 6
        for i in range(n_slots):
            name, margin, demand = DEMO_SKUS[i % len(DEMO_SKUS)]
            shelf, pos = i // per_shelf + 1, i % per_shelf + 1
            sid = f"A4-B1-S{shelf}-P{pos}"
            cap = self.rng.randint(6, 12)
            # rate_scale speeds up depletion so the demo shows events within minutes
            self.slots[sid] = LiveSlot(cap, cap, demand * rate_scale / 3600.0)
            self.infos[sid] = SlotInfo(sid, aisle="4", bay="B1", shelf=shelf, sku_id=f"SKU{i:03d}", sku_name=name,
                                       demand_per_h=demand, unit_margin=margin,
                                       critical=name.startswith("Amul"),
                                       backroom_location=f"B-{10 + shelf}")

    def refill(self, slot_id: str):
        s = self.slots.get(slot_id)
        if s:
            s.units, s.need_since, s.refill_at = s.cap, None, None

    def step(self) -> list[tuple[str, float, float, float, float]]:
        """Advance one frame; returns observations (slot_id, ts, coverage, visibility, quality)."""
        self.t += self.dt
        rng, out = self.rng, []
        p_block = self.blocks_per_h * self.dt / 3600.0
        for sid, s in self.slots.items():
            if s.refill_at is not None and self.t >= s.refill_at:
                self.refill(sid)
            # depletion (Poisson thinning per frame)
            lam = s.rate_per_s * self.dt
            if rng.random() < lam:
                s.units = max(0, s.units - 1)
            if s.units <= 2 and s.need_since is None:
                s.need_since = self.t
                s.refill_at = self.t + rng.uniform(600, 2400)
            # occlusion
            if self.t >= s.blocked_until and rng.random() < p_block:
                dur = min(30.0, max(2.0, rng.lognormvariate(1.8, 0.7)))
                s.blocked_until = self.t + dur
                s.mask_miss = rng.random() < 0.05
            blocked = self.t < s.blocked_until
            # detector coverage
            if s.units >= 3:
                cov = rng.uniform(0.55, 1.0) if rng.random() >= 0.01 else rng.uniform(0.0, 0.2)
            elif s.units >= 1:
                cov = rng.uniform(0.05, 0.35)
            else:
                cov = rng.uniform(0.5, 0.9) if rng.random() < 0.03 else rng.uniform(0.0, 0.06)
            if blocked and rng.random() < 0.7:
                cov = rng.uniform(0.0, 0.3)
            vis = 1.0 if (not blocked or s.mask_miss) else rng.uniform(0.0, 0.5)
            out.append((sid, self.t, cov, vis, 1.0))
        return out

    def truth(self) -> dict[str, dict]:
        return {sid: dict(units=s.units, needs_restock=s.units <= 2, blocked=self.t < s.blocked_until)
                for sid, s in self.slots.items()}
