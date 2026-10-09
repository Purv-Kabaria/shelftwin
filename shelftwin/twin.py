"""The twin: per-slot record fed by observations from any source (camera pipeline,
recorded outputs, or the simulator). Runs the temporal engine, logs every decision,
and drives the task service."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from .engine import ACT, CLEAR, UNKNOWN, EngineConfig, SingleFrameBaseline, TemporalEngine
from .store import TwinStore
from .tasks import SlotInfo, TaskService


@dataclass
class SlotView:
    slot_id: str
    state: str = "HEALTHY"
    decision: str = "OK"
    score: float = 0.0
    coverage: float | None = None
    visibility: float | None = None
    quality: float | None = None
    last_ts: float | None = None
    last_clean_ts: float | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=12))  # (ts, cov, vis, q, frame_id)
    baseline_alerts: int = 0
    engine_alerts: int = 0


class Twin:
    def __init__(self, slots: dict[str, SlotInfo], engine_cfg: EngineConfig | None = None,
                 store: TwinStore | None = None, service: TaskService | None = None, track_baseline: bool = True):
        self.slots = slots
        self.engine = TemporalEngine(engine_cfg)
        self.baseline = SingleFrameBaseline(use_mask=False) if track_baseline else None
        self.store = store or TwinStore()
        self.service = service or TaskService(slots=slots)
        self.views: dict[str, SlotView] = {sid: SlotView(sid) for sid in slots}
        self.lock = threading.RLock()
        self.now: float = 0.0

    def observe(self, slot_id: str, ts: float, coverage: float, visibility: float, quality: float = 1.0,
                frame_id: str = "", identity: dict | None = None) -> str:
        with self.lock:
            self.now = max(self.now, ts)
            v = self.views.setdefault(slot_id, SlotView(slot_id))
            prev_state = self.engine.track(slot_id).state
            prev_decision = v.decision
            d = self.engine.update_full(slot_id, ts, coverage, visibility, quality)
            v.recent.append((ts, coverage, visibility, quality, frame_id))
            v.state, v.decision, v.score = d.state, d.decision, d.score
            v.coverage, v.visibility, v.quality, v.last_ts = coverage, visibility, quality, ts
            v.last_clean_ts = d.last_clean_t
            self.store.observation(slot_id, ts, coverage, visibility, quality, frame_id,
                                   (identity or {}).get("sku_id"), (identity or {}).get("s1"),
                                   (identity or {}).get("s2"))
            if self.baseline is not None and self.baseline.update(slot_id, ts, coverage, visibility, quality) == ACT:
                v.baseline_alerts += 1
            if d.decision in (ACT, CLEAR) or (d.decision == UNKNOWN and prev_decision != UNKNOWN):
                inputs = self.explain(slot_id)
                self.store.state_event(slot_id, ts, prev_state, d.state, d.score, d.decision,
                                       [r[4] for r in v.recent], inputs)
                if d.decision in (ACT, CLEAR):
                    if d.decision == ACT:
                        v.engine_alerts += 1
                    t = self.service.on_engine(slot_id, d.decision, ts, inputs)
                    if t is not None:
                        self.store.upsert_task(t.to_dict())
            return d.decision

    def touch(self, slot_id: str, ts: float, clean: bool) -> str:
        """Unchanged view: advance time without adding evidence (see TemporalEngine.touch)."""
        with self.lock:
            self.now = max(self.now, ts)
            v = self.views.setdefault(slot_id, SlotView(slot_id))
            prev_decision = v.decision
            d = self.engine.touch(slot_id, ts, clean)
            tr = self.engine.track(slot_id)
            v.decision, v.score, v.state, v.last_ts, v.last_clean_ts = d, tr.score, tr.state, ts, tr.last_clean_t
            if d == UNKNOWN and prev_decision != UNKNOWN:
                self.store.state_event(slot_id, ts, tr.state, tr.state, tr.score, d, [], self.explain(slot_id))
            return d

    def pending(self) -> bool:
        with self.lock:
            return any(self.engine.pending(sid) for sid in self.views)

    def explain(self, slot_id: str) -> dict:
        """'Why this alert?': the clean views behind the decision."""
        v = self.views[slot_id]
        cfg = self.engine.cfg
        clean = [r for r in v.recent if r[2] >= cfg.min_visibility and r[3] >= cfg.min_quality]
        blocked = len(v.recent) - len(clean)
        confirm = [r for r in clean if r[1] < cfg.empty_cov] if v.state == "EMPTY" else clean
        span = (confirm[-1][0] - confirm[0][0]) if len(confirm) > 1 else 0.0
        stocked_before = [r[1] for r in clean if r[1] > cfg.full_cov]
        info = self.slots.get(slot_id)
        return dict(
            clean_views=len(confirm), span_s=round(span, 1),
            coverage_now=round(clean[-1][1], 3) if clean else None,
            coverage_before=round(stocked_before[0], 3) if stocked_before else None,
            recent_coverage=round(sum(r[1] for r in confirm[-3:]) / max(1, len(confirm[-3:])), 3) if confirm else 0.0,
            share_blocked=round(blocked / max(1, len(v.recent)), 2),
            score=round(v.score, 2), sku=info.sku_name if info else "",
            backroom=info.backroom_location if info else None,
        )

    def tick(self, ts: float):
        with self.lock:
            self.now = max(self.now, ts)
            self.service.tick(ts)
            for t in self.service.tasks.values():
                self.store.upsert_task(t.to_dict())
            self.store.commit()

    def snapshot(self) -> dict:
        with self.lock:
            cfg = self.engine.cfg
            out = []
            for sid, v in self.views.items():
                info = self.slots.get(sid)
                age = (self.now - v.last_clean_ts) if v.last_clean_ts is not None else None
                ui = v.state
                if v.decision == "UNKNOWN":
                    ui = "CANNOT_SEE"
                elif v.state == "HEALTHY" and v.score > 0.5:
                    ui = "WAIT"
                out.append(dict(slot_id=sid, ui_state=ui, state=v.state, decision=v.decision,
                                score=round(v.score, 2), coverage=v.coverage, visibility=v.visibility,
                                last_clean_age_s=round(age, 1) if age is not None else None,
                                sku=info.sku_name if info else "", shelf=info.shelf if info else 0,
                                aisle=info.aisle if info else "", bay=info.bay if info else "",
                                baseline_alerts=v.baseline_alerts, engine_alerts=v.engine_alerts,
                                act_threshold=cfg.act_threshold))
            return dict(now=self.now, slots=out,
                        totals=dict(baseline_alerts=sum(v.baseline_alerts for v in self.views.values()),
                                    engine_alerts=sum(v.engine_alerts for v in self.views.values())))
