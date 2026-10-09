"""Temporal state engine (spec Section 3.8).

Each slot is an occupancy-grid cell. Evidence L is the log-odds that the slot
needs restock:

    L_t = exp(-lambda * dt) * L_{t-1} + v_t * q_t * l(c_t)

A single confident frame never opens a task; hidden or low-quality views add no
evidence and, if they persist, the slot is reported as CANNOT SEE (UNKNOWN).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict


# Slot states
HEALTHY = "HEALTHY"
EMPTY = "EMPTY"

# Engine decisions
OK = "OK"
WAIT = "WAIT"
ACT = "ACT"
CLEAR = "CLEAR"
UNKNOWN = "UNKNOWN"  # shown as CANNOT SEE in the UI


@dataclass
class EngineConfig:
    min_visibility: float = 0.6
    min_quality: float = 0.5
    empty_cov: float = 0.40
    full_cov: float = 0.45
    llr_empty: float = 1.6
    llr_stocked: float = 2.2
    decay_per_s: float = 0.02
    act_threshold: float = 4.0
    clear_threshold: float = -1.0
    unknown_after_s: float = 20.0
    score_min: float = -6.0
    score_max: float = 8.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "EngineConfig":
        d = d or {}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class SlotTrack:
    state: str = HEALTHY
    score: float = 0.0
    last_t: float | None = None
    last_clean_t: float | None = None
    last_coverage: float | None = None
    clean_views: int = 0          # clean confirming views since the last state change
    blocked_views: int = 0
    total_views: int = 0


@dataclass
class Decision:
    slot_id: str
    t: float
    decision: str
    state: str
    score: float
    coverage: float
    visibility: float
    quality: float
    weight: float
    clean_views: int
    last_clean_t: float | None

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalEngine:
    """Visibility-weighted log-odds evidence with decay and hysteresis."""

    def __init__(self, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()
        self.tracks: dict[str, SlotTrack] = {}

    def track(self, slot_id: str) -> SlotTrack:
        if slot_id not in self.tracks:
            self.tracks[slot_id] = SlotTrack()
        return self.tracks[slot_id]

    def evidence(self, coverage: float) -> float:
        """l(c): positive means 'needs restock'."""
        if coverage < self.cfg.empty_cov:
            return self.cfg.llr_empty
        if coverage > self.cfg.full_cov:
            return -self.cfg.llr_stocked
        return 0.0

    def update(self, slot_id: str, t: float, coverage: float, visibility: float,
               quality: float = 1.0) -> str:
        return self.update_full(slot_id, t, coverage, visibility, quality).decision

    def update_full(self, slot_id: str, t: float, coverage: float, visibility: float,
                    quality: float = 1.0) -> Decision:
        cfg = self.cfg
        tr = self.track(slot_id)
        tr.total_views += 1

        if tr.last_t is not None:
            dt = max(0.0, t - tr.last_t)
            tr.score *= math.exp(-cfg.decay_per_s * dt)
        tr.last_t = t
        if tr.last_clean_t is None:
            tr.last_clean_t = t  # grace period starts at the first observation

        def out(decision: str, w: float) -> Decision:
            return Decision(slot_id, t, decision, tr.state, tr.score, coverage, visibility,
                            quality, w, tr.clean_views, tr.last_clean_t)

        if visibility < cfg.min_visibility or quality < cfg.min_quality:
            tr.blocked_views += 1
            if t - tr.last_clean_t >= cfg.unknown_after_s:
                return out(UNKNOWN, 0.0)
            return out(WAIT, 0.0)

        w = visibility * quality
        llr = self.evidence(coverage)
        tr.score = min(cfg.score_max, max(cfg.score_min, tr.score + w * llr))
        tr.last_clean_t = t
        tr.last_coverage = coverage
        if llr != 0.0:
            tr.clean_views += 1

        if tr.state == HEALTHY and tr.score >= cfg.act_threshold:
            tr.state = EMPTY
            tr.clean_views = 0
            return out(ACT, w)
        if tr.state == EMPTY and tr.score <= cfg.clear_threshold:
            tr.state = HEALTHY
            tr.clean_views = 0
            return out(CLEAR, w)
        if tr.state == HEALTHY and tr.score <= 0.0:
            return out(OK, w)
        return out(WAIT, w)

    def touch(self, slot_id: str, t: float, clean: bool) -> str:
        """Advance time for a slot whose view did not change (change gate said 'same scene').

        A repeated view is not an independent observation, so it adds NO evidence (otherwise one
        wrong frame would be re-counted every frame). It only applies decay and, if the cached
        view is clean, keeps the slot 'seen' so it does not drift into CANNOT SEE."""
        cfg = self.cfg
        tr = self.track(slot_id)
        if tr.last_t is not None:
            tr.score *= math.exp(-cfg.decay_per_s * max(0.0, t - tr.last_t))
        tr.last_t = t
        if tr.last_clean_t is None:
            tr.last_clean_t = t
        if clean:
            tr.last_clean_t = t
        elif t - tr.last_clean_t >= cfg.unknown_after_s:
            return UNKNOWN
        return OK if (tr.state == HEALTHY and tr.score <= 0.0) else WAIT

    def pending(self, slot_id: str, margin: float = 0.3) -> bool:
        """True while evidence is building toward a state change (needs fresh views)."""
        tr = self.tracks.get(slot_id)
        if tr is None:
            return False
        if tr.state == HEALTHY:
            return tr.score > margin
        return tr.score < 0.0  # EMPTY with stocked evidence arriving (a refill to verify)

    def reset(self, slot_id: str | None = None) -> None:
        if slot_id is None:
            self.tracks.clear()
        else:
            self.tracks.pop(slot_id, None)

    def snapshot(self) -> dict[str, dict]:
        return {k: asdict(v) for k, v in self.tracks.items()}
