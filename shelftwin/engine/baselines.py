"""Baselines the temporal engine is compared against (spec Section 3.8 / 7.1)."""
from __future__ import annotations

from collections import deque

from .temporal import ACT, CLEAR, OK, WAIT, HEALTHY, EMPTY


class SingleFrameBaseline:
    """Alerts on the first empty-looking frame."""

    def __init__(self, empty_cov: float = 0.40, use_mask: bool = False, min_visibility: float = 0.6):
        self.empty_cov = empty_cov
        self.use_mask = use_mask
        self.min_visibility = min_visibility
        self.state: dict[str, str] = {}

    def update(self, slot_id, t, coverage, visibility, quality=1.0) -> str:
        st = self.state.get(slot_id, HEALTHY)
        if self.use_mask and visibility < self.min_visibility:
            return WAIT
        looks_empty = coverage < self.empty_cov
        if st == HEALTHY and looks_empty:
            self.state[slot_id] = EMPTY
            return ACT
        if st == EMPTY and not looks_empty:
            self.state[slot_id] = HEALTHY
            return CLEAR
        return OK if st == HEALTHY else WAIT


class MajorityVoteBaseline:
    """Alerts when k of the last n usable frames look empty."""

    def __init__(self, n: int = 5, k: int = 3, use_mask: bool = True, empty_cov: float = 0.40,
                 min_visibility: float = 0.6):
        self.n, self.k = n, k
        self.use_mask = use_mask
        self.empty_cov = empty_cov
        self.min_visibility = min_visibility
        self.hist: dict[str, deque] = {}
        self.state: dict[str, str] = {}

    def update(self, slot_id, t, coverage, visibility, quality=1.0) -> str:
        if self.use_mask and visibility < self.min_visibility:
            return WAIT
        h = self.hist.setdefault(slot_id, deque(maxlen=self.n))
        h.append(coverage < self.empty_cov)
        st = self.state.get(slot_id, HEALTHY)
        votes = sum(h)
        if st == HEALTHY and votes >= self.k:
            self.state[slot_id] = EMPTY
            return ACT
        if st == EMPTY and (len(h) - votes) >= self.k:
            self.state[slot_id] = HEALTHY
            return CLEAR
        return OK if st == HEALTHY else WAIT
