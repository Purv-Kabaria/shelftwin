"""Task service: turns engine decisions and alignment verdicts into routed staff tasks.

Alert-fatigue controls (spec 3.9): empty / cold-chain go at once, low-stock batches every
15 min per aisle, one active task per (slot, kind), escalation after 10 min without ack,
camera verification closes tasks, unverified fixes are reopened.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

from ..engine import ACT, CLEAR
from .lifecycle import (ACK, DISMISSED, IMMEDIATE, OPEN, REOPENED, SNOOZED, VERIFIED, VERIFYING, Task)
from .priority import P_LOST_DEFAULT, priority


@dataclass
class SlotInfo:
    slot_id: str
    aisle: str = "1"
    bay: str = "B1"
    shelf: int = 1
    sku_id: str = ""
    sku_name: str = ""
    demand_per_h: float = 2.0
    unit_margin: float = 5.0
    temperature_zone: str = "ambient"
    critical: bool = False
    backroom_location: str | None = None


@dataclass
class ServiceConfig:
    low_batch_s: float = 900.0
    escalate_after_s: float = 600.0
    verify_timeout_s: float = 900.0
    displacement_confirm: int = 3     # consecutive substitution verdicts before a task
    low_coverage: float = 0.15        # ACT with mean coverage above this -> restock_low


@dataclass
class TaskService:
    slots: dict[str, SlotInfo] = field(default_factory=dict)
    cfg: ServiceConfig = field(default_factory=ServiceConfig)
    tasks: dict[int, Task] = field(default_factory=dict)
    notifiers: list[Callable[[Task, str], None]] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))
    _subst_streak: dict = field(default_factory=dict)
    _pending_low: list = field(default_factory=list)
    _last_batch: float | None = None
    _layout_alert: set = field(default_factory=set)

    # ------------------------------------------------------------------ helpers
    def active_task(self, slot_id: str, kind: str | None = None) -> Task | None:
        for t in self.tasks.values():
            if t.slot_id == slot_id and t.active and (kind is None or t.kind == kind):
                return t
        return None

    def _notify(self, task: Task, event: str):
        for n in self.notifiers:
            try:
                n(task, event)
            except Exception as e:  # a failing channel must not stop the service
                print(f"notifier error: {e}")

    def _open(self, slot_id: str, kind: str, ts: float, reason: dict) -> Task | None:
        if self.active_task(slot_id, kind):
            return None
        info = self.slots.get(slot_id, SlotInfo(slot_id))
        crit = info.critical or kind == "cold_chain"
        p = priority(info.demand_per_h, 0.5, info.unit_margin, P_LOST_DEFAULT, crit)
        t = Task(next(self._ids), slot_id, kind, p, ts, backroom_location=info.backroom_location,
                 reason=reason)
        self.tasks[t.task_id] = t
        if kind in IMMEDIATE:
            self._notify(t, "created")
        else:
            self._pending_low.append(t.task_id)
        return t

    # ------------------------------------------------------------------ inputs
    def on_engine(self, slot_id: str, decision: str, ts: float, reason: dict | None = None) -> Task | None:
        reason = reason or {}
        if decision == ACT:
            cov = reason.get("recent_coverage", 0.0)
            kind = "restock_low" if cov > self.cfg.low_coverage else "restock_empty"
            return self._open(slot_id, kind, ts, reason)
        if decision == CLEAR:
            for t in list(self.tasks.values()):
                if t.slot_id == slot_id and t.active and t.kind.startswith("restock"):
                    if t.status == SNOOZED:
                        t.transition(OPEN, ts, "refilled while snoozed")
                    t.transition(VERIFIED, ts, "camera verified (CLEAR)")
                    self._notify(t, "verified")
        return None

    def on_alignment(self, bay: str, verdicts: dict[str, dict], ts: float, layout_changed: bool) -> list[Task]:
        """verdicts: slot_id -> {'kind': match|substitution|gap|uncertain, 'observed': sku, 'observed_zone': ...}"""
        out = []
        if layout_changed:
            if bay not in self._layout_alert:
                self._layout_alert.add(bay)
                t = self._open(f"{bay}:layout", "layout_check", ts,
                               dict(msg="Many slots changed at once. Was the layout changed?"))
                if t:
                    out.append(t)
            return out  # design rule 6: no per-slot displacement storm
        self._layout_alert.discard(bay)
        for sid, v in verdicts.items():
            if v.get("kind") == "substitution":
                self._subst_streak[sid] = self._subst_streak.get(sid, 0) + 1
            else:
                self._subst_streak[sid] = 0
                t = self.active_task(sid, "displacement") or self.active_task(sid, "cold_chain")
                if t and v.get("kind") == "match":
                    if t.status == SNOOZED:
                        t.transition(OPEN, ts)
                    t.transition(VERIFIED, ts, "camera verified (match)")
                    self._notify(t, "verified")
                continue
            if self._subst_streak[sid] >= self.cfg.displacement_confirm:
                info = self.slots.get(sid, SlotInfo(sid))
                chilled = v.get("observed_zone") in ("chilled", "frozen") and info.temperature_zone == "ambient"
                kind = "cold_chain" if chilled else "displacement"
                t = self._open(sid, kind, ts, dict(expected=info.sku_id, observed=v.get("observed"),
                                                   views=self._subst_streak[sid]))
                if t:
                    out.append(t)
        return out

    # ------------------------------------------------------------------ staff actions
    def ack(self, task_id: int, ts: float):
        self.tasks[task_id].transition(ACK, ts)

    def done(self, task_id: int, ts: float):
        self.tasks[task_id].transition(VERIFYING, ts, "staff marked done")

    def snooze(self, task_id: int, until_ts: float, ts: float, reason: str = ""):
        t = self.tasks[task_id]
        t.transition(SNOOZED, ts, reason)
        t.snooze_until = until_ts

    def wrong_alert(self, task_id: int, ts: float, note: str = "") -> dict:
        t = self.tasks[task_id]
        t.transition(DISMISSED, ts, f"wrong alert: {note}")
        return dict(task_id=task_id, slot_id=t.slot_id, kind="wrong_alert", note=note, ts=ts)  # training label

    # ------------------------------------------------------------------ clock
    def tick(self, ts: float) -> None:
        if self._last_batch is None:
            self._last_batch = ts
        if ts - self._last_batch >= self.cfg.low_batch_s and self._pending_low:
            batch = [self.tasks[i] for i in self._pending_low if self.tasks[i].active]
            for t in sorted(batch, key=lambda t: (self.slots.get(t.slot_id, SlotInfo(t.slot_id)).aisle, -t.priority)):
                self._notify(t, "created")
            self._pending_low.clear()
            self._last_batch = ts
        for t in self.tasks.values():
            if t.status in (OPEN, REOPENED) and ts - t.created_ts >= self.cfg.escalate_after_s \
                    and not t.reason.get("escalated"):
                t.reason["escalated"] = True
                t.priority *= 1.5
                self._notify(t, "escalated")
            elif t.status == VERIFYING and t.done_ts and ts - t.done_ts >= self.cfg.verify_timeout_s:
                t.transition(REOPENED, ts, "slot still not verified full")
                self._notify(t, "reopened")
            elif t.status == SNOOZED and t.snooze_until and ts >= t.snooze_until:
                t.transition(OPEN, ts, "snooze ended")
                self._notify(t, "created")

    def open_tasks(self) -> list[Task]:
        return sorted((t for t in self.tasks.values() if t.active), key=lambda t: -t.priority)
