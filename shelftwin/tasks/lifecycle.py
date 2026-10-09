"""Task lifecycle: Open -> Acknowledged -> In progress -> Verifying -> Verified (or Reopened)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

OPEN, ACK, IN_PROGRESS, VERIFYING, VERIFIED, REOPENED, SNOOZED, DISMISSED = (
    "open", "acknowledged", "in_progress", "verifying", "verified", "reopened", "snoozed", "dismissed")

ALLOWED = {
    OPEN: {ACK, SNOOZED, DISMISSED, VERIFIED},
    REOPENED: {ACK, SNOOZED, DISMISSED, VERIFIED},
    ACK: {IN_PROGRESS, VERIFYING, SNOOZED, VERIFIED},
    IN_PROGRESS: {VERIFYING, VERIFIED},
    VERIFYING: {VERIFIED, REOPENED},
    SNOOZED: {OPEN, VERIFIED},
    VERIFIED: set(),
    DISMISSED: set(),
}

KINDS = ("restock_low", "restock_empty", "displacement", "cold_chain", "record_check", "layout_check")
ROUTE = {"restock_low": "floor", "restock_empty": "backroom", "displacement": "floor",
         "cold_chain": "floor", "record_check": "inventory", "layout_check": "manager"}
IMMEDIATE = {"restock_empty", "cold_chain", "layout_check"}


class TransitionError(ValueError):
    pass


@dataclass
class Task:
    task_id: int
    slot_id: str
    kind: str
    priority: float
    created_ts: float
    status: str = OPEN
    assignee_team: str = ""
    ack_ts: float | None = None
    done_ts: float | None = None
    verified_ts: float | None = None
    snooze_until: float | None = None
    backroom_location: str | None = None
    reason: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def __post_init__(self):
        if not self.assignee_team:
            self.assignee_team = ROUTE.get(self.kind, "floor")

    def transition(self, new: str, ts: float, note: str = "") -> None:
        if new not in ALLOWED.get(self.status, set()):
            raise TransitionError(f"{self.status} -> {new} not allowed")
        self.history.append(dict(ts=ts, from_=self.status, to=new, note=note))
        self.status = new
        if new == ACK:
            self.ack_ts = ts
        elif new == VERIFYING:
            self.done_ts = ts
        elif new == VERIFIED:
            self.verified_ts = ts

    @property
    def active(self) -> bool:
        return self.status not in (VERIFIED, DISMISSED)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time_to_fix_s"] = (self.verified_ts - self.created_ts) if self.verified_ts else None
        return d
