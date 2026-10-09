import pytest

from shelftwin.engine import ACT, CLEAR
from shelftwin.tasks import (DemandPosterior, SlotInfo, TaskService, TransitionError, low_stock_threshold,
                             priority, time_to_empty)
from shelftwin.tasks.lifecycle import ACK, OPEN, REOPENED, VERIFIED, VERIFYING


def test_threshold_and_priority():
    assert low_stock_threshold(8, 0.5, 0.25) == pytest.approx(6.0)
    assert low_stock_threshold(0.1) == 1.0
    assert priority(4, 0.5, 10, 0.4) == pytest.approx(8.0)
    assert priority(4, 0.5, 10, 0.4, critical=True) > 1000


def test_gamma_posterior_tte():
    p = DemandPosterior()
    p.update(units_removed=20, hours=10)
    assert p.mean == pytest.approx(21 / 10.5)
    r = time_to_empty(6, p)
    assert r["lo_h"] < r["mean_h"] < r["hi_h"]
    assert time_to_empty(0, p)["mean_h"] == 0.0


def test_service_flow_and_dedupe():
    svc = TaskService(slots={"s1": SlotInfo("s1")})
    t = svc.on_engine("s1", ACT, 0.0, dict(recent_coverage=0.0))
    assert t.kind == "restock_empty" and t.assignee_team == "backroom"
    assert svc.on_engine("s1", ACT, 5.0) is None  # one active task per (slot, kind)
    svc.ack(t.task_id, 10)
    svc.done(t.task_id, 20)
    assert t.status == VERIFYING
    svc.on_engine("s1", CLEAR, 30.0)
    assert t.status == VERIFIED and t.to_dict()["time_to_fix_s"] == 30.0


def test_low_stock_batched_and_escalation():
    events = []
    svc = TaskService(slots={"s1": SlotInfo("s1")}, notifiers=[lambda t, e: events.append((t.task_id, e))])
    svc.tick(0)
    t = svc.on_engine("s1", ACT, 1.0, dict(recent_coverage=0.3))
    assert t.kind == "restock_low" and events == []
    svc.tick(901)
    assert (t.task_id, "created") in events
    svc.tick(1000)
    assert (t.task_id, "escalated") in events


def test_reopen_when_not_verified():
    svc = TaskService(slots={"s1": SlotInfo("s1")})
    t = svc.on_engine("s1", ACT, 0.0)
    svc.ack(t.task_id, 1)
    svc.done(t.task_id, 2)
    svc.tick(2 + 901)
    assert t.status == REOPENED


def test_invalid_transition():
    svc = TaskService(slots={"s1": SlotInfo("s1")})
    t = svc.on_engine("s1", ACT, 0.0)
    assert t.status == OPEN
    with pytest.raises(TransitionError):
        t.transition(VERIFYING, 1)  # cannot be "done" before it was acknowledged


def test_displacement_needs_confirmation_and_layout_guard():
    svc = TaskService(slots={"s1": SlotInfo("s1", sku_id="A")})
    v = {"s1": dict(kind="substitution", observed="B")}
    assert svc.on_alignment("B1", v, 0, False) == []
    assert svc.on_alignment("B1", v, 1, False) == []
    out = svc.on_alignment("B1", v, 2, False)
    assert len(out) == 1 and out[0].kind == "displacement"
    svc2 = TaskService(slots={"s1": SlotInfo("s1")})
    out = svc2.on_alignment("B1", v, 0, True)
    assert len(out) == 1 and out[0].kind == "layout_check"


def test_cold_chain():
    svc = TaskService(slots={"s1": SlotInfo("s1", temperature_zone="ambient")})
    v = {"s1": dict(kind="substitution", observed="MILK", observed_zone="chilled")}
    for ts in range(3):
        out = svc.on_alignment("B1", v, ts, False)
    assert out[0].kind == "cold_chain" and out[0].priority > 1000
