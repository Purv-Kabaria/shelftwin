from shelftwin.engine import (ACT, CLEAR, UNKNOWN, WAIT, EngineConfig, MajorityVoteBaseline,
                              SingleFrameBaseline, TemporalEngine)


def test_single_empty_frame_never_acts():
    e = TemporalEngine()
    assert e.update("s", 0, 0.9, 1.0) != ACT
    assert e.update("s", 2, 0.0, 1.0) != ACT  # one confident empty frame (rule 1)


def test_three_clean_empty_views_act():
    e = TemporalEngine()
    decisions = [e.update("s", t, 0.0, 1.0) for t in (0, 2, 4)]
    assert decisions[-1] == ACT  # 3 x 1.6 = 4.8 >= 4.0 minus small decay
    assert e.track("s").state == "EMPTY"


def test_hidden_slot_adds_no_evidence_and_becomes_unknown():
    e = TemporalEngine()
    e.update("s", 0, 0.9, 1.0)
    for t in range(2, 40, 2):
        d = e.update("s", t, 0.0, 0.2)  # blocked: detector says gap, mask says hidden
        assert d != ACT
    assert d == UNKNOWN  # no clean view for >= 20 s
    assert e.track("s").score <= 0.0


def test_low_quality_frames_ignored():
    e = TemporalEngine()
    for t in range(0, 20, 2):
        assert e.update("s", t, 0.0, 1.0, quality=0.3) in (WAIT, UNKNOWN)
    assert e.track("s").score == 0.0


def test_refill_clears_with_hysteresis():
    e = TemporalEngine()
    for t in (0, 2, 4):
        e.update("s", t, 0.0, 1.0)
    assert e.track("s").state == "EMPTY"
    out = [e.update("s", t, 0.9, 1.0) for t in (6, 8, 10, 12)]
    assert CLEAR in out
    assert e.track("s").state == "HEALTHY"


def test_decay_forgets_old_evidence():
    e = TemporalEngine(EngineConfig(decay_per_s=0.1))
    e.update("s", 0, 0.0, 1.0)
    e.update("s", 2, 0.0, 1.0)
    e.update("s", 100, 0.0, 1.0)  # long gap: first two views decayed away
    assert e.track("s").state == "HEALTHY"


def test_clamp():
    e = TemporalEngine()
    for t in range(0, 200, 2):
        e.update("s", t, 0.0, 1.0)
    assert e.track("s").score <= 8.0


def test_baselines():
    b = SingleFrameBaseline()
    assert b.update("s", 0, 0.1, 1.0) == ACT
    m = MajorityVoteBaseline(n=5, k=3)
    assert [m.update("s", t, 0.1, 1.0) for t in range(3)][-1] == ACT
    m2 = MajorityVoteBaseline(use_mask=True)
    assert all(m2.update("s", t, 0.0, 0.1) == WAIT for t in range(10))


def test_simulator_quick_engine_beats_baselines():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sim.simulate import SimConfig, run
    res = run(2, SimConfig(hours=1.0, n_slots=12), verbose=False)
    eng, base = res["ShelfTwin engine"], res["Single-frame"]
    assert eng["false_alerts_per_cam_hour"] < base["false_alerts_per_cam_hour"] / 20
    assert eng["precision"] > base["precision"]
