"""Light end-to-end checks: twin + task service + API on the simulated store, and calibration."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "src"))


def test_twin_on_live_sim_opens_and_verifies_tasks():
    from shelftwin.twin import Twin
    from sim.live import LiveStore
    live = LiveStore(n_slots=12, seed=3, rate_scale=20.0)
    twin = Twin(live.infos)
    for _ in range(2500):  # ~83 simulated minutes
        for sid, ts, cov, vis, q in live.step():
            twin.observe(sid, ts, cov, vis, q)
        twin.tick(live.t)
    tasks = list(twin.service.tasks.values())
    assert tasks, "expected restock tasks"
    assert any(t.status == "verified" for t in tasks), "refills should be camera-verified"
    snap = twin.snapshot()
    assert snap["totals"]["baseline_alerts"] > 5 * max(1, snap["totals"]["engine_alerts"])
    ev = twin.store.query("SELECT COUNT(*) AS n FROM state_event WHERE decision='ACT'")[0]["n"]
    assert ev == snap["totals"]["engine_alerts"]


def test_api_endpoints():
    from fastapi.testclient import TestClient
    from shelftwin.api.server import create_app
    from shelftwin.twin import Twin
    from sim.live import LiveStore
    live = LiveStore(n_slots=6, seed=1)
    twin = Twin(live.infos)
    sid = next(iter(live.infos))
    for t in (2, 4, 6, 8):
        twin.observe(sid, t, 0.0, 1.0)
    c = TestClient(create_app(twin, on_done=live.refill))
    assert c.get("/").status_code == 200
    st = c.get("/api/state").json()
    assert st["totals"]["engine_alerts"] == 1
    tasks = c.get("/api/tasks?lang=hi").json()
    assert len(tasks) == 1 and "खाली" in tasks[0]["message"]
    tid = tasks[0]["task_id"]
    assert c.post(f"/api/tasks/{tid}/done").status_code == 409  # must acknowledge first
    assert c.post(f"/api/tasks/{tid}/ack").status_code == 200
    assert c.post(f"/api/tasks/{tid}/done").status_code == 200
    for t in (10, 12, 14, 16, 18, 20):  # hysteresis: ~4 stocked views to verify
        twin.observe(sid, t, 0.9, 1.0)
    done = c.get("/api/tasks?include_closed=true").json()
    assert done[0]["status"] == "verified"
    assert c.get(f"/api/slots/{sid}/replay").json()["events"]
    assert c.get("/api/slots/nope/explain").status_code == 404


def test_calibration_grid_tiny():
    import calibrate_engine as ce
    from sim.simulate import SimConfig
    best, res = ce.grid_search(SimConfig(hours=0.5, n_slots=6),
                               dict(llr_empty=[1.6], llr_stocked=[2.2], act_threshold=[3.0, 4.0], decay_per_s=[0.02]),
                               seeds=[100], workers=1, min_recall=0.0, max_p95=1e9)
    assert len(res) == 2 and best["params"]["act_threshold"] in (3.0, 4.0)


def test_calibration_resumes_from_cache(tmp_path):
    import calibrate_engine as ce
    from sim.simulate import SimConfig
    grid = dict(llr_empty=[1.6], llr_stocked=[2.2], act_threshold=[3.0, 4.0], decay_per_s=[0.02])
    cache = tmp_path / "grid.jsonl"
    ce.grid_search(SimConfig(hours=0.5, n_slots=6), dict(grid, act_threshold=[3.0]), seeds=[100], workers=1,
                   min_recall=0.0, max_p95=1e9, cache_path=cache)
    assert len(cache.read_text().splitlines()) == 1
    best, res = ce.grid_search(SimConfig(hours=0.5, n_slots=6), grid, seeds=[100], workers=1,
                               min_recall=0.0, max_p95=1e9, cache_path=cache)
    assert len(res) == 2 and len(cache.read_text().splitlines()) == 2
