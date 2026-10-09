"""Engine calibration (spec 6.8): grid-search llr_empty, llr_stocked, act_threshold, decay
to minimise false alerts per camera-hour subject to recall >= 99% and p95 delay <= 30 s.

Tuning uses simulator seeds 100-109; the chosen config is reported on held-out seeds 0-19,
so the reported numbers are not tuned on the data they are measured on. If a recorded CSV
(slot_id, ts, coverage, visibility, quality, needs_restock) is attached, it is evaluated too.
"""
from __future__ import annotations

import itertools
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from shelftwin.engine import EngineConfig, TemporalEngine
from sim.simulate import SimConfig, World, aggregate, run_method, simulate_world

GRID = dict(llr_empty=[1.2, 1.6, 2.0], llr_stocked=[1.6, 2.2, 2.8],
            act_threshold=[3.0, 3.5, 4.0, 4.5, 5.0], decay_per_s=[0.01, 0.02, 0.04])
TUNE_SEEDS = list(range(100, 110))
_WORLDS: list[World] = []


def _init(seeds, sim_cfg):
    global _WORLDS
    _WORLDS = [simulate_world(s, sim_cfg) for s in seeds]


def _eval(params: dict) -> dict:
    cfg = EngineConfig(**params)
    r = aggregate([run_method(lambda: TemporalEngine(cfg), w) for w in _WORLDS])
    return dict(params=params, **r)


def grid_search(sim_cfg: SimConfig | None = None, grid: dict | None = None, seeds=TUNE_SEEDS,
                min_recall=0.99, max_p95=30.0, workers: int | None = None,
                cache_path: Path | None = None) -> tuple[dict, list[dict]]:
    """Parallel over all CPU cores; with `cache_path` (JSON lines) finished configs are kept,
    so an interrupted search resumes where it stopped."""
    import json
    grid = grid or GRID
    sim_cfg = sim_cfg or SimConfig()
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    done: dict[str, dict] = {}
    if cache_path is not None and Path(cache_path).exists():
        for line in Path(cache_path).read_text().splitlines():
            try:
                r = json.loads(line)
                done[json.dumps(r["params"], sort_keys=True)] = r
            except Exception:
                continue
    todo = [c for c in combos if json.dumps(c, sort_keys=True) not in done]
    if done:
        print(f"Resuming grid search: {len(done)} configs cached, {len(todo)} to run")
    workers = workers or max(1, os.cpu_count() or 1)
    results = [done[json.dumps(c, sort_keys=True)] for c in combos if json.dumps(c, sort_keys=True) in done]
    if todo:
        with Pool(min(workers, len(todo)), initializer=_init, initargs=(seeds, sim_cfg)) as pool:
            for r in pool.imap_unordered(_eval, todo):
                results.append(r)
                if cache_path is not None:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "a") as f:
                        f.write(json.dumps(r, default=float) + "\n")
    ok = [r for r in results if r["recall"] >= min_recall and r["p95_delay_s"] <= max_p95]
    if not ok:
        print("No config met the constraints; relaxing to best recall.")
        ok = sorted(results, key=lambda r: -r["recall"])[:10]
    best = min(ok, key=lambda r: (r["false_alerts_per_cam_hour"], r["median_delay_s"]))
    return best, results


def evaluate_recorded_csv(path: Path, cfg: EngineConfig) -> dict:
    """Score the engine on recorded per-frame outputs with ground truth."""
    import pandas as pd
    df = pd.read_csv(path).sort_values(["slot_id", "ts"])
    alerts, correct, events, detected, delays = 0, 0, 0, 0, []
    for sid, g in df.groupby("slot_id"):
        e = TemporalEngine(cfg)
        need = g["needs_restock"].astype(bool).to_numpy()
        ts = g["ts"].to_numpy()
        acts = [i for i, r in enumerate(g.itertuples()) if
                e.update(sid, r.ts, r.coverage, r.visibility, getattr(r, "quality", 1.0)) == "ACT"]
        alerts += len(acts)
        correct += sum(need[i] for i in acts)
        d = np.diff(np.concatenate([[0], need.astype(int), [0]]))
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            events += 1
            hit = [i for i in acts if a <= i < b]
            if hit:
                detected += 1
                delays.append(ts[hit[0]] - ts[a])
    hours = df.groupby("slot_id")["ts"].agg(lambda s: s.max() - s.min()).max() / 3600
    return dict(alerts=alerts, precision=correct / max(1, alerts), recall=detected / max(1, events),
                false_alerts_per_cam_hour=(alerts - correct) / max(hours, 1e-9),
                median_delay_s=float(np.median(delays)) if delays else None)


# ============================================================================ experiments / error analysis
SENSITIVITY = dict(
    p_gap_when_blocked=[0.3, 0.5, 0.7, 0.9],      # how often a shopper makes the detector see a gap
    p_mask_miss=[0.0, 0.05, 0.15, 0.30],          # person-mask failures
    p_glare_gap=[0.0, 0.01, 0.03, 0.06],          # glare/blur false gaps on stocked slots
    blocks_per_h=[10, 25, 50, 100],               # shopper traffic
    frame_dt_s=[1.0, 2.0, 4.0, 8.0],              # camera frame interval
)


def _sens_job(args):
    from dataclasses import replace
    from sim.simulate import methods
    param, value, base_cfg, seeds, engine_params = args
    cfg = replace(base_cfg, **{param: value})
    ms = methods(EngineConfig(**engine_params) if engine_params else None)
    keep = ("Majority vote 3 of 5 + mask", "ShelfTwin engine")
    parts = {k: [] for k in keep}
    for sd in seeds:
        w = simulate_world(sd, cfg)
        for k in keep:
            parts[k].append(run_method(ms[k], w))
    return dict(param=param, value=value, **{k: aggregate(v) for k, v in parts.items()})


def sensitivity(base_cfg: SimConfig | None = None, sweeps: dict | None = None, seeds=(0, 1, 2, 3),
                engine_params: dict | None = None, workers: int | None = None) -> list[dict]:
    """How false alerts / recall / delay degrade as each noise source gets worse (one at a time),
    for the engine and the strongest baseline. All (param, value) jobs run in parallel."""
    base_cfg = base_cfg or SimConfig(hours=4.0)
    sweeps = sweeps or SENSITIVITY
    jobs = [(p, v, base_cfg, list(seeds), engine_params) for p, vs in sweeps.items() for v in vs]
    with Pool(min(len(jobs), workers or os.cpu_count() or 1)) as pool:
        return pool.map(_sens_job, jobs)


def attribute_false_alerts(engine_params: dict | None = None, sim_cfg: SimConfig | None = None,
                           seeds=(0, 1, 2, 3), lookback_frames: int = 10) -> dict:
    """Error analysis: for every false ACT, which noise sources were present in the frames
    leading up to it? (A false alert can have several contributing causes.)"""
    from collections import Counter
    sim_cfg = sim_cfg or SimConfig()
    cfg = EngineConfig(**(engine_params or {}))
    causes, n_false, examples = Counter(), 0, []
    delays = []
    for sd in seeds:
        w = simulate_world(sd, sim_cfg)
        sc, alerts = run_method(lambda: TemporalEngine(cfg), w, return_alerts=True)
        delays += list(sc["delays"])
        for s, f in alerts:
            if w.needs[s, f]:
                continue
            n_false += 1
            f0 = max(0, f - lookback_frames)
            present = [k for k, arr in w.causes.items() if arr[s, f0:f + 1].any()]
            if not present:
                present = ["none of the modelled sources"]
            for k in present:
                causes[k] += 1
            if len(examples) < 20:
                examples.append(dict(seed=sd, slot=int(s), frame=int(f), causes=present,
                                     coverage=[round(float(x), 2) for x in w.coverage[s, f0:f + 1]],
                                     visibility=[round(float(x), 2) for x in w.visibility[s, f0:f + 1]],
                                     units=[int(x) for x in w.units[s, f0:f + 1]]))
    d = np.asarray(delays, float)
    return dict(n_false=n_false, cause_counts=dict(causes), examples=examples,
                delay_quantiles={q: float(np.percentile(d, q)) for q in (50, 90, 95, 99)} if d.size else {})
