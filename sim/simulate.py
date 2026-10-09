"""Shelf simulator for the temporal engine (spec Section 7.1).

Generates a ground-truth world (units per slot, shopper occlusions, refills) and
noisy per-frame detector outputs, then replays the *same* observations through
every decision method so the comparison is fair.

Usage:
    python sim/simulate.py 20            # 20 runs (seeds 0-19), writes sim/results.json
    python sim/simulate.py 3 --quick     # 3 runs, 2 trading hours each
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelftwin.engine import (ACT, EngineConfig, MajorityVoteBaseline,  # noqa: E402
                              SingleFrameBaseline, TemporalEngine)


@dataclass
class SimConfig:
    n_slots: int = 24
    frame_dt_s: float = 2.0
    hours: float = 8.0
    units_min: int = 6
    units_max: int = 12
    demand_min_per_h: float = 0.5
    demand_max_per_h: float = 6.0
    blocks_per_h: float = 25.0
    block_logmean: float = 1.8
    block_logsd: float = 0.7
    block_min_s: float = 2.0
    block_max_s: float = 30.0
    p_gap_when_blocked: float = 0.70
    p_mask_miss: float = 0.05
    p_glare_gap: float = 0.01
    p_empty_misread: float = 0.03
    restock_at_units: int = 2
    refill_min_s: float = 600.0
    refill_max_s: float = 2400.0


@dataclass
class World:
    """Arrays shaped (n_slots, n_frames)."""
    t: np.ndarray
    needs: np.ndarray       # bool: slot needs restock (<= 2 units)
    coverage: np.ndarray    # detector coverage
    visibility: np.ndarray  # mask-reported visibility
    quality: np.ndarray
    causes: dict | None = None   # per-frame noise sources, for error analysis
    units: np.ndarray | None = None


def simulate_world(seed: int, cfg: SimConfig) -> World:
    rng = np.random.default_rng(seed)
    n_frames = int(cfg.hours * 3600 / cfg.frame_dt_s)
    S = cfg.n_slots
    t = np.arange(n_frames) * cfg.frame_dt_s

    cap = rng.integers(cfg.units_min, cfg.units_max + 1, size=S)
    rate = rng.uniform(cfg.demand_min_per_h, cfg.demand_max_per_h, size=S) / 3600.0

    units = np.zeros((S, n_frames), dtype=np.int32)
    for s in range(S):
        u = int(cap[s])
        refill_at = None
        lam = rate[s] * cfg.frame_dt_s
        takes = rng.poisson(lam, size=n_frames)
        for f in range(n_frames):
            if refill_at is not None and t[f] >= refill_at:
                u = int(cap[s])
                refill_at = None
            u = max(0, u - int(takes[f]))
            if u <= cfg.restock_at_units and refill_at is None:
                refill_at = t[f] + rng.uniform(cfg.refill_min_s, cfg.refill_max_s)
            units[s, f] = u
    needs = units <= cfg.restock_at_units

    # Shopper occlusion episodes (per slot Poisson process)
    blocked = np.zeros((S, n_frames), dtype=bool)
    mask_missed = np.zeros((S, n_frames), dtype=bool)
    horizon = cfg.hours * 3600
    for s in range(S):
        n_ev = rng.poisson(cfg.blocks_per_h * cfg.hours)
        starts = rng.uniform(0, horizon, size=n_ev)
        durs = np.clip(rng.lognormal(cfg.block_logmean, cfg.block_logsd, size=n_ev),
                       cfg.block_min_s, cfg.block_max_s)
        missed = rng.random(n_ev) < cfg.p_mask_miss
        for st, du, mi in zip(starts, durs, missed):
            f0 = int(np.ceil(st / cfg.frame_dt_s))
            f1 = int(np.floor((st + du) / cfg.frame_dt_s))
            if f1 < f0:
                continue
            blocked[s, f0:f1 + 1] = True
            if mi:
                mask_missed[s, f0:f1 + 1] = True

    # Detector coverage given true units
    r = rng.random((S, n_frames))
    cov = np.where(units >= 3, rng.uniform(0.55, 1.0, (S, n_frames)), 0.0)
    low = (units >= 1) & (units <= 2)
    cov = np.where(low, rng.uniform(0.05, 0.35, (S, n_frames)), cov)
    empty = units == 0
    misread = rng.random((S, n_frames)) < cfg.p_empty_misread
    cov = np.where(empty & ~misread, rng.uniform(0.0, 0.06, (S, n_frames)), cov)
    cov = np.where(empty & misread, rng.uniform(0.5, 0.9, (S, n_frames)), cov)
    glare = (units >= 3) & (r < cfg.p_glare_gap)
    cov = np.where(glare & ~blocked, rng.uniform(0.0, 0.2, (S, n_frames)), cov)

    # While blocked, the detector sees a gap 70% of the time
    blocked_gap = blocked & (rng.random((S, n_frames)) < cfg.p_gap_when_blocked)
    cov = np.where(blocked_gap, rng.uniform(0.0, 0.3, (S, n_frames)), cov)

    vis = np.where(blocked, rng.uniform(0.0, 0.5, (S, n_frames)), 1.0)
    vis = np.where(blocked & mask_missed, 1.0, vis)
    qual = np.ones((S, n_frames))
    causes = dict(blocked_mask_ok=blocked & ~mask_missed, blocked_mask_missed=blocked & mask_missed,
                  glare=glare & ~blocked, empty_misread=empty & misread)
    return World(t=t, needs=needs, coverage=cov, visibility=vis, quality=qual, causes=causes, units=units)


def run_method(make, world: World, return_alerts: bool = False):
    S, F = world.coverage.shape
    method = make()
    alerts = []  # (slot, frame)
    for s in range(S):
        sid = f"s{s}"
        cov, vis, q = world.coverage[s], world.visibility[s], world.quality[s]
        for f in range(F):
            if method.update(sid, float(world.t[f]), float(cov[f]), float(vis[f]), float(q[f])) == ACT:
                alerts.append((s, f))
    sc = score(alerts, world)
    return (sc, alerts) if return_alerts else sc


def episodes(needs_row: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], needs_row.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))  # [start, end)


def score(alerts, world: World) -> dict:
    S, F = world.coverage.shape
    correct = sum(1 for s, f in alerts if world.needs[s, f])
    false = len(alerts) - correct
    by_slot: dict[int, list[int]] = {}
    for s, f in alerts:
        by_slot.setdefault(s, []).append(f)
    n_events, detected, delays = 0, 0, []
    for s in range(S):
        fs = np.array(sorted(by_slot.get(s, [])), dtype=int)
        for a, b in episodes(world.needs[s]):
            n_events += 1
            hit = fs[(fs >= a) & (fs < b)]
            if hit.size:
                detected += 1
                delays.append(world.t[hit[0]] - world.t[a])
    cam_hours = F * (world.t[1] - world.t[0]) / 3600.0
    return dict(alerts=len(alerts), correct=correct, false=false, events=n_events,
                detected=detected, delays=delays, camera_hours=cam_hours)


def aggregate(parts: list[dict]) -> dict:
    alerts = sum(p["alerts"] for p in parts)
    correct = sum(p["correct"] for p in parts)
    false = sum(p["false"] for p in parts)
    events = sum(p["events"] for p in parts)
    detected = sum(p["detected"] for p in parts)
    delays = np.concatenate([np.asarray(p["delays"], dtype=float) for p in parts]) if parts else np.array([])
    hours = sum(p["camera_hours"] for p in parts)
    return dict(
        false_alerts_per_cam_hour=false / hours if hours else 0.0,
        precision=correct / alerts if alerts else 0.0,
        recall=detected / events if events else 0.0,
        median_delay_s=float(np.median(delays)) if delays.size else float("nan"),
        p95_delay_s=float(np.percentile(delays, 95)) if delays.size else float("nan"),
        alerts=alerts, false_alerts=false, true_events=events, camera_hours=hours,
    )


def methods(engine_cfg: EngineConfig | None = None):
    return {
        "Single-frame": lambda: SingleFrameBaseline(use_mask=False),
        "Single-frame + person mask": lambda: SingleFrameBaseline(use_mask=True),
        "Majority vote 3 of 5 + mask": lambda: MajorityVoteBaseline(n=5, k=3, use_mask=True),
        "ShelfTwin engine": lambda: TemporalEngine(engine_cfg or EngineConfig()),
    }


def run(n_runs: int = 20, sim_cfg: SimConfig | None = None, engine_cfg: EngineConfig | None = None,
        only: list[str] | None = None, verbose: bool = True) -> dict:
    sim_cfg = sim_cfg or SimConfig()
    ms = methods(engine_cfg)
    if only:
        ms = {k: v for k, v in ms.items() if k in only}
    parts: dict[str, list[dict]] = {k: [] for k in ms}
    t0 = time.time()
    for seed in range(n_runs):
        world = simulate_world(seed, sim_cfg)
        for name, make in ms.items():
            parts[name].append(run_method(make, world))
        if verbose:
            print(f"  run {seed + 1}/{n_runs} done ({time.time() - t0:.0f}s)", flush=True)
    return {name: aggregate(p) for name, p in parts.items()}


def print_table(results: dict) -> None:
    hdr = f"{'Method':<30} {'FA/cam-h':>9} {'Prec':>7} {'Recall':>7} {'Med(s)':>7} {'p95(s)':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<30} {r['false_alerts_per_cam_hour']:>9.2f} {r['precision'] * 100:>6.1f}% "
              f"{r['recall'] * 100:>6.1f}% {r['median_delay_s']:>7.0f} {r['p95_delay_s']:>7.0f}")
    any_r = next(iter(results.values()))
    print(f"\nTrue restock events: {any_r['true_events']}, camera-hours: {any_r['camera_hours']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="2 trading hours per run")
    ap.add_argument("--out", default=str(ROOT / "sim" / "results.json"))
    args = ap.parse_args()
    sim_cfg = SimConfig(hours=2.0) if args.quick else SimConfig()
    res = run(args.runs, sim_cfg)
    print_table(res)
    Path(args.out).write_text(json.dumps({"sim_config": asdict(sim_cfg), "n_runs": args.runs,
                                          "results": res}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
