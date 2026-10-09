"""Parallel, resumable ablation runs for Kaggle T4 x2.

Short experiments are faster as *one run per GPU* than as DDP runs (no DDP overhead, two runs
at once, and Kaggle bills session time, not per GPU). Every run is an ordinary resumable
yolo_runner run, so re-running the notebook skips finished runs and continues unfinished ones.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from kaggle_common import WORK, cpu_workers, gpu_info, log
import yolo_runner as yr

EPOCH_LINE = ("all ", "epochs completed", "Error", "Traceback", "OutOfMemory")


def _echo(line: str) -> bool:
    # Ultralytics' validation summary row starts with 'all'; print only that and errors
    s = line.strip()
    return s.startswith("all ") or any(k in line for k in EPOCH_LINE[1:])


def run_parallel(runs: list[dict], budget_h: float, stall_min: float = 20.0) -> dict[str, dict]:
    """runs: list of train_args (unique 'name'). Each GPU takes the next run from a queue.
    Stops starting/continuing runs when the budget is used; unfinished runs resume next session."""
    gpus = gpu_info() or [dict(i="cpu")]
    deadline = time.time() + budget_h * 3600
    q: queue.Queue = queue.Queue()
    for r in runs:
        q.put(r)
    results: dict[str, dict] = {}
    per_gpu_workers = max(2, cpu_workers() // len(gpus))
    log(f"{len(runs)} runs on {len(gpus)} device(s), {per_gpu_workers} data workers each, budget {budget_h:.2f} h")

    def worker(dev):
        while True:
            try:
                args = q.get_nowait()
            except queue.Empty:
                return
            left = (deadline - time.time()) / 3600
            if left < 0.05:
                results[args["name"]] = dict(skipped="no time left this session")
                continue
            a = dict(args, workers=args.get("workers") or per_gpu_workers)
            try:
                t0 = time.time()
                r = yr.train(a, budget_h=left, multi_gpu=False, device=str(dev), stall_min=stall_min, echo=_echo)
                r["train_minutes_this_session"] = (time.time() - t0) / 60
                results[args["name"]] = r
                log(f"[{args['name']}] finished={r['finished']} epochs={r['epochs_done']}")
            except Exception as e:  # one failed experiment must not stop the others
                results[args["name"]] = dict(error=f"{type(e).__name__}: {e}")
                log(f"[{args['name']}] FAILED: {e}")

    ths = [threading.Thread(target=worker, args=(g["i"],), daemon=True) for g in gpus]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return results


def evaluate_runs(runs: list[dict], data_yaml: Path, split: str, results_path: Path,
                  imgsz_key: str = "imgsz", latency: bool = True, extra=None) -> list[dict]:
    """Evaluate every finished run once (cached in results_path, JSON lines)."""
    done = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            d = json.loads(line)
            done[d["name"]] = d
    rows = []
    for a in runs:
        name = a["name"]
        if name in done:
            rows.append(done[name]); continue
        w = yr.run_dir(name) / "weights" / "best.pt"
        last = yr.run_dir(name) / "weights" / "last.pt"
        if not (w.exists() and last.exists() and yr.ckpt_epoch(last) == -1):
            log(f"[{name}] not finished yet; skipped in evaluation")
            continue
        m = yr.evaluate(w, data_yaml, imgsz=a[imgsz_key], split=split, max_det=1000, iou=0.6)
        row = dict(name=name, **{k: a.get(k) for k in ("model", "imgsz", "mosaic", "epochs")}, **m,
                   epochs_done=yr.epochs_done(name))
        if latency:
            import det_analysis as da
            row["latency_ms_t4_fp16"] = da.latency_ms(w, a[imgsz_key])
        if extra is not None:
            row.update(extra(a, w))
        rows.append(row)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
    return rows


def pick(rows: list[dict], metric: str = "mAP50_95", tol: float = 0.01, speed_key: str = "latency_ms_t4_fp16") -> dict | None:
    """Best metric; among runs within `tol` of the best, the fastest (edge deployment matters)."""
    rows = [r for r in rows if r.get(metric) is not None]
    if not rows:
        return None
    best = max(r[metric] for r in rows)
    near = [r for r in rows if r[metric] >= best - tol]
    return min(near, key=lambda r: (r.get(speed_key) or 1e9, -r[metric]))


def table(rows: list[dict], cols: list[str]) -> str:
    out = [" | ".join(cols), " | ".join("---" for _ in cols)]
    for r in rows:
        out.append(" | ".join(f"{r.get(c):.4f}" if isinstance(r.get(c), float) else str(r.get(c)) for c in cols))
    return "\n".join(out)
