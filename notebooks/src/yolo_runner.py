"""Resumable, fast Ultralytics training for Kaggle and Colab free tiers.

* Runs as a subprocess script (Ultralytics multi-GPU DDP is fragile inside Jupyter).
* Uses both T4s on Kaggle (DDP); falls back to one GPU if DDP fails.
* Fixed-epoch cosine schedule spanning sessions: each session trains until its time budget,
  stops right AFTER an epoch checkpoint is written, and the next session resumes the same run
  (`resume=True`) from last.pt. Checkpoints live in WORK (Kaggle output / Google Drive).
* Stall watchdog kills a hung process group; corrupted last.pt falls back to best.pt.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from kaggle_common import (STOPPED_BY_POLICY, WORK, cpu_workers, elapsed_h, gpu_info, log, run_streaming,
                           save_json)

HERE = Path(__file__).resolve().parent
CKPT_SETTLE_S = (10.0, 120.0)  # a checkpoint is "complete and recent" in this age window


def run_dir(name: str) -> Path:
    return WORK / "runs" / name


def _find_run_file(run_name: str, fname: str) -> Path | None:
    p = run_dir(run_name) / ("weights" if fname.endswith(".pt") else "") / fname
    if p.exists():
        return p
    c = [q for q in WORK.rglob(fname) if run_name in q.parts]
    return max(c, key=lambda q: q.stat().st_mtime) if c else None


def ckpt_epoch(path: Path) -> int | None:
    """Epoch stored in an Ultralytics checkpoint: -1 once training finished (optimizer stripped),
    else the last completed epoch; None if unreadable (e.g. cut mid-write)."""
    try:
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=False)  # our own Ultralytics file
        return int(ck.get("epoch", -1))
    except Exception as e:
        log(f"cannot read {path}: {e}")
        return None


def epochs_done(name: str) -> int:
    r = run_dir(name) / "results.csv"
    if not r.exists():
        return 0
    return max(0, len([l for l in r.read_text().splitlines() if l.strip()]) - 1)


def _stop_policy(name: str, deadline_ts: float):
    """Stop when the next epoch would not finish before the deadline, right after last.pt is saved."""
    last = run_dir(name) / "weights" / "last.pt"
    state = dict(mtimes=[], seen=None)

    def should_stop() -> bool:
        now = time.time()
        if now >= deadline_ts:
            return True
        if not last.exists():
            return False
        m = last.stat().st_mtime
        if m != state["seen"]:
            state["seen"] = m
            state["mtimes"].append(m)
            return False  # just written; let the write finish
        if len(state["mtimes"]) >= 2:
            ep = (state["mtimes"][-1] - state["mtimes"][0]) / (len(state["mtimes"]) - 1)
        else:
            ep = None
        stable = CKPT_SETTLE_S[0] <= now - m <= CKPT_SETTLE_S[1]  # checkpoint complete and recent
        return bool(ep and stable and now + 1.15 * ep > deadline_ts)

    return should_stop


def train(train_args: dict, budget_h: float, multi_gpu: bool = True, stall_min: float = 25.0,
          device: str | None = None, echo=None) -> dict:
    """Train (or resume) run `train_args['name']` for at most `budget_h` hours this session.

    device: force a device (e.g. "1" to run an experiment on the second GPU in parallel).
    echo:   optional line filter for the console (full output always goes to logs/<name>.log).
    Returns dict(best=Path|None, last=Path|None, finished=bool, epochs_done=int)."""
    name = train_args["name"]
    rd = run_dir(name)
    w = rd / "weights"
    deadline = time.time() + budget_h * 3600
    gpus = gpu_info()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    if device is not None:
        use_ddp = "," in device
    else:
        log("GPUs:", gpus)
        use_ddp = multi_gpu and len(gpus) >= 2
        device = "0,1" if use_ddp else ("0" if gpus else "cpu")
    batch = int(train_args.get("batch", 16))
    if not use_ddp and train_args.get("batch_per_gpu"):
        batch = int(train_args["batch_per_gpu"])
    workers = train_args.get("workers") or max(2, cpu_workers() // (2 if use_ddp else 1))

    restart_from, restarted = None, False
    for attempt in range(5):
        if time.time() > deadline - min(120.0, 0.1 * budget_h * 3600):
            log("No session time left for training.")
            break
        last, best = w / "last.pt", w / "best.pt"
        ep = ckpt_epoch(last) if last.exists() else None
        if ep is None and last.exists() and best.exists():
            be = ckpt_epoch(best)
            if be is not None and be >= 0:
                log("last.pt unreadable; resuming from best.pt instead")
                shutil.copy(best, last)
                ep = be
        if ep == -1:
            log(f"Run '{name}' already finished.")
            break
        if restart_from is not None:
            # resume kept failing (e.g. GPU count changed): new run initialised from the old weights
            init = WORK / "runs" / f"{name}_init.pt"
            shutil.copy(restart_from, init)
            remaining = max(1, int(train_args.get("epochs", 50)) - epochs_done(name))
            shutil.rmtree(rd, ignore_errors=True)
            cfg = {k: v for k, v in train_args.items() if k != "batch_per_gpu"}
            cfg.update(model=str(init), epochs=remaining, project=str(WORK / "runs"), exist_ok=True,
                       device=device, batch=batch, workers=workers, warmup_epochs=0)
            log(f"Restarting '{name}' from saved weights for the remaining {remaining} epochs")
            restart_from = None
        elif ep is not None and ep >= 0:
            # resume keeps the original schedule; device/batch may change between sessions
            cfg = dict(model=str(last), resume=True, device=device, batch=batch, workers=workers)
            log(f"Resuming '{name}' after epoch {ep + 1} on device {device}")
        else:
            if rd.exists():
                shutil.rmtree(rd)
            cfg = {k: v for k, v in train_args.items() if k != "batch_per_gpu"}
            cfg.update(project=str(WORK / "runs"), exist_ok=True, device=device, batch=batch, workers=workers)
            if device == "cpu":
                log("WARNING: no GPU visible; enable a GPU accelerator. Training on CPU is very slow.")
                cfg["amp"] = False
        if device == "0,1" and cfg["batch"] % 2:
            cfg["batch"] += 1
        cfg_path = WORK / f"train_cfg_{name}.json"
        save_json(cfg, cfg_path)
        before = epochs_done(name)
        rc = run_streaming([sys.executable, str(HERE / "train_yolo_cli.py"), str(cfg_path)],
                           WORK / "logs" / f"{name}.log", stall_min=stall_min, env=env,
                           should_stop=_stop_policy(name, deadline), echo=echo,
                           prefix=f"[{name}] " if echo is not None else "")
        _free_gpu()
        progressed = epochs_done(name) > before
        log(f"attempt {attempt + 1}: exit {rc}, epochs done {epochs_done(name)} (+{epochs_done(name) - before})")
        if rc == STOPPED_BY_POLICY:
            break
        if rc == 0:
            continue  # loop re-reads last.pt: -1 means finished
        if device == "0,1" and not progressed:
            log("Multi-GPU run failed without progress; falling back to a single GPU")
            device = "0"
            batch = int(train_args.get("batch_per_gpu") or max(2, batch // 2))
            workers = train_args.get("workers") or cpu_workers()
        elif not progressed and last.exists() and not restarted:
            restart_from, restarted = last, True
        elif not progressed:
            break
    last, best = w / "last.pt", w / "best.pt"
    finished = last.exists() and ckpt_epoch(last) == -1
    res = dict(best=best if best.exists() else (last if last.exists() else None),
               last=last if last.exists() else None, finished=finished, epochs_done=epochs_done(name))
    log("Training status:", {k: str(v) for k, v in res.items()})
    if res["best"] is None:
        raise RuntimeError(f"No checkpoint produced for '{name}'. See {WORK / 'logs' / (name + '.log')}")
    if not finished:
        log("NOT FINISHED: run this notebook again (Kaggle: attach this version's output as input; "
            "Colab: just Run all) to resume from the last epoch.")
    return res


def _free_gpu():
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(3)


def evaluate(weights: Path, data_yaml: Path, imgsz: int, split: str = "test", max_det: int = 1000,
             iou: float = 0.6, batch: int = 16) -> dict:
    from ultralytics import YOLO
    import torch
    dev = 0 if torch.cuda.is_available() else "cpu"
    m = YOLO(str(weights))
    try:
        r = m.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, max_det=max_det, iou=iou,
                  conf=0.001, device=dev, half=dev != "cpu", plots=True, project=str(WORK / "runs"),
                  name=f"eval_{split}", exist_ok=True, verbose=True, workers=cpu_workers())
    except Exception as e:
        if split != "val":
            log(f"Eval on '{split}' failed ({e}); falling back to 'val'")
            return evaluate(weights, data_yaml, imgsz, "val", max_det, iou, batch)
        raise
    out = dict(split=split, mAP50=float(r.box.map50), mAP50_95=float(r.box.map),
               precision=float(r.box.mp), recall=float(r.box.mr))
    try:
        names = r.names
        out["per_class_AP50"] = {names[int(c)]: float(ap) for c, ap in zip(r.box.ap_class_index, r.box.ap50)}
    except Exception:
        pass
    log("Eval:", out)
    return out


def export_onnx(weights: Path, imgsz: int) -> Path | None:
    try:
        from ultralytics import YOLO
        p = YOLO(str(weights)).export(format="onnx", dynamic=True, simplify=True, imgsz=imgsz)
        return Path(p)
    except Exception as e:
        log("ONNX export failed (non-fatal):", e)
        return None


def package_outputs(files: dict[str, Path | None], out_dir: Path = WORK / "outputs") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for dst, src in files.items():
        if src is not None and Path(src).exists() and Path(src).resolve() != (out_dir / dst).resolve():
            shutil.copy2(src, out_dir / dst)
            log("saved", out_dir / dst)
    return out_dir
