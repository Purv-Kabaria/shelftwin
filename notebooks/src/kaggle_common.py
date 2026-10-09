"""Shared helpers for ShelfTwin Kaggle notebooks (embedded into each notebook).

Everything here is defensive: no hardcoded dataset paths, clear errors, and
environment fixes for known Kaggle pitfalls (W&B login prompts, NCCL hangs).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("ST_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("ST_WORK", "/kaggle/working"))
TMP = Path(os.environ.get("ST_TMP", "/tmp/shelftwin"))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SESSION_START = time.time()


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def quiet_loggers():
    """Ultralytics auto-enables W&B/Comet/ClearML/MLflow/DVC when installed; on Kaggle a
    W&B login prompt can block a non-interactive run. Disable all of them."""
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["COMET_MODE"] = "disabled"
    os.environ["CLEARML_OFF"] = "1"
    # Known T4 x2 NCCL peer-to-peer hangs in containers; harmless when single GPU
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    try:
        from ultralytics import settings
        settings.update({k: False for k in ("wandb", "comet", "clearml", "mlflow", "dvc",
                                            "neptune", "raytune", "tensorboard", "hub")
                         if k in settings})
    except Exception as e:  # ultralytics not installed yet or older settings keys
        log("note: could not update ultralytics settings:", e)


def pip_install(*pkgs: str, quiet: bool = True) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"] + (["-q"] if quiet else []) + list(pkgs)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        log("pip install failed for", pkgs, "- continuing; check that Internet is ON in notebook settings")
    return r.returncode == 0


def gpu_info() -> list[dict]:
    try:
        import torch
        return [dict(i=i, name=torch.cuda.get_device_name(i),
                     mem_gb=round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1))
                for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def disk_free_gb(path: Path | str = "/") -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


def elapsed_h() -> float:
    return (time.time() - SESSION_START) / 3600


def walk_limited(root: Path, max_depth: int = 6):
    """os.walk with a depth limit; yields (dirpath, dirnames, filenames)."""
    root = Path(root)
    if not root.exists():
        return
    base = len(root.parts)
    for dp, dns, fns in os.walk(root, followlinks=True):
        depth = len(Path(dp).parts) - base
        if depth >= max_depth:
            dns[:] = []
        dns.sort()
        yield dp, dns, fns


def find_files(name_pred, root: Path = INPUT_ROOT, max_depth: int = 7, limit: int = 1000) -> list[Path]:
    out = []
    for dp, _, fns in walk_limited(root, max_depth):
        for f in fns:
            if name_pred(f):
                out.append(Path(dp) / f)
                if len(out) >= limit:
                    return out
    return out


def print_input_tree(root: Path = INPUT_ROOT, max_depth: int = 3, max_entries: int = 12):
    """Prints what is attached, so a missing dataset is easy to diagnose."""
    log(f"Contents of {root} (depth {max_depth}):")
    if not Path(root).exists():
        print("  (does not exist)")
        return
    for dp, dns, fns in walk_limited(root, max_depth):
        depth = len(Path(dp).parts) - len(Path(root).parts)
        ind = "  " * depth
        print(f"{ind}{Path(dp).name}/  ({len(dns)} dirs, {len(fns)} files)")
        for f in fns[:3]:
            print(f"{ind}  {f}")
        if len(dns) > max_entries:
            dns[:] = dns[:max_entries]


def find_weights(patterns=("best.pt",), roots=(INPUT_ROOT, WORK), must_contain: str | None = None) -> Path | None:
    """Find weights from a previous notebook's output attached as input. Newest wins."""
    cands = []
    for r in roots:
        for p in find_files(lambda f: f in patterns, Path(r), max_depth=9):
            if must_contain is None or must_contain in str(p):
                cands.append(p)
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def save_json(obj, path: Path | str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def _progress_line(raw: bytes) -> str:
    """tqdm redraws a bar with a carriage return; keep only its final state."""
    s = raw.decode("utf-8", errors="replace").rstrip("\r\n ")  # CRLF-terminated lines too
    return s.split("\r")[-1].rstrip()


STOPPED_BY_POLICY = -2


def cpu_workers(reserve: int = 0) -> int:
    return max(1, (os.cpu_count() or 2) - reserve)


def find_cached(name: str) -> Path | None:
    """A cache file produced by an earlier session: in WORK/cache (Colab: Drive) or in an
    attached previous notebook output under INPUT_ROOT (Kaggle)."""
    p = WORK / "cache" / name
    if p.exists():
        return p
    hits = find_files(lambda f: f == name, INPUT_ROOT, max_depth=6, limit=5)
    return hits[0] if hits else None


def cached_dir(name: str, dst: Path, build, keep_cache: bool = True) -> Path:
    """Return `dst`, restoring it from `<name>.tar` if an earlier session cached it, else
    building it with `build(dst)` and caching it to WORK/cache (saved as notebook output on
    Kaggle, on Drive in Colab). Restoring a single tar is much faster than rebuilding."""
    import tarfile
    marker = dst / ".complete"
    if marker.exists():
        log("Using prepared", dst)
        return dst
    tar = find_cached(name + ".tar")
    if tar is not None:
        log(f"Restoring {dst.name} from cache {tar} ...")
        t0 = time.time()
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tar) as tf:
                try:
                    tf.extractall(dst.parent, filter="data")
                except TypeError:
                    tf.extractall(dst.parent)
            if marker.exists():
                log(f"  restored in {time.time() - t0:.0f}s")
                return dst
            log("  cache incomplete; rebuilding")
        except Exception as e:
            log("  cache restore failed (", e, "); rebuilding")
    if dst.exists():
        shutil.rmtree(dst)
    build(dst)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    if keep_cache:
        out = WORK / "cache" / (name + ".tar")
        out.parent.mkdir(parents=True, exist_ok=True)
        part = Path(str(out) + ".part")
        t0 = time.time()
        with tarfile.open(part, "w") as tf:  # uncompressed: JPEGs don't compress further
            tf.add(dst, arcname=dst.name)
        os.replace(part, out)
        log(f"Cached {dst.name} -> {out} ({out.stat().st_size / 2**30:.2f} GB, {time.time() - t0:.0f}s)")
    return dst


def save_decision(topic: str, decision: dict, evidence: dict | None = None) -> Path:
    """Experiment notebooks record what won (and why) for the final training notebooks."""
    p = WORK / "outputs" / f"decision_{topic}.json"
    save_json(dict(topic=topic, decision=decision, evidence=evidence or {},
                   created=time.strftime("%Y-%m-%d %H:%M:%S")), p)
    log(f"DECISION [{topic}]:", decision)
    return p


def load_decision(topic: str) -> dict:
    """Decision from an attached experiment notebook output (or this session); {} if none."""
    name = f"decision_{topic}.json"
    p = WORK / "outputs" / name
    if not p.exists():
        hits = find_files(lambda f: f == name, INPUT_ROOT, max_depth=6, limit=5)
        p = max(hits, key=lambda q: q.stat().st_mtime) if hits else None
    if p is None:
        log(f"No decision_{topic}.json attached: using the notebook defaults.")
        return {}
    d = json.loads(Path(p).read_text()).get("decision", {})
    log(f"Applying experiment decision [{topic}] from {p}: {d}")
    return d


def mark_session(tag: str) -> None:
    """Write a marker so a later session of the SAME notebook can recognise this output."""
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / f"st_session_{tag}.json"
    n = json.loads(p.read_text()).get("sessions", 0) if p.exists() else 0
    p.write_text(json.dumps({"tag": tag, "sessions": n + 1, "last": time.strftime("%Y-%m-%d %H:%M:%S")}))


def restore_previous_outputs(tag: str, subdirs=("runs", "outputs", "cache")) -> list[str]:
    """Kaggle resume: copy checkpoints/outputs from an attached previous version of THIS
    notebook (Add Input -> Your Work -> this notebook) into WORK. Recognised by its
    st_session_<tag>.json marker, so other notebooks' outputs are never mixed in."""
    marker = f"st_session_{tag}.json"
    srcs = sorted((Path(dp) for dp, _, fns in walk_limited(INPUT_ROOT, 4) if marker in fns),
                  key=lambda p: (p / marker).stat().st_mtime)
    if not srcs:
        return []
    src = srcs[-1]
    restored = []
    for sub in subdirs:
        if (src / sub).is_dir() and not ((WORK / sub).exists() and any((WORK / sub).iterdir())):
            shutil.copytree(src / sub, WORK / sub, dirs_exist_ok=True)
            restored.append(sub)
    if (src / marker).exists() and not (WORK / marker).exists():
        shutil.copy(src / marker, WORK / marker)
    log(f"Resuming from previous session output {src}: restored {restored}")
    return restored


def run_streaming(cmd: list[str], log_path: Path, stall_min: float = 20.0, env=None,
                  should_stop=None, echo=None, prefix: str = "") -> int:
    """Run a subprocess, mirror its output, kill it if silent for `stall_min` minutes.

    Output is read as raw bytes: every byte (including tqdm carriage-return redraws) resets
    the stall timer, but only completed lines are printed, so a 10 h run does not flood the
    notebook. Returns the exit code (-9 when killed by the stall watchdog)."""
    import signal
    import threading
    log_path.parent.mkdir(parents=True, exist_ok=True)
    posix = hasattr(os, "killpg")
    # New session so the whole process group (DDP spawns workers) can be killed together
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                         start_new_session=posix)
    last = [time.time()]

    def pump():
        buf = b""
        fd = p.stdout.fileno()
        with open(log_path, "ab") as f:
            while True:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                last[0] = time.time()
                f.write(chunk)
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    txt = _progress_line(ln)
                    if txt.strip() and (echo is None or echo(txt)):
                        print(prefix + txt, flush=True)
                if len(buf) > 1_000_000:  # a very long redraw-only stream: keep the tail
                    buf = buf[-10_000:]
            if buf.strip() and (echo is None or echo(_progress_line(buf))):
                print(prefix + _progress_line(buf), flush=True)

    def kill_group():
        if posix:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    while p.poll() is None:
        time.sleep(min(15.0, stall_min * 60 / 4))
        if should_stop is not None and should_stop():
            log("Stop policy triggered (session time budget): stopping training at a checkpoint")
            kill_group()
            p.kill()
            p.wait()
            th.join(timeout=5)
            return STOPPED_BY_POLICY
        if time.time() - last[0] > stall_min * 60:
            log(f"WATCHDOG: no output for {stall_min} min, killing process group {p.pid}")
            kill_group()
            p.kill()
            p.wait()
            th.join(timeout=5)
            return -9
    th.join(timeout=30)
    kill_group()  # reap any DDP worker left behind so the next attempt gets free GPUs
    return p.returncode
