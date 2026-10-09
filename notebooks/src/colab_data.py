"""Colab helpers: Kaggle credentials, dataset download via kagglehub, Drive caching.

Colab disks are wiped every session, so expensive artefacts (prepared datasets, checkpoints)
are cached on Google Drive and restored on the next run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from kaggle_common import INPUT_ROOT, disk_free_gb, log


def setup_kaggle_credentials() -> bool:
    """Kaggle token from Colab Secrets (KAGGLE_USERNAME / KAGGLE_KEY), env vars, or ~/.kaggle/kaggle.json."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    try:
        from google.colab import userdata  # type: ignore
        u, k = userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY")
        if u and k:
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = u, k
            log("Kaggle credentials loaded from Colab Secrets")
            return True
    except Exception:
        pass
    for p in (Path.home() / ".kaggle" / "kaggle.json", Path("/content/kaggle.json")):
        if p.exists():
            dst = Path.home() / ".kaggle" / "kaggle.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if p != dst:
                shutil.copy(p, dst)
            os.chmod(dst, 0o600)
            log("Kaggle credentials found at", p)
            return True
    log("No Kaggle credentials (public downloads may still work without them).")
    return False


def kaggle_download(slugs: list[str], link_name: str, min_free_gb: float = 30.0) -> Path | None:
    """Try each Kaggle dataset slug with kagglehub; link the result into INPUT_ROOT/<link_name>."""
    link = INPUT_ROOT / link_name
    if link.exists() and any(link.iterdir()):
        log("Already available:", link)
        return link
    free = disk_free_gb("/content")
    if free < min_free_gb:
        log(f"Only {free:.0f} GB free disk; skipping Kaggle download of {link_name}")
        return None
    try:
        import kagglehub  # noqa: F401
    except ImportError:
        subprocess.run(["pip", "install", "-q", "kagglehub"])
    import kagglehub
    for slug in slugs:
        try:
            log(f"Downloading {slug} via kagglehub (several GB, ~5-20 min)...")
            path = Path(kagglehub.dataset_download(slug))
            INPUT_ROOT.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(path, target_is_directory=True)
            log("Downloaded:", slug, "->", path)
            return link
        except Exception as e:
            log(f"kagglehub failed for {slug}: {type(e).__name__}: {e}")
    return None


def remove_download(link_name: str):
    """Free disk after a dataset has been prepared (keeps the Drive cache)."""
    link = INPUT_ROOT / link_name
    if link.is_symlink():
        target = link.resolve()
        link.unlink()
        shutil.rmtree(target, ignore_errors=True)
        log("Removed raw download", target)


def cache_dir_to_drive(src: Path, tar_path: Path) -> None:
    """Uncompressed tar (JPEGs are already compressed): one big file copies fast to Drive."""
    if tar_path.exists():
        return
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path("/content") / (tar_path.name + ".part")
    with tarfile.open(tmp, "w") as tar:
        tar.add(src, arcname=src.name)
    shutil.copy(tmp, str(tar_path) + ".part")
    os.replace(str(tar_path) + ".part", tar_path)  # atomic on the same filesystem
    tmp.unlink(missing_ok=True)
    log(f"Cached {src} -> {tar_path} ({tar_path.stat().st_size / 2**30:.2f} GB)")


def restore_dir_from_drive(tar_path: Path, dst_parent: Path) -> bool:
    if not tar_path.exists():
        return False
    dst_parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path) as tar:
            try:
                tar.extractall(dst_parent, filter="data")  # Python >= 3.12 safe extraction
            except TypeError:
                tar.extractall(dst_parent)
        log("Restored", tar_path, "->", dst_parent)
        return True
    except Exception as e:
        log("Cache restore failed (will rebuild):", e)
        return False
