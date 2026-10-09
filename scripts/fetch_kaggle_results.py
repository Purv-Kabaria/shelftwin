"""Download the small result files of the public Kaggle run into results/kaggle/.

    python scripts/fetch_kaggle_results.py            # latest completed version of the notebook
    python scripts/update_deck.py                     # then rebuild the deck from them

Uses Kaggle's public kernel-output API (no credentials needed for a public notebook). Only
reports, metrics and figures are fetched: no weights, datasets or caches.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://www.kaggle.com/api/v1/kernels/output"
KEEP = (".json", ".png", ".md", ".csv", ".mp4")
MAX_BYTES = 25_000_000


def list_output(user: str, slug: str) -> tuple[list[dict], str]:
    files, log, token = [], "", None
    while True:
        q = {"userName": user, "kernelSlug": slug}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=120) as r:
            d = json.loads(r.read())
        files += d.get("files") or []
        log = log or d.get("logNullable") or ""
        token = d.get("nextPageToken")
        if not d.get("hasNextPageToken") or not token:
            return files, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="purv2k2")
    ap.add_argument("--slug", default="notebook77852dc329")
    ap.add_argument("--out", default=str(ROOT / "results" / "kaggle"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files, log = list_output(a.user, a.slug)
    print(f"{len(files)} output files in the latest version")
    if not files:
        print("No outputs yet: the latest version is probably still running. Try again when it has finished.")
        return
    n = 0
    for f in files:
        name = f["fileName"]
        if not name.startswith("outputs/") or not name.endswith(KEEP):
            continue
        dst = out / name[len("outputs/"):]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f["url"], timeout=300) as r:
            data = r.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            print("skip (too large):", name)
            continue
        dst.write_bytes(data)
        n += 1
        print(f"{len(data):>10,}  {dst}")
    if log:
        entries = json.loads(log)
        (out / "kaggle_run.log").write_text("".join(e.get("data", "") for e in entries), encoding="utf-8")
        print("log saved:", out / "kaggle_run.log")
    progress = out / "progress_final.json"
    if progress.exists():
        print("phase status:", {k: v.get("status") for k, v in json.loads(progress.read_text()).items()})
    print(f"fetched {n} files")


if __name__ == "__main__":
    main()
