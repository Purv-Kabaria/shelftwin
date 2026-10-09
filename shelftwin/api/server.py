"""FastAPI server: dashboard + staff actions + replay.

Demo (simulated store, no models needed):
    python -m shelftwin.api.server --demo --speed 20
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..notify import TelegramNotifier, task_message
from ..tasks import TransitionError
from ..twin import Twin

STATIC = Path(__file__).parent / "static"


class SnoozeBody(BaseModel):
    minutes: float = 30
    reason: str = ""


class NoteBody(BaseModel):
    note: str = ""


def create_app(twin: Twin, on_done=None, lang: str = "en") -> FastAPI:
    app = FastAPI(title="ShelfTwin")
    app.state.twin = twin
    app.state.lang = lang

    def slot_dict(sid):
        i = twin.slots.get(sid)
        return dict(aisle=i.aisle, shelf=i.shelf, sku=i.sku_name) if i else {}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state():
        return twin.snapshot()

    @app.get("/api/tasks")
    def tasks(lang: str | None = None, include_closed: bool = False):
        with twin.lock:
            ts = sorted(twin.service.tasks.values(), key=lambda t: (not t.active, -t.priority, -t.created_ts))
            out = []
            for t in ts:
                if not include_closed and not t.active:
                    continue
                d = t.to_dict()
                d["message"] = task_message(d, slot_dict(t.slot_id), lang or app.state.lang)
                out.append(d)
            return out[:200]

    def act(task_id: int, fn):
        with twin.lock:
            if task_id not in twin.service.tasks:
                raise HTTPException(404, "task not found")
            try:
                r = fn(twin.service.tasks[task_id])
            except TransitionError as e:
                raise HTTPException(409, str(e))
            twin.store.upsert_task(twin.service.tasks[task_id].to_dict())
            return r or {"ok": True}

    @app.post("/api/tasks/{task_id}/ack")
    def ack(task_id: int):
        return act(task_id, lambda t: twin.service.ack(task_id, twin.now))

    @app.post("/api/tasks/{task_id}/done")
    def done(task_id: int):
        def f(t):
            twin.service.done(task_id, twin.now)
            if on_done:
                on_done(t.slot_id)  # demo: staff actually refills the shelf
        return act(task_id, f)

    @app.post("/api/tasks/{task_id}/snooze")
    def snooze(task_id: int, body: SnoozeBody):
        return act(task_id, lambda t: twin.service.snooze(task_id, twin.now + body.minutes * 60, twin.now, body.reason))

    @app.post("/api/tasks/{task_id}/wrong")
    def wrong(task_id: int, body: NoteBody):
        def f(t):
            fb = twin.service.wrong_alert(task_id, twin.now, body.note)
            twin.store.feedback(task_id, "staff", "wrong_alert", body.note, twin.now)
            return fb
        return act(task_id, f)

    @app.get("/api/slots/{slot_id}/replay")
    def replay(slot_id: str, seconds: float = 300):
        return twin.store.replay(slot_id, twin.now - seconds, twin.now + 1)

    @app.get("/api/slots/{slot_id}/explain")
    def explain(slot_id: str):
        if slot_id not in twin.views:
            raise HTTPException(404, "slot not found")
        with twin.lock:
            return twin.explain(slot_id)

    return app


def run_demo(speed: float, host: str, port: int, lang: str, db: str):
    import uvicorn
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sim.live import LiveStore
    from ..store import TwinStore

    from ..engine import EngineConfig
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "engine.yaml"
    engine_cfg = None
    if cfg_path.exists():
        import yaml
        engine_cfg = EngineConfig.from_dict(yaml.safe_load(cfg_path.read_text()))
    live = LiveStore()
    twin = Twin(live.infos, engine_cfg=engine_cfg, store=TwinStore(db))
    tg = TelegramNotifier(lang=lang, slot_lookup=lambda sid: dict(aisle=live.infos[sid].aisle,
                                                                   shelf=live.infos[sid].shelf,
                                                                   sku=live.infos[sid].sku_name))
    if tg.enabled:
        twin.service.notifiers.append(tg)
    app = create_app(twin, on_done=live.refill, lang=lang)

    def loop():
        period = live.dt / max(speed, 0.1)
        while True:
            t0 = time.time()
            for sid, ts, cov, vis, q in live.step():
                twin.observe(sid, ts, cov, vis, q, frame_id=f"f{int(ts / live.dt)}")
            twin.tick(live.t)
            time.sleep(max(0.0, period - (time.time() - t0)))

    threading.Thread(target=loop, daemon=True).start()
    print(f"ShelfTwin demo at http://{host}:{port}  (simulated time x{speed})")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run against the simulated store")
    ap.add_argument("--speed", type=float, default=20.0, help="simulated seconds per real second")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lang", default="en", choices=["en", "hi", "gu"])
    ap.add_argument("--db", default=":memory:")
    a = ap.parse_args()
    if not a.demo:
        print("Use --demo for the simulated store, or scripts/run_video.py for camera/video input.")
        return
    run_demo(a.speed, a.host, a.port, a.lang, a.db)


if __name__ == "__main__":
    main()
