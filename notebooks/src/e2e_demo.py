"""End-to-end demo: the full ShelfTwin pipeline on a real shelf photo with a scripted episode.

There is no public shelf *video* with ground-truth stockout events, so we script one on a
real SKU-110K test photo and run every stage on each frame: quality gate, change gate,
visibility mask, the trained detector, geometry and coverage, identity and planogram
alignment, temporal engine and tasks.

Timeline (2 s per frame, 8 min):
  * slot A runs out at 40 s (pack erased) and is refilled at 280 s
  * a "shopper" blocks other slots for 30 s and 14 s, and walks past A while it is empty
  * glare washes out the frame for 12 s
  * two different products C and D are swapped at 320 s (a displacement)

The shopper is a drawn silhouette, so the person mask is *oracle* (we know its pixels), with a
5% chance per frame that the mask misses it, matching the simulator assumptions. Everything
else comes from the trained models.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from shelftwin.engine import ACT, CLEAR
from shelftwin.gate import ChangeGate
from shelftwin.pipeline import BayPipeline, PipelineConfig
from shelftwin.tasks import SlotInfo
from shelftwin.twin import Twin
from shelftwin.vision.geometry import BayGeometry, Slot, cluster_rows
from shelftwin.vision.identity import Catalogue, IdentityConfig

DT = 2.0
N_FRAMES = 240
EMPTY_T = (40.0, 280.0)
BLOCKS = [("B", 100.0, 130.0), ("B", 200.0, 214.0), ("A", 60.0, 72.0)]
GLARE_T = (226.0, 238.0)
SWAP_T = 320.0


def square_crop(img: np.ndarray, frac: float = 0.75, size: int = 960) -> np.ndarray:
    """Central square crop resized: object scale similar to the detector's training crops."""
    h, w = img.shape[:2]
    s = int(min(h, w) * frac)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA)


def crop_boxes(boxes: np.ndarray, shape: tuple, frac: float = 0.75, size: int = 960) -> np.ndarray:
    """Map boxes (xyxy, original image pixels) into square_crop() coordinates; keep boxes fully inside."""
    h, w = shape[:2]
    s = int(min(h, w) * frac)
    y, x = (h - s) // 2, (w - s) // 2
    b = (np.asarray(boxes, float).reshape(-1, 4) - [x, y, x, y]) * (size / s)
    inside = (b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 2] <= size) & (b[:, 3] <= size)
    return b[inside]


def _shrink(b, f=0.06):
    x1, y1, x2, y2 = b
    dx, dy = (x2 - x1) * f, (y2 - y1) * f
    return [x1 + dx, y1 + dy, x2 - dx, y2 - dy]


def draft_slots(packs_px: list[list[float]], px_per_cm: float) -> list[Slot]:
    """Each detected facing becomes a slot (its own box, slightly shrunk); rows from box bottoms."""
    b = np.array(packs_px, dtype=float)
    rows = cluster_rows(b[:, 3], 0.5 * float(np.median(b[:, 3] - b[:, 1])))
    slots = []
    for r in np.unique(rows):
        idx = np.where(rows == r)[0]
        idx = idx[np.argsort(b[idx, 0])]
        for k, i in enumerate(idx, 1):
            x1, y1, x2, y2 = (v / px_per_cm for v in _shrink(b[i]))
            sid = f"R{int(r) + 1}-P{k}"
            slots.append(Slot(sid, int(r) + 1, x1, x2 - x1, y1, y2 - y1, sku_id=sid))
    return slots


def _hist(crop: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
    return h / (np.linalg.norm(h) + 1e-9)


def choose_roles(img: np.ndarray, slots: list[Slot], px_per_cm: float, rng: random.Random) -> dict[str, str]:
    """A (runs out), B (gets blocked), C/D (swapped: same row, similar size, most different look)."""
    by_row: dict[int, list[Slot]] = {}
    for s in slots:
        by_row.setdefault(s.shelf, []).append(s)
    rows = sorted(r for r, ss in by_row.items() if len(ss) >= 5)
    if len(rows) < 2:
        raise RuntimeError("Need at least two shelf rows with 5+ packs; pick another image.")
    mid = rows[len(rows) // 2]

    def px(s):
        return [int(round(v * px_per_cm)) for v in (s.x_cm, s.y_cm, s.x_cm + s.width_cm, s.y_cm + s.height_cm)]

    a = by_row[mid][len(by_row[mid]) // 2]
    b_row = rows[0] if rows[0] != mid else rows[-1]
    b = by_row[b_row][len(by_row[b_row]) // 3]
    best, best_d = None, -1.0
    for r in rows:
        ss = [s for s in by_row[r] if s.slot_id not in (a.slot_id, b.slot_id)]
        for i in range(len(ss)):
            for j in range(i + 2, len(ss)):  # not neighbours
                si, sj = ss[i], ss[j]
                if not (0.75 < si.width_cm / sj.width_cm < 1.33 and 0.75 < si.height_cm / sj.height_cm < 1.33):
                    continue
                ci, cj = (img[y1:y2, x1:x2] for x1, y1, x2, y2 in (px(si), px(sj)))
                if ci.size == 0 or cj.size == 0:
                    continue
                d = 1.0 - float(_hist(ci) @ _hist(cj))
                if d > best_d:
                    best, best_d = (si, sj), d
    if best is None:
        raise RuntimeError("No similar-size pair to swap; pick another image.")
    return dict(A=a.slot_id, B=b.slot_id, C=best[0].slot_id, D=best[1].slot_id)


@dataclass
class ScriptedShelf:
    base: np.ndarray
    slots: dict[str, Slot]
    roles: dict[str, str]
    px_per_cm: float
    seed: int = 0
    mask: np.ndarray | None = None
    truth: dict = field(default_factory=dict)

    def px(self, sid):
        s = self.slots[sid]
        return [int(round(v * self.px_per_cm)) for v in (s.x_cm, s.y_cm, s.x_cm + s.width_cm, s.y_cm + s.height_cm)]

    def _erase(self, img, box):
        x1, y1, x2, y2 = box
        # dark shelf-back with a top shadow (same look as notebook 02's synthetic gaps)
        base = np.percentile(self.base.reshape(-1, 3), 8, axis=0)
        h, w = y2 - y1, x2 - x1
        grad = np.linspace(0.6, 1.0, h)[:, None, None]
        img[y1:y2, x1:x2] = np.clip(base[None, None] * grad + 4, 0, 255).astype(np.uint8)

    def _person(self, img, box, rng):
        """A shopper silhouette covering `box` and some surroundings. Returns its mask."""
        H, W = img.shape[:2]
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2 + rng.randint(-10, 10)
        bw = int((x2 - x1) * 2.2) + 40
        top = max(0, y1 - (y2 - y1))
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (cx, max(0, top - 10)), (bw // 5, bw // 4), 0, 0, 360, 1, -1)       # head
        cv2.rectangle(m, (cx - bw // 2, top), (cx + bw // 2, min(H, y2 + (y2 - y1))), 1, -1)  # body
        colour = np.array([rng.randint(40, 120), rng.randint(40, 120), rng.randint(60, 160)], np.uint8)
        noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 12, img.shape)
        person = np.clip(colour[None, None] + noise, 0, 255).astype(np.uint8)
        img[m.astype(bool)] = person[m.astype(bool)]
        return m

    def frame(self, i: int) -> tuple[float, np.ndarray]:
        rng = random.Random(self.seed * 100_003 + i)
        t = (i + 1) * DT
        img = self.base.copy()
        A, B, C, D = (self.roles[k] for k in "ABCD")
        empty = EMPTY_T[0] <= t < EMPTY_T[1]
        if empty:
            self._erase(img, self.px(A))
        if t >= SWAP_T:
            bc, bd = self.px(C), self.px(D)
            pc = cv2.resize(self.base[bd[1]:bd[3], bd[0]:bd[2]], (bc[2] - bc[0], bc[3] - bc[1]))
            pd = cv2.resize(self.base[bc[1]:bc[3], bc[0]:bc[2]], (bd[2] - bd[0], bd[3] - bd[1]))
            img[bc[1]:bc[3], bc[0]:bc[2]] = pc
            img[bd[1]:bd[3], bd[0]:bd[2]] = pd
        mask = np.zeros(img.shape[:2], np.uint8)
        for role, t0, t1 in BLOCKS:
            if t0 <= t < t1:
                mask |= self._person(img, self.px(self.roles[role]), rng)
        if GLARE_T[0] <= t < GLARE_T[1]:
            H, W = img.shape[:2]
            g = np.zeros((H, W), np.float32)
            cv2.ellipse(g, (W // 2, H // 2), (int(W * 0.35), int(H * 0.3)), 0, 0, 360, 1.0, -1)
            g = cv2.GaussianBlur(g, (0, 0), 25)[..., None]
            img = np.clip(img * (1 - g) + 255 * g, 0, 255).astype(np.uint8)
        # camera noise: small brightness jitter and sensor noise
        img = np.clip(img.astype(np.float32) * rng.uniform(0.96, 1.04)
                      + np.random.default_rng(i).normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        # Oracle person mask that misses the shopper 5% of the time
        self.mask = mask if rng.random() >= 0.05 else np.zeros_like(mask)
        self.truth = dict(t=t, a_empty=empty, swapped=t >= SWAP_T, blocked=bool(mask.any()))
        return t, img


class OracleMasker:
    def __init__(self, scene: ScriptedShelf):
        self.scene = scene

    def mask(self, bgr):
        return self.scene.mask if self.scene.mask is not None else np.zeros(bgr.shape[:2], np.uint8)


def run_demo(base_img: np.ndarray, detector, embedder=None, thresholds: dict | None = None,
             px_per_cm: float = 10.0, seed: int = 0, n_frames: int = N_FRAMES, video_path: str | None = None,
             log=print, on_scene=None, min_conf: float = 0.5, layout_boxes=None) -> dict:
    """layout_boxes: optional pack boxes (base_img pixels) used as the approved slot layout, as a
    store's planogram would be. Without it the layout is drafted from detections on the base image.
    Either way, every frame is analysed by the real detector."""
    rng = random.Random(seed)
    if layout_boxes is not None and len(layout_boxes) >= 12:
        packs = [list(map(float, b)) for b in layout_boxes]
        log(f"{len(packs)} slots from the annotated layout")
    else:
        dets = detector(base_img)
        packs = [d["bbox_xyxy"] for d in dets if d["cls"] == "pack" and d["conf"] >= min_conf]
        log(f"{len(packs)} packs detected on the base image")
    if len(packs) < 12:
        raise RuntimeError("Too few packs detected; pick another image.")
    slots = draft_slots(packs, px_per_cm)
    roles = choose_roles(base_img, slots, px_per_cm, rng)
    log("roles:", roles)
    slot_by_id = {s.slot_id: s for s in slots}
    scene = ScriptedShelf(base_img, slot_by_id, roles, px_per_cm, seed)
    if on_scene is not None:
        on_scene(scene)

    infos = {s.slot_id: SlotInfo(s.slot_id, aisle="1", bay="B1", shelf=s.shelf, sku_id=s.sku_id,
                                 sku_name=f"pack {s.slot_id}", backroom_location="B-14") for s in slots}
    twin = Twin(infos)
    catalogue = None
    if embedder is not None:
        th = thresholds or {}
        crops, labels = [], []
        for s in slots:
            x1, y1, x2, y2 = scene.px(s.slot_id)
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                crops.append(base_img[y1:y2, x1:x2])
                labels.append(s.slot_id)
        emb = embedder(crops)
        catalogue = Catalogue(emb.shape[1], IdentityConfig(th.get("accept_sim", 0.75), th.get("accept_margin", 0.15)))
        for lab, e in zip(labels, emb):
            catalogue.add(lab, e[None])
    geo = BayGeometry.identity(base_img.shape[1], base_img.shape[0], px_per_cm)
    pipe = BayPipeline(twin, geo, slots, detector, OracleMasker(scene), embedder, catalogue,
                       PipelineConfig(bay_id="B1", identity_every_n=1))
    # one bay = the whole frame here, so a single erased pack is a small change: lower the gate
    pipe.change = ChangeGate(changed_frac=0.002, force_every_s=6)

    writer = None
    trace = {k: [] for k in ("t", "q", "a_cov", "a_vis", "a_score", "b_cov", "b_vis", "b_score",
                             "base_alerts", "eng_alerts")}
    events = []
    for i in range(n_frames):
        t, frame = scene.frame(i)
        out = pipe.process(frame, t, f"f{i}")
        for sid, d in out["decisions"].items():
            if d in (ACT, CLEAR):
                events.append(dict(t=t, slot=sid, decision=d, role=next((k for k, v in roles.items() if v == sid), "")))
        snap = {v.slot_id: v for v in twin.views.values()}
        a, b = snap[roles["A"]], snap[roles["B"]]
        trace["t"].append(t); trace["q"].append(out["q"])
        trace["a_cov"].append(a.coverage); trace["a_vis"].append(a.visibility); trace["a_score"].append(a.score)
        trace["b_cov"].append(b.coverage); trace["b_vis"].append(b.visibility); trace["b_score"].append(b.score)
        trace["base_alerts"].append(sum(v.baseline_alerts for v in twin.views.values()))
        trace["eng_alerts"].append(sum(v.engine_alerts for v in twin.views.values()))
        if video_path:
            vis = annotate(frame, twin, scene, t)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 8, (w, h))
            writer.write(vis)
    if writer is not None:
        writer.release()

    tasks = [t.to_dict() for t in twin.service.tasks.values()]
    eng_acts = [e for e in events if e["decision"] == ACT]
    true_acts = [e for e in eng_acts if e["role"] == "A" and EMPTY_T[0] <= e["t"] < EMPTY_T[1]]
    first_act = min((e["t"] for e in true_acts), default=None)
    summary = dict(
        roles=roles, n_slots=len(slots),
        single_frame_alerts=trace["base_alerts"][-1],
        engine_alerts=len(eng_acts), engine_correct_alerts=len(true_acts),
        engine_false_alerts=len(eng_acts) - len(true_acts),
        detection_delay_s=(first_act - EMPTY_T[0]) if first_act is not None else None,
        refill_verified=any(t["slot_id"] == roles["A"] and t["status"] == "verified" for t in tasks),
        displacement_task=any(t["kind"] in ("displacement", "cold_chain") and t["slot_id"] in (roles["C"], roles["D"])
                              for t in tasks),
        tasks=[dict(kind=t["kind"], slot=t["slot_id"], status=t["status"], created_s=t["created_ts"],
                    time_to_fix_s=t["time_to_fix_s"]) for t in tasks],
    )
    return dict(summary=summary, trace=trace, events=events, twin=twin, scene=scene, pipe=pipe)


STATE_COLOURS = {"HEALTHY": (60, 180, 75), "EMPTY": (40, 40, 230), "CANNOT_SEE": (150, 150, 150), "WAIT": (0, 190, 255)}


def annotate(frame, twin: Twin, scene: ScriptedShelf, t: float, width: int = 640) -> np.ndarray:
    img = frame.copy()
    snap = {s["slot_id"]: s for s in twin.snapshot()["slots"]}
    for sid, s in snap.items():
        x1, y1, x2, y2 = scene.px(sid)
        col = STATE_COLOURS.get(s["ui_state"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if s["ui_state"] != "HEALTHY" else 1)
    for role, sid in scene.roles.items():
        x1, y1, _, _ = scene.px(sid)
        cv2.putText(img, role, (x1 + 2, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * s)))
    tot = twin.snapshot()["totals"]
    cv2.rectangle(img, (0, 0), (width, 26), (0, 0, 0), -1)
    cv2.putText(img, f"t={t:5.0f}s  single-frame alerts={tot['baseline_alerts']}  ShelfTwin alerts={tot['engine_alerts']}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def plot_trace(res: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = res["trace"]
    t = np.array(tr["t"])
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, key, title in ((axs[0], "a", "Slot A: runs out at 40 s, shopper passes 60-72 s, refilled at 280 s"),
                           (axs[1], "b", "Slot B: stocked, blocked by shoppers (must NOT alert)")):
        cov = np.array([np.nan if v is None else v for v in tr[f"{key}_cov"]], float)
        vis = np.array([np.nan if v is None else v for v in tr[f"{key}_vis"]], float)
        ax.plot(t, cov, ".", ms=3, label="coverage (detector)")
        ax.plot(t, vis, "-", lw=0.8, alpha=0.6, label="visibility (mask)")
        ax2 = ax.twinx()
        ax2.plot(t, tr[f"{key}_score"], "k-", lw=1.4, label="evidence L")
        ax2.axhline(4.0, color="r", ls="--", lw=0.8)
        ax2.axhline(-1.0, color="g", ls="--", lw=0.8)
        ax2.set_ylabel("evidence")
        for e in res["events"]:
            if e["role"] == key.upper():
                ax2.axvline(e["t"], color="r" if e["decision"] == ACT else "g", lw=2, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left", fontsize=8)
    axs[0].axvspan(*EMPTY_T, color="r", alpha=0.07)
    for role, t0, t1 in BLOCKS:
        axs[0 if role == "A" else 1].axvspan(t0, t1, color="grey", alpha=0.2)
    axs[2].plot(t, tr["base_alerts"], label="single-frame baseline alerts")
    axs[2].plot(t, tr["eng_alerts"], label="ShelfTwin alerts", lw=2)
    axs[2].axvspan(*GLARE_T, color="y", alpha=0.3, label="glare")
    axs[2].set_xlabel("time (s)")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
