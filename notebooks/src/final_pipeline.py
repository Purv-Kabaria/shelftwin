"""ShelfTwin end-to-end Kaggle pipeline: EDA -> experiments -> decisions -> final training ->
evaluation and error analysis -> end-to-end demo. Runs as phases across several Kaggle sessions.

* CPU phases (EDA, engine analysis, ALL data preparation) run in a CPU-only session, which
  costs no GPU quota. Prepared data is cached as tars in the notebook output.
* GPU phases (experiments, final training, evaluation) need an accelerator; experiments run
  one per GPU in parallel (T4 x2), final detector training uses DDP.
* Every phase is idempotent and resumable: finished phases are skipped, unfinished training
  continues from its last epoch. Attach the previous version of the notebook as input.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import kaggle_common as kc
from kaggle_common import WORK, log

OUT = WORK / "outputs"


# ============================================================================ phase bookkeeping
def topic(cfg: dict, name: str) -> str:
    """Smoke-test runs use separate progress/decision files, so they can never leak into a real run."""
    return name + cfg.get("SFX", "")


class Progress:
    def __init__(self, sfx: str = ""):
        self.path = OUT / f"progress_final{sfx}.json"
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}

    def done(self, name: str) -> bool:
        return self.d.get(name, {}).get("status") == "done"

    def mark(self, name: str, status: str, **info):
        self.d[name] = dict(status=status, at=time.strftime("%Y-%m-%d %H:%M:%S"), **info)
        kc.save_json(self.d, self.path)

    def summary(self) -> dict:
        return {k: v.get("status") for k, v in self.d.items()}


def session_left_h(cfg: dict) -> float:
    return cfg["SESSION_HOURS"] - kc.elapsed_h() - cfg["RESERVE_H"]


def run_phase(prog: Progress, name: str, fn, cfg: dict, needs_gpu: bool, min_h: float = 0.1):
    """Run a phase unless done / impossible in this session. fn(cfg, budget_h) -> (done: bool, info)."""
    if prog.done(name):
        log(f"[{name}] already done")
        return True
    if needs_gpu and not kc.gpu_info():
        log(f"[{name}] needs a GPU: skipped in this CPU session")
        prog.mark(name, "waiting_for_gpu")
        return False
    left = session_left_h(cfg)
    if left < min_h:
        log(f"[{name}] not enough session time left ({left:.2f} h); continue next session")
        return False
    log(f"===== phase {name} (session time left {left:.2f} h) =====")
    t0 = time.time()
    try:
        ok, info = fn(cfg, left)
        prog.mark(name, "done" if ok else "partial", minutes=round((time.time() - t0) / 60, 1), **(info or {}))
        return ok
    except Exception as e:
        log(f"[{name}] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        prog.mark(name, "failed", error=f"{type(e).__name__}: {e}")
        return False


# ============================================================================ data (CPU)
def sku_exp(cfg) -> Path:
    import sku110k_prep as sp
    name = "sku110k_exp" + cfg["SFX"]

    def build(d: Path):
        # Cut from the prepared full set: no second decode of ~3000 px photos, identical preprocessing
        import os
        import random
        import shutil
        full = sku_full(cfg)
        sp._prepare_out(d)
        for split, n in cfg["EXP_SUBSET"].items():
            ims = sorted((full / "images" / split).glob("*.jpg"))
            random.Random(0).shuffle(ims)
            for im in ims[:n]:
                for src, dst in ((im, d / "images" / split / im.name),
                                 (full / "labels" / split / f"{im.stem}.txt", d / "labels" / split / f"{im.stem}.txt")):
                    if src.exists():
                        try:
                            os.link(src, dst)
                        except OSError:
                            shutil.copy2(src, dst)
        sp.write_data_yaml(d, {0: "pack"})
    return kc.cached_dir(name, kc.TMP / name, build)


def sku_full(cfg) -> Path:
    import sku110k_prep as sp
    name = "sku110k_full" + cfg["SFX"]
    return kc.cached_dir(name, kc.TMP / name, lambda d: sp.prepare_sku110k(
        out=d, max_side=cfg["SRC_SIDE"], limits=cfg["FULL_SUBSET"], allow_download=True))


def packgap(cfg, src: Path, tag: str, p_synth: float, fill, max_train: int | None) -> Path:
    import gap_synth as gs
    name = f"pg_{tag}" + cfg["SFX"]
    lim = dict(train=max_train) if max_train else None
    return kc.cached_dir(name, kc.TMP / name, lambda d: gs.build_pack_gap(
        src, d, limits=lim, n_crops=2, out_size=640, p_synth=p_synth, fill_probs=tuple(fill)))


def rp2k_subset(cfg, which: str) -> Path | None:
    """Builds (or restores) the pre-resized RP2K subset used by the embedder phases."""
    import embed_train as et
    e = embed_cfg(cfg, which)
    try:
        et.load_or_build_splits(e, kc.TMP)
        return kc.TMP / et.subset_key(e)
    except FileNotFoundError as err:
        log("RP2K not attached:", err)
        return None


def phase_data(cfg, budget_h):
    t0 = time.time()
    full = sku_full(cfg)
    exp = sku_exp(cfg)
    for g in cfg["GAP_RUNS"]:
        packgap(cfg, exp, "exp_" + g["name"], g["p_synth"], g["fill"], cfg["GAP_EXP_TRAIN"])
    dg = cfg["DEFAULT_GAP"]
    packgap(cfg, full, "full_default", dg["p_synth"], dg["fill"], cfg["GAP_FULL_TRAIN"])
    r_exp, r_full = rp2k_subset(cfg, "exp"), rp2k_subset(cfg, "final")
    import sku110k_prep as sp
    return True, dict(sku_exp=sp.count_split(exp), sku_full=sp.count_split(full), rp2k=bool(r_exp and r_full),
                      prep_minutes=round((time.time() - t0) / 60, 1))


# ============================================================================ EDA + engine (CPU)
def phase_eda(cfg, budget_h):
    import eda
    import embed_train as et
    import sku110k_prep as sp
    info = {}
    src = sp.find_csv_source()
    if src is not None:
        rep = eda.sku_eda(src[0], OUT / "eda", gap_sample=1500 if not cfg["SMOKE"] else 50)
        try:
            eda.show_sku_samples(src[0], OUT / "eda", n=4)
        except Exception as e:
            log("sample plot skipped:", e)
        info["sku"] = rep["recommendations"]
    else:
        log("No SKU-110K annotation CSVs (YOLO-format copy attached): EDA runs on the prepared YOLO dataset")
        full = sku_full(cfg)
        rep = eda.sku_eda_yolo(full, OUT / "eda", gap_sample=1500 if not cfg["SMOKE"] else 50)
        try:
            eda.show_sku_samples_yolo(full, OUT / "eda", n=4)
        except Exception as e:
            log("sample plot skipped:", e)
        info["sku"] = rep["recommendations"]
    try:
        items = et.discover_dataset()
        info["rp2k"] = eda.rp2k_eda(items, OUT / "eda")["recommendations"]
    except FileNotFoundError:
        log("RP2K not attached: RP2K EDA skipped")
    return True, info


def phase_engine(cfg, budget_h):
    import calibrate_engine as ce
    from sim.simulate import SimConfig, print_table, run
    from shelftwin.engine import EngineConfig
    smoke = cfg["SMOKE"]
    sim = SimConfig(hours=1.0) if smoke else SimConfig()
    seeds = (0,) if smoke else (0, 1, 2, 3)
    sens = ce.sensitivity(SimConfig(hours=1.0 if smoke else 4.0), seeds=seeds,
                          sweeps=dict(p_mask_miss=[0.0, 0.3]) if smoke else None)
    att = ce.attribute_false_alerts(sim_cfg=sim, seeds=seeds)
    grid = dict(llr_empty=[1.6], llr_stocked=[2.2], act_threshold=[3.5, 4.0], decay_per_s=[0.02]) if smoke else ce.GRID
    best, _ = ce.grid_search(sim, grid, seeds=ce.TUNE_SEEDS[:1] if smoke else ce.TUNE_SEEDS,
                             cache_path=OUT / ("engine_grid_smoke.jsonl" if smoke else "engine_grid.jsonl"))
    n = 2 if smoke else 20
    base = run(n, sim, verbose=False)
    tuned = run(n, sim, EngineConfig(**best["params"]), only=["ShelfTwin engine"], verbose=False)
    print_table({**base, "ShelfTwin (tuned)": tuned["ShelfTwin engine"]})
    rep = dict(baselines=base, tuned=tuned, best_params=best["params"], sensitivity=sens, attribution=att)
    kc.save_json(rep, OUT / "engine_analysis.json")
    _plot_engine(rep, OUT / "engine_analysis.png")
    kc.save_decision(topic(cfg, "engine"), best["params"], dict(tuned=tuned["ShelfTwin engine"]))
    return True, dict(best_params=best["params"], false_alert_causes=att["cause_counts"])


def _plot_engine(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sens = rep["sensitivity"]
    params = list(dict.fromkeys(r["param"] for r in sens))
    fig, axs = plt.subplots(1, len(params) + 1, figsize=(4.2 * (len(params) + 1), 3.6))
    for ax, p in zip(axs, params):
        rows = [r for r in sens if r["param"] == p]
        x = [r["value"] for r in rows]
        for k, st in (("ShelfTwin engine", "o-"), ("Majority vote 3 of 5 + mask", "s--")):
            ax.plot(x, [max(1e-3, r[k]["false_alerts_per_cam_hour"]) for r in rows], st, label=k.split(" +")[0])
        ax.set_yscale("log"); ax.set_title(p, fontsize=9); ax.set_ylabel("false alerts / cam-h")
    axs[0].legend(fontsize=7)
    cc = rep["attribution"]["cause_counts"] or {"none": 0}
    axs[-1].barh(list(cc), list(cc.values())); axs[-1].set_title("causes of engine false alerts", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# ============================================================================ experiments (GPU)
def _det_common(cfg, data_yaml):
    return dict(data=str(data_yaml), epochs=cfg["DET_EXP_EPOCHS"], optimizer="AdamW", lr0=0.002, cos_lr=True,
                warmup_epochs=1, close_mosaic=min(3, cfg["DET_EXP_EPOCHS"] - 1), fliplr=0.5, degrees=0.0, max_det=1000,
                amp=True, patience=100, seed=0, deterministic=False, plots=False)


def phase_exp_detector(cfg, budget_h):
    import ablation as ab
    import det_analysis as da
    import yolo_runner as yr
    d = sku_exp(cfg)
    yaml = d / "data.yaml"
    runs = [dict(_det_common(cfg, yaml), **r) for r in cfg["DET_RUNS"]]
    ab.run_parallel(runs, budget_h=budget_h)
    rows = ab.evaluate_runs(runs, yaml, "test", OUT / "exp_detector.jsonl")
    if len(rows) < len(runs):
        return False, dict(evaluated=len(rows), total=len(runs))
    print(ab.table(rows, ["name", "model", "imgsz", "mosaic", "mAP50", "mAP50_95", "recall", "latency_ms_t4_fp16"]))
    win = ab.pick(rows, "mAP50_95", tol=cfg["SPEED_TOL"])
    w = yr.run_dir(win["name"]) / "weights" / "best.pt"
    recs = da.collect(w, d / "images" / "test", d / "labels" / "test", imgsz=win["imgsz"], n=300)
    summ = da.summarize(recs)
    da.worst_gallery(recs, OUT / "exp_detector_worst.png")
    sweep = da.sweep_val(w, yaml, win["imgsz"], split="test", ious=(0.5, 0.6, 0.7), max_dets=(300, 1000))
    tiled = da.tiled_vs_full(w, d / "images" / "test", d / "labels" / "test", imgsz=640, n=150)
    best_iou = max([r for r in sweep if r["max_det"] == 1000], key=lambda r: r["mAP50_95"])["nms_iou"]
    rep = dict(runs=rows, winner=win, error_analysis=summ, eval_sweep=sweep, tiled_vs_full_at_640=tiled)
    da.save(rep, OUT / "exp_detector.json")
    kc.save_decision(topic(cfg, "detector"), dict(model=win["model"], imgsz=win["imgsz"], mosaic=win["mosaic"], nms_iou=best_iou),
                     dict(table=rows))
    return True, dict(winner=win["name"], mAP50_95=win["mAP50_95"])


def phase_exp_gap(cfg, budget_h):
    import ablation as ab
    import gap_synth as gs
    import yolo_runner as yr
    exp = sku_exp(cfg)
    runs, data = [], {}
    for g in cfg["GAP_RUNS"]:
        pg = packgap(cfg, exp, "exp_" + g["name"], g["p_synth"], g["fill"], cfg["GAP_EXP_TRAIN"])
        data[g["name"]] = pg
        runs.append(dict(name="gap_" + g["name"], model="yolo11s.pt", data=str(pg / "data.yaml"), imgsz=640,
                         epochs=cfg["GAP_EXP_EPOCHS"], batch=16, batch_per_gpu=16, optimizer="AdamW", lr0=0.002,
                         cos_lr=True, warmup_epochs=1, mosaic=0.5, close_mosaic=min(3, cfg["GAP_EXP_EPOCHS"] - 1),
                         fliplr=0.5, degrees=0.0, hsv_v=0.5, max_det=1000, amp=True, patience=100, seed=0,
                         deterministic=False, plots=False))
    ab.run_parallel(runs, budget_h=budget_h)
    rows = []
    for g, r in zip(cfg["GAP_RUNS"], runs):
        last = yr.run_dir(r["name"]) / "weights" / "last.pt"
        if not (last.exists() and yr.ckpt_epoch(last) == -1):
            return False, dict(unfinished=r["name"])
        best = yr.run_dir(r["name"]) / "weights" / "best.pt"
        m = yr.evaluate(best, data[g["name"]] / "data.yaml", imgsz=640, split="test")
        rec = gs.gap_recall_at_precision(best, data[g["name"]], split="test", target_p=0.9, imgsz=640,
                                         max_images=500 if not cfg["SMOKE"] else 50)
        ap = m.get("per_class_AP50", {})
        rows.append(dict(name=g["name"], p_synth=g["p_synth"], fill=list(g["fill"]), gap_AP50=ap.get("gap"),
                         pack_AP50=ap.get("pack"), gap_recall_at_p90=rec.get("recall_at_p"), n_gap_gt=rec.get("n_gt")))
    print(ab.table(rows, ["name", "p_synth", "gap_AP50", "gap_recall_at_p90", "pack_AP50"]))
    win = max(rows, key=lambda r: (r["gap_recall_at_p90"] or 0, r["gap_AP50"] or 0))
    kc.save_json(dict(runs=rows, winner=win), OUT / "exp_gap.json")
    kc.save_decision(topic(cfg, "gap"), dict(p_synth=win["p_synth"], fill=win["fill"]), dict(table=rows))
    return True, dict(winner=win["name"])


def embed_cfg(cfg, which: str, **over) -> dict:
    base = dict(cfg["EMBED_BASE"])
    base.update(cfg["EMBED_EXP"] if which == "exp" else cfg["EMBED_FINAL"])
    base.update(over)
    return base


def phase_exp_embed(cfg, budget_h):
    import embed_train as et
    if rp2k_subset(cfg, "exp") is None:
        return True, dict(skipped="RP2K not attached")
    base = embed_cfg(cfg, "exp")
    zs_path = OUT / "exp_embed_zero_shot.json"
    if zs_path.exists():
        zs = json.loads(zs_path.read_text())
    else:
        zs = [et.zero_shot_eval(n, base) for n in cfg["EMBED_ZERO_SHOT"]]
        kc.save_json(zs, zs_path)
    rows = []
    deadline = time.time() + budget_h * 3600
    for v in cfg["EMBED_RUNS"]:
        left = (deadline - time.time()) / 3600
        if left < 0.05:
            return False, dict(stopped_at=v["name"])
        c = embed_cfg(cfg, "exp", **{k: x for k, x in v.items() if k != "name"},
                      out_dir=str(WORK / "exp_embed" / v["name"]), zero_shot_baseline=False, time_budget_h=left)
        rep = et.train(c)
        if not rep.get("training_complete"):
            return False, dict(unfinished=v["name"])
        rows.append(dict(name=v["name"], **{k: v[k] for k in v if k != "name"}, **rep["test"],
                         train_minutes=round(rep["train_minutes"], 1)))
    for r in rows:
        log({k: r.get(k) for k in ("name", "recall_at_1", "map_at_r", "known_coverage", "unknown_rejection", "train_minutes")})
    win = max(rows, key=lambda r: (round(r["map_at_r"], 3), r["recall_at_1"]))
    err = et.analyze_errors(base, WORK / "exp_embed" / win["name"] / "embedder_best.pt", OUT / "exp_embed_errors")
    # "No fine-tuning" is a candidate too: the deployed backbone (DINOv2 ViT-S) used as is
    zs_s = next((z for z in zs if "vit_small" in z.get("model", "") and "map_at_r" in z), None)
    zero_shot_wins = bool(zs_s and zs_s["map_at_r"] > win["map_at_r"] + 0.005)
    kc.save_json(dict(zero_shot=zs, runs=rows, winner=win, errors=err, zero_shot_wins=zero_shot_wins),
                 OUT / "exp_embed.json")
    if zero_shot_wins:
        log(f"Zero-shot ViT-S (mAP@R {zs_s['map_at_r']:.4f}) beats the best fine-tuned run "
            f"({win['name']}, {win['map_at_r']:.4f}): final phase will only calibrate thresholds")
        dec = dict(zero_shot=True)
    else:
        dec = {k: win[k] for k in ("unfreeze_blocks", "arc_margin", "arc_scale", "lr_head") if k in win}
    kc.save_decision(topic(cfg, "embedder"), dec, dict(table=rows, zero_shot=zs))
    for v in cfg["EMBED_RUNS"]:  # phase done: resume states are no longer needed (~110 MB each)
        (WORK / "exp_embed" / v["name"] / "embedder_resume.pt").unlink(missing_ok=True)
    return True, dict(winner="zero_shot" if zero_shot_wins else win["name"], recall_at_1=win["recall_at_1"])


# ============================================================================ final training (GPU)
def _time_boxed(args: dict, hours: float | None) -> dict:
    """Ultralytics `time=` re-plans the epoch count and LR schedule to finish in `hours`, so a run
    under a deadline ends properly annealed instead of being cut off mid-schedule."""
    if hours:
        args = dict(args, time=float(hours))
        log(f"time-boxed training: {hours} h (epochs are re-planned by Ultralytics to fit)")
    return args


def phase_final_detector(cfg, budget_h):
    import yolo_runner as yr
    dec = kc.load_decision(topic(cfg, "detector"))
    full = sku_full(cfg)
    imgsz = dec.get("imgsz", 1024)   # fallback from the EDA: 11.7% of packs < 16 px at 1024 vs 40.7% at 640
    a = dict(model=dec.get("model", "yolo11s.pt"), data=str(full / "data.yaml"), imgsz=imgsz,
             epochs=cfg["DET_FINAL_EPOCHS"], batch=16 if imgsz <= 896 else 12, batch_per_gpu=8 if imgsz <= 896 else 6,
             name="final_pack" + cfg["SFX"], optimizer="AdamW", lr0=0.002, cos_lr=True, warmup_epochs=2,
             mosaic=dec.get("mosaic", 1.0), close_mosaic=min(10, max(1, cfg["DET_FINAL_EPOCHS"] // 4)), fliplr=0.5,
             degrees=0.0, perspective=0.0005, max_det=1000, amp=True, patience=15, seed=0, deterministic=False)
    r = yr.train(_time_boxed(a, cfg.get("DET_FINAL_TIME_H")), budget_h=min(budget_h, cfg["DET_FINAL_MAX_H"]))
    if not r["finished"]:
        return False, dict(epochs_done=r["epochs_done"])
    kc.save_json(dict(args=a, decision=dec), OUT / "final_pack_args.json")
    return True, dict(epochs=r["epochs_done"], weights=str(r["best"]))


def phase_final_packgap(cfg, budget_h):
    import yolo_runner as yr
    dec_g = kc.load_decision(topic(cfg, "gap")) or cfg["DEFAULT_GAP"]
    dg = cfg["DEFAULT_GAP"]
    same = abs(dec_g.get("p_synth", dg["p_synth"]) - dg["p_synth"]) < 1e-9 and \
        list(dec_g.get("fill", dg["fill"])) == list(dg["fill"])
    full = sku_full(cfg)
    tag = "full_default" if same else f"full_p{dec_g['p_synth']}_{'_'.join(str(x) for x in dec_g['fill'])}"
    pg = packgap(cfg, full, tag, dec_g.get("p_synth", dg["p_synth"]), dec_g.get("fill", dg["fill"]), cfg["GAP_FULL_TRAIN"])
    init = yr.run_dir("final_pack" + cfg["SFX"]) / "weights" / "best.pt"
    a = dict(model=str(init) if init.exists() else "yolo11s.pt", data=str(pg / "data.yaml"), imgsz=640,
             epochs=cfg["PG_FINAL_EPOCHS"], batch=32, batch_per_gpu=16, name="final_packgap" + cfg["SFX"],
             optimizer="AdamW", lr0=0.001, cos_lr=True, warmup_epochs=1, mosaic=0.5,
             close_mosaic=min(5, max(1, cfg["PG_FINAL_EPOCHS"] // 4)), fliplr=0.5, degrees=0.0, hsv_v=0.5,
             max_det=1000, amp=True, patience=12, seed=0, deterministic=False)
    r = yr.train(_time_boxed(a, cfg.get("PG_FINAL_TIME_H")), budget_h=min(budget_h, cfg["PG_FINAL_MAX_H"]))
    if not r["finished"]:
        return False, dict(epochs_done=r["epochs_done"])
    return True, dict(epochs=r["epochs_done"], data=str(pg))


def phase_final_embed(cfg, budget_h):
    import embed_train as et
    if rp2k_subset(cfg, "final") is None:
        return True, dict(skipped="RP2K not attached")
    dec = kc.load_decision(topic(cfg, "embedder"))
    c = embed_cfg(cfg, "final", **{k: v for k, v in dec.items() if k != "zero_shot"}, out_dir=str(OUT / "embedder"),
                  time_budget_h=min(budget_h, cfg["EMB_FINAL_MAX_H"]))
    if dec.get("zero_shot"):
        rep = et.calibrate_zero_shot(c, OUT / "embedder")
        return True, dict(mode="zero_shot", test=rep["test"])
    rep = et.train(c)
    if not rep.get("training_complete"):
        return False, dict(epochs=len(rep.get("history", [])))
    et.export_onnx(OUT / "embedder")
    et.analyze_errors(c, OUT / "embedder" / "embedder_best.pt", OUT / "final_embed_errors")
    return True, dict(test=rep["test"], zero_shot=rep.get("zero_shot_test"))


# ============================================================================ final evaluation + demo (GPU)
def phase_final_eval(cfg, budget_h):
    import det_analysis as da
    import gap_synth as gs
    import yolo_runner as yr
    full = sku_full(cfg)
    dec = kc.load_decision(topic(cfg, "detector"))
    nms = dec.get("nms_iou", 0.6)
    pack_w = yr.run_dir("final_pack" + cfg["SFX"]) / "weights" / "best.pt"
    pg_w = yr.run_dir("final_packgap" + cfg["SFX"]) / "weights" / "best.pt"
    info = {}
    if pack_w.exists():
        m = yr.evaluate(pack_w, full / "data.yaml", imgsz=dec.get("imgsz", 896), split="test", iou=nms)
        recs = da.collect(pack_w, full / "images" / "test", full / "labels" / "test", imgsz=dec.get("imgsz", 896),
                          iou=nms, n=None if not cfg["SMOKE"] else 10)
        info["pack_detector"] = dict(metrics=m, error_analysis=da.summarize(recs))
        da.worst_gallery(recs, OUT / "final_detector_worst.png")
        yr.export_onnx(pack_w, dec.get("imgsz", 896))
        yr.package_outputs({"pack_detector_sku110k.pt": pack_w})
    pg_info = json.loads(json.dumps(Progress(cfg["SFX"]).d.get("final_packgap", {})))
    if pg_w.exists() and pg_info.get("data"):
        pg = Path(pg_info["data"])
        m = yr.evaluate(pg_w, pg / "data.yaml", imgsz=640, split="test", iou=nms)
        rec = gs.gap_recall_at_precision(pg_w, pg, split="test", target_p=0.9, imgsz=640,
                                         max_images=2000 if not cfg["SMOKE"] else 20)
        info["pack_gap_detector"] = dict(metrics=m, gap_recall_at_p90=rec)
        yr.export_onnx(pg_w, 640)
        yr.package_outputs({"pack_gap.pt": pg_w})
    # end-to-end demo with every final model
    if pg_w.exists():
        info["e2e_demo"] = run_demo(cfg, pg_w, full)
    kc.save_json(info, OUT / "final_report.json")
    write_report(cfg, info)
    return True, dict(parts=list(info))


def run_demo(cfg, det_w: Path, full: Path) -> dict:
    import subprocess
    import cv2
    import e2e_demo as ed
    from shelftwin.vision.detector import DetectorConfig, PackGapDetector
    from shelftwin.vision.identity import Embedder
    det = PackGapDetector(DetectorConfig(weights=str(det_w), imgsz=640, device=0, half=True, conf=0.25))
    ck = OUT / "embedder" / "embedder_best.pt"
    emb = Embedder(str(ck) if ck.exists() else None)
    th_p = OUT / "embedder" / "identity_thresholds.json"
    th = json.loads(th_p.read_text()) if th_p.exists() else None
    lbl = full / "labels" / "test"
    cands = sorted(lbl.glob("*.txt"), key=lambda p: -len(p.read_text().splitlines()))
    cands = [p for p in cands if 60 <= len(p.read_text().splitlines()) <= 260][:8] or cands[:8]
    import det_analysis as da
    for p in cands:
        img = cv2.imread(str(full / "images" / "test" / (p.stem + ".jpg")))
        if img is None:
            continue
        # The image's annotated packs act as the store's approved layout (slots); every frame of the
        # episode is still analysed by the trained detector, mask, identity and engine.
        layout = ed.crop_boxes(da.read_labels(p, img.shape[1], img.shape[0], 0), img.shape)
        try:
            res = ed.run_demo(ed.square_crop(img), det, embedder=emb, thresholds=th, seed=0, layout_boxes=layout,
                              video_path=str(WORK / "demo_raw.mp4"), n_frames=60 if cfg["SMOKE"] else ed.N_FRAMES)
        except RuntimeError as e:
            log("demo: skip image", p.stem, e)
            continue
        ed.plot_trace(res, str(OUT / "e2e_trace.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(WORK / "demo_raw.mp4"), "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "28", str(OUT / "e2e_demo.mp4")])
        s = res["summary"]
        kc.save_json(s, OUT / "e2e_summary.json")
        return {k: v for k, v in s.items() if k != "tasks"}
    return dict(error="no suitable demo image")


def write_report(cfg, info: dict):
    """One markdown page summarising decisions and results (also displayed in the notebook)."""
    L = ["# ShelfTwin training report", ""]
    for name in ("detector", "gap", "embedder", "engine"):
        d = kc.load_decision(topic(cfg, name))
        if d:
            L.append(f"* **{name} decision**: `{json.dumps(d)}`")
    for f, title in (("exp_detector.json", "Detector experiments"), ("exp_gap.json", "Gap-label experiments"),
                     ("exp_embed.json", "Embedder experiments")):
        p = OUT / f
        if p.exists():
            rows = json.loads(p.read_text()).get("runs", [])
            if rows:
                keys = [k for k in rows[0] if not isinstance(rows[0][k], (list, dict))][:9]
                L += ["", f"## {title}", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
                L += ["| " + " | ".join(f"{r.get(k):.4f}" if isinstance(r.get(k), float) else str(r.get(k)) for k in keys) + " |"
                      for r in rows]
    L += ["", "## Final results", "", "```json", json.dumps(info, indent=1, default=str)[:6000], "```"]
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
