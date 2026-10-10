# ShelfTwin

**Trusted shelf-availability alerts from noisy cameras.** E-Cell IIT Bombay x Everseen CV Hackathon 2026, Track 1.

ShelfTwin keeps a per-slot digital twin of the shelf. A temporal evidence engine decides for each slot whether to
**ACT**, **WAIT**, or report that it **CANNOT SEE**. Tasks are ranked, routed to staff, and closed only when the camera
verifies the fix. See `ShelfTwin-project-spec.md` for the full design and `ShelfTwin.pptx` for the deck.

**Proof of concept**
* Code (this repo): engine, simulator, vision pipeline, task service, dashboard, 56 tests.
* Training run on real data, public with full logs: [Kaggle notebook `purv2k2/shelftwin-training`](https://www.kaggle.com/code/purv2k2/shelftwin-training)
  (built from `notebooks/ShelfTwin_Kaggle_final.ipynb`).
* Results: [trained models](#trained-models-held-out-real-data) and [real-data findings](#real-data-findings-kaggle-session-1) below; all result files are in `results/kaggle/`.

```
frames → quality/change gate → person mask → pack+gap detector → shelf geometry → coverage per slot
       → identity (DINOv2+ArcFace, abstains) → planogram alignment → temporal engine → tasks → dashboard / Telegram
```

## Quick start (CPU, no models needed)

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                            # 56 tests
python sim/simulate.py 20                            # engine vs baselines, 160 camera-hours (~30 s)
python -m shelftwin.api.server --demo --speed 20     # live dashboard on a simulated store
# open http://127.0.0.1:8000   (--lang hi / gu for Hindi / Gujarati task messages)
```

In the dashboard, compare the **single-frame alert counter** with **ShelfTwin alerts**. Press *I am on it*, then *Done*
on a task: staff "refill" the slot and the camera verifies it, which closes the task.

## Simulation results (`sim/simulate.py 20`, 1,452 restock events, 160 camera-hours)

| Method | False alerts / camera-hour | Precision | Recall | Median delay | p95 delay |
|---|---|---|---|---|---|
| Single-frame | 965.86 | 5.2% | 97.5% | 0 s | 0 s |
| Single-frame + person mask | 376.43 | 12.3% | 99.6% | 0 s | 2 s |
| Majority vote 3 of 5 + mask | 11.17 | 45.5% | 99.9% | 4 s | 8 s |
| **ShelfTwin engine** | **0.82** | **91.7%** | **100%** | **12 s** | **18 s** |

These numbers test the decision logic on simulated detector outputs (spec 7.1 assumptions). Real-camera accuracy has
to be measured on recorded store footage.

## Trained models (held-out real data)

| Model | Result | Test data |
|---|---|---|
| Pack detector (YOLO11s @ 1024 px) | **mAP50 95.0%**, mAP50-95 60.2%, precision 91.6%, recall 88.9% | 2,920 unseen SKU-110K photos |
| Pack + gap detector | gap AP50 57.4% (recall at 90% precision 7.8%: heuristic gap labels are noisy) | real gaps on unseen photos |
| Product identity (DINOv2 ViT-S/14 + ArcFace) | **Recall@1 93.1%** (untrained DINOv2: 70.1%), mAP@R 0.82, 94.3% precision on accepted matches, 96.9% of new products answered "unknown" | the 20% of RP2K SKUs held out from training (1,549 SKUs used for training) |
| End-to-end pipeline demo | **8 false alerts vs 53** for single-frame; stockout caught in 32 s; refill camera-verified; swapped products flagged | real shelf photo, scripted 8-minute episode |

Experiment winners (subset runs, picked automatically): YOLO11s at 1024 px over 896 px, no-mosaic and YOLO11n;
fine-tuning the last 4 DINOv2 blocks over 2 blocks, head-only and zero-shot. Video: `results/kaggle/e2e_demo.mp4`.

## Real-data findings (Kaggle session 1)

From the CPU session of the Kaggle notebook (EDA on all 11,689 SKU-110K images and 381,303 RP2K images):

| Finding | Number | Decision it drives |
|---|---|---|
| Packs per shelf photo | mean 147, max **718** | `max_det` 1000; YOLO11 (NMS) not YOLO26 (300-box cap) |
| Packs smaller than 16 px at the training size | 640 px: **40.7%**, 896 px: 17.7%, 1024 px: 11.7% | 640 px dropped from the experiments; 1024 px tested |
| Photos with at least one real shelf gap | **88%**, 5.4 gaps per photo | enough real gap labels to evaluate on real photos |
| RP2K products | 2,233 SKUs, median 47 photos/SKU, 56% of photos < 112 px | 224 px input; open-set split by SKU |
| Engine false alerts by cause (simulation) | **all** come from shoppers the person mask missed | the person/cart mask is the lever, not the detector |

Engine tuned on held-out simulator seeds: **0.12 false alerts per camera-hour**, 98.6% precision, 100% recall,
18 s median and 26 s p95 delay (spec defaults: 0.82, 91.7%, 100%, 12 s, 18 s).

## Training (free Colab or free Kaggle)

See **[notebooks/README.md](notebooks/README.md)**. Every notebook is resumable, parallelised and speed-tuned.

* **Colab, one notebook:** `notebooks/ShelfTwin_Colab_full.ipynb` covers data, both models, the end-to-end demo and the dashboard. Everything is saved to Google Drive; after a disconnect, *Run all* again.
* **Kaggle, four notebooks** (two T4s, faster):

1. `01_sku110k_pretrain`: YOLO11s pack detector on SKU-110K
2. `02_pack_gap_finetune`: pack + gap detector (real geometric gaps, plus synthetic gaps for training only)
3. `04_embedder_arcface`: DINOv2 ViT-S/14 + ArcFace product embedder, open-set evaluation, abstention thresholds
4. `07_engine_calibration`: engine grid search on CPU (no GPU quota)

## Deck and results (reproducible)

`ShelfTwin.pptx` / `ShelfTwin.pdf` are generated from the original deck (`deck/ShelfTwin_base.pptx`) and the result
files in this repository, so every number on the proof-of-concept slides traces back to a file:

```bash
python scripts/fetch_kaggle_results.py   # small outputs of the public Kaggle run -> results/kaggle/
python scripts/update_deck.py            # rebuilds ShelfTwin.pptx (slides 1, 12-14, 20)
```

## Running on real video (after training)

```bash
pip install "ultralytics>=8.4.134" timm faiss-cpu
python scripts/run_video.py --source shelf.mp4 --detector models/pack_gap.pt --fps 1
python scripts/run_video.py --source rtsp://cam --detector models/pack_gap.pt --layout configs/planogram_demo.csv \
       --corners 102,80,1830,95,1815,1010,110,990 --size-cm 180,120
```

Without `--layout`, slots are drafted from the first frame.

## Layout

```
shelftwin/   engine/ (temporal engine, baselines)  gate/ (quality, change)  vision/ (detector, tiling, visibility,
             geometry, identity, ocr)  planogram/ (alignment, layouts)  tasks/ (priority, TTE, lifecycle, service)
             store/ (SQLite twin store)  notify/ (en/hi/gu messages, Telegram)  api/ (FastAPI + dashboard)
             twin.py  pipeline.py
sim/         simulate.py (benchmark)  live.py (live demo store)
notebooks/   Kaggle notebooks + src/ (tested helper code) + build_notebooks.py
scripts/     run_video.py
tests/       unit + system tests
```

## Status vs spec

| Area | Status |
|---|---|
| Temporal engine, baselines, simulator | Done, tested; simulation reproduces the spec table |
| Quality/change gate, tiling + box fusion, geometry, coverage, drift check | Done, tested |
| Identity catalogue (FAISS or numpy), abstention, OCR token tie-break | Done, tested; embedder trained in notebook 04 |
| Planogram alignment (Needleman-Wunsch), layouts, bay "layout changed?" guard | Done, tested |
| Task service: priority, batching, escalation, verification, reopen, cold chain | Done, tested |
| FastAPI + dashboard, Telegram, en/hi/gu messages | Done (dashboard is plain HTML, not Next.js) |
| Detector and embedder training | Done on Kaggle (experiments → decisions → final training → evaluation) |
| Visibility (cart/basket fine-tune), OCR fine-tune, shelf-edge model, Jetson export | Not yet: COCO person mask is used as is |
| WhatsApp, PostgreSQL, MQTT | Not yet: SQLite and in-process calls for the prototype |
