# Training runbook (free Kaggle or free Colab)

## ⭐ Recommended: `ShelfTwin_Kaggle_final.ipynb`, the whole project in one Kaggle notebook

EDA → engine analysis → **experiments with error analysis** (6 detector configs, 4 gap-label strategies,
zero-shot ViT-S vs ViT-B plus 4 embedder strategies) → winners picked automatically (`decision_*.json`) →
final training with the winners → full evaluation and error analysis → end-to-end demo video → `REPORT.md`.

| Session | Accelerator | Runs | GPU quota |
|---|---|---|---|
| 1 | None (CPU) | all data preparation (cached), EDA, engine simulation and error analysis | 0 h |
| 2 | GPU T4 x2 | experiments (one run per GPU in parallel) + error analysis | ~3-4 h |
| 3 | GPU T4 x2 | final detector, full SKU-110K, DDP | ~6-8 h |
| 4 | GPU T4 x2 | final pack+gap detector, final embedder, evaluation, demo, report | ~4-6 h |

Every session: attach `thedatasith/sku110k-annotations` and `khyeh0719/rp2k-dataset`; from session 2 on, also
attach this notebook's latest version (*Your Work → Notebooks*). Then *Save & Run All (Commit)*. Finished phases are
skipped and interrupted training resumes. The last cell says which accelerator to use next. Run once with
`SMOKE = True` first (~30 min on GPU).

The notebooks below are the same pieces split up (and a Colab variant).

All notebooks are **self-contained**, **resumable** and tuned for **speed**. Upload one `.ipynb` file and run it.
The code inside is built from tested sources (`notebooks/src/*.py`, `shelftwin/`) by `python notebooks/build_notebooks.py`.

## Option A: Google Colab, everything in one notebook

`ShelfTwin_Colab_full.ipynb` runs the whole pipeline: engine calibration → SKU-110K data → pack+gap detector →
DINOv2/ArcFace embedder → end-to-end pipeline demo on a real shelf photo (video + plots) → live dashboard in Colab.

1. Colab → *File → Upload notebook*. *Runtime → Change runtime type → T4 GPU*.
2. 🔑 *Secrets*: add `KAGGLE_USERNAME` and `KAGGLE_KEY` (kaggle.com → Settings → API → Create New Token) and allow notebook access.
3. *Runtime → Run all*. Allow Google Drive access when asked. Keep the tab open (idle tabs disconnect after ~90 min).
4. **Disconnected or session over? Runtime → Run all again.** Finished stages are skipped, data is restored from Drive,
   and training continues from the last epoch.

Per session it trains the detector for up to `DETECTOR_HOURS` (3 h) and the embedder for `EMBED_HOURS` (1 h), then
runs the demo with the best weights so far. Expect 2-3 sessions to finish everything. It uses about 4 GB of Google Drive.

## Option B: Kaggle, four notebooks (faster: two T4s, ~30 GPU h/week)

| # | Notebook | Accelerator | Attach as input | Est. GPU h | Main outputs |
|---|---|---|---|---|---|
| 01 | `01_sku110k_pretrain.ipynb` | GPU T4 x2 | SKU-110K (e.g. `thedatasith/sku110k-annotations`) | ~5-8 | `pack_detector_sku110k.pt` |
| 02 | `02_pack_gap_finetune.ipynb` | GPU T4 x2 | SKU-110K + **your 01 output** | ~2-4 | `pack_gap.pt` |
| 04 | `04_embedder_arcface.ipynb` | GPU T4 x2 | RP2K (e.g. `khyeh0719/rp2k-dataset`) | ~1.5-2.5 | `embedder_best.pt`, `identity_thresholds.json` |
| 07 | `07_engine_calibration.ipynb` | **None (CPU)** | nothing | 0 | `engine_calibration.json` |

The GPU-hour figures are estimates; the first epoch's log shows the real speed. 04 and 07 are independent of 01/02.

**Run:** import the notebook, set Accelerator/Internet/inputs, run once with `SMOKE_TEST = True` (10-15 min, must print
`ALL DONE`), then set it to `False` and *Save Version → Save & Run All (Commit)*.

**Resume:** if the output says `NOT FINISHED`, add *Add Input → Your Work → Notebooks →* **the same notebook** (latest
version) and commit again. The notebook recognises its own previous output (`st_session_<nb>.json`) and restores the
checkpoints, dataset cache and training state.

**Combine:** Kaggle-trained weights can go into `MyDrive/ShelfTwin/models/` for the Colab demo stage, or into `models/` locally.

## Resumability

| What | How it survives a session ending |
|---|---|
| Detector training | Fixed-epoch cosine schedule across sessions; stops **right after** an epoch checkpoint when the session budget is nearly used; next session `resume=True` from `last.pt`. A corrupted `last.pt` falls back to `best.pt`; if a resume keeps failing (e.g. GPU count changed), it restarts from the saved weights for the remaining epochs. |
| Embedder training | Model, ArcFace weights, optimizer, AMP scaler, LR step, history and best score saved atomically every epoch. |
| Prepared datasets | Built once, stored as single tar files (Kaggle output / Google Drive) and restored in seconds. |
| Engine grid search | Each finished configuration is appended to `engine_grid.jsonl`; only the missing ones run. |
| Stages (Colab) | `progress.json` on Drive; finished stages are skipped. |

All of this logic is tested locally with fake trainers and simulated disconnects (`tests/`, plus CPU smoke scripts).

## Speed

| Bottleneck | What the notebooks do |
|---|---|
| Decoding 3000 px SKU-110K JPEGs | One-time parallel resize on **all CPU cores** (896 px; reduced-resolution JPEG decode), cached |
| Pack+gap crop generation | Multiprocessing over all cores, cached |
| RP2K: 500k files to scan, full-size decode | Only the used subset is resized to 256 px in parallel (PIL draft mode) and cached with a manifest. Later sessions skip the scan |
| GPU | Both T4s on Kaggle (DDP for YOLO, DataParallel for the embedder), AMP everywhere, data-loader workers = all cores |
| Evaluation | Quick val check while unfinished; full test evaluation once finished |
| Engine calibration | Grid search across all CPU cores |
| Re-runs | Finished stages and finished training runs return immediately |

Knobs: `IMGSZ` (896 → 640 is ~2× faster with lower small-pack recall), `MODEL='yolo11n.pt'` (~2× faster),
`MAX_TRAIN_IMAGES`, `EPOCHS`.

## Robustness

| Risk | Mitigation |
|---|---|
| Dataset layout differs between uploads | Auto-discovery (CSV, YOLO or class-folder layout); prints the input tree on failure |
| SKU-110K not attached / no Kaggle token | Streams the official archive and resizes on the fly (never stores the 13.6 GB archive) |
| Multi-GPU DDP fragile in notebooks | Training runs as a subprocess; DDP failure without progress → one GPU |
| Hung process | Watchdog kills the process group after 25 min without output |
| Dense images (up to 700 boxes) run out of GPU memory | `ultralytics>=8.4.134` (per-image GPU retry), `max_det=1000`, YOLO11 rather than YOLO26 (300-box cap) |
| W&B or other logger prompts block a background run | All integrations disabled |
| Huge notebook output | Progress bars collapsed to their final state; full logs in `logs/` |
| Embedder OOM | P×K batch halves automatically |
| timm/HF API differences | Backbone loader tries `img_size=224`, then `dynamic_img_size`, then torch.hub, and checks the output shape |

## Honest limitations

* Gap labels come from geometry (real photos) plus synthetic erasing (train only), not hand labels.
* The end-to-end demo scripts events on a real photo with a drawn shopper (oracle person mask). It isn't real store video.
* SKU-110K and RP2K are for non-commercial research. RP2K is Chinese retail, so add Indian pack photos to the catalogue.
* Locally verified with unit tests, fake-trainer resume tests and CPU smoke tests; the first real cloud run is the
  real test, so run `SMOKE_TEST = True` first.
