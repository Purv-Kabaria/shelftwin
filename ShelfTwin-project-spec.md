# ShelfTwin: Project Specification for Building and Training the Prototype

Version 1.0, October 2026. This document collects everything known about the ShelfTwin project: the problem, requirements, design, mathematics, datasets, training recipes for Kaggle and AWS, evaluation, deployment, costs and references. Use it as the single source of truth when generating code, notebooks and the final build.

Status labels used below:

- **Built**: code exists and runs (temporal engine, baselines, simulator, figures).
- **Specified**: design and parameters are fixed; code still to be written.
- **Open**: needs a decision or measurement.

---

## 1. Context

| Item | Value |
|---|---|
| Event | E-Cell IIT Bombay x Everseen, Computer Vision AI Hackathon 2026 |
| Track | Track 1: Shelf Auditing and Real-Time Stockouts |
| Team | Team ShelfTwin, SVNIT Surat |
| Submission | Idea plus prototype (deck `ShelfTwin.pptx`), repo `github.com/<team>/shelftwin` |
| Data from organisers | None assumed. All data comes from public datasets, our own capture and rendering. |

### 1.1 Problem statement (verbatim)

> "Design and build an automated computer vision solution that processes real-time imagery from shelf-scanning cameras, compares it against a digital planogram, and automatically triggers immediate re-stock or displacement alerts the moment inventory thresholds drop or items are misplaced."

Organiser context: human-led shelf audits are slow and error-prone and cannot catch stockouts or misplaced items in real time. Smart camera networks or roaming robots scan aisles continuously, and the scan is compared with the store planogram to raise alerts for backroom staff.

### 1.2 Requirements

| # | Requirement | Meaning in practice | How ShelfTwin meets it |
|---|---|---|---|
| R1 | Real-time imagery | Read shelf cameras, CCTV or phone video continuously | Edge pipeline at 0.5-2 frames/s per camera; change-triggered analysis |
| R2 | Planogram comparison | Know which product belongs in each slot | Approved layout; per-row sequence alignment of detections to slots |
| R3 | Threshold alerts | Warn when a slot runs low, not only when empty | Per-slot coverage states; adaptive low-stock threshold; time-to-empty |
| R4 | Displacement alerts | Detect a product in the wrong slot | Catalogue retrieval with abstention; identity kept separate from location |
| R5 | Immediate staff alerts | The right person gets a clear task fast | Ranked tasks with photos, stock location, reason, camera verification |
| R6 | Trust (our addition) | False alarms teach staff to ignore alerts | Temporal evidence engine with an explicit CANNOT SEE state |

### 1.3 Facts that motivate the design

| Figure | Meaning | Source |
|---|---|---|
| 8.3% | Average out-of-stock rate across 661 outlets worldwide | Gruen, Corsten, Bharadwaj 2002 [1] |
| Above 10% | Out-of-stock rate for fast-moving and promoted items | [1] |
| 70-75% | Out-of-stocks caused by store practice (ordering, forecasting, shelving) | [1] |
| About 25% | Product is in the store but not on the shelf | [1] |
| 47% / 25% / 28% | Root causes: store ordering and forecasting / store shelving and replenishment / upstream | [1] |
| 31% / 26% / 19% / 15% / 9% | Shopper response: buy elsewhere / switch brand / same brand other size / delay / do not buy | [1] |
| Up to 4% | Yearly revenue lost by grocery retailers to out-of-stocks | Corsten and Gruen 2003 [2] |

### 1.4 Pain points and responses

| # | Pain point | Affected | ShelfTwin response | Req. |
|---|---|---|---|---|
| 1 | Gaps found hours late | Shoppers, store | Continuous, change-triggered analysis | R1 |
| 2 | Only "empty" is detected | Floor staff | Low-stock state, adaptive threshold, time-to-empty | R3 |
| 3 | False alarms from shoppers, staff, glare | Floor staff | Visibility-weighted evidence; CANNOT SEE state | R6 |
| 4 | No maintained digital planogram | Most Indian stores | Layout drafted from shelf tags and stable views; manager approval | R2 |
| 5 | New, seasonal and look-alike packs | Recognition model | Catalogue retrieval with abstention; add a product with 3-5 photos | R4 |
| 6 | Stock in backroom, not on shelf | Backroom staff | Backroom rack camera; pick location in every task | R5 |
| 7 | Misplaced and abandoned items | Shoppers, cold chain | Slot alignment; displacement and cold-chain alerts | R4 |
| 8 | No proof the shelf was fixed | Managers | Task closes only on camera or photo verification | R5 |
| 9 | Costly hardware, weak internet | Small chains | Reuse CCTV and phones; all inference offline in store | R1 |
| 10 | No record of availability over time | Category managers | Per-slot history: time empty, time to fix, repeat offenders | R2 |

Indian retail specifics: few stores keep a maintained digital planogram; frequent pack redesigns, regional variants and look-alike multipacks; patchy internet; tight hardware budgets; high staff turnover.

---

## 2. Solution overview

ShelfTwin keeps a per-slot digital record (the "twin") built from cameras. A temporal engine decides for each slot whether to ACT, WAIT, or report that it CANNOT SEE. Tasks are ranked, routed to staff, and closed only when the camera verifies the fix.

Core idea: detection quality is no longer the bottleneck. Turning noisy per-frame detections into trusted actions is.

### 2.1 Pipeline

```
camera frames
  -> [1] quality + change gate          (drop blurred/glared frames; only process bays that changed)
  -> [2] person/cart visibility mask    (visibility v in [0,1] per slot)
  -> [3] pack + gap detector            (YOLO, class-agnostic "pack" and "gap"; 2 tiles of 640 px per 1080p frame)
  -> [4] shelf geometry                 (shelf-edge lines, homography to shelf cm, slot coverage c_s)
  -> [5] product identity               (DINOv2 ViT-S/14 + ArcFace head, FAISS k-NN, margin test, OCR tie-break)
  -> [6] planogram alignment            (Needleman-Wunsch with position cost; observed / draft / approved layouts)
  -> [7] temporal state engine          (visibility-weighted log-odds evidence with decay and hysteresis)
  -> [8] task service                   (priority, routing, batching, verification, escalation)
  -> [9] interfaces                     (dashboard, staff web app, Telegram, WhatsApp; English, Hindi, Gujarati)
```

### 2.2 The twin (per-slot record)

For every slot store:

- location (store, aisle, bay, shelf, x position, width in cm);
- expected product (from the approved layout version);
- last trusted observation (time, frame id, crop);
- state: HEALTHY, LOW, EMPTY, MISPLACED, CANNOT SEE;
- separate confidences: detection, identity, visibility, event;
- full history of state changes linked to the frames that caused them, so any alert can be replayed.

### 2.3 Design rules

1. One confident frame never triggers a task on its own.
2. Hidden slots are never marked empty; their evidence is down-weighted.
3. Identity is kept separate from location. The expected product of a slot never boosts the match score, so a misplaced pack is not "corrected" into the expected product.
4. The model abstains ("unknown") rather than guessing.
5. Observed, draft and approved layouts stay separate, so a messy shelf is never learned as correct.
6. Many simultaneous mismatches in one bay trigger one "layout changed?" question, not dozens of alerts.
7. Every decision is logged with its inputs.

---

## 3. Module specifications and I/O contracts

All modules run on the edge box unless noted. Use Python 3.10+ for the prototype; export models to ONNX and TensorRT for deployment.

### 3.1 Frame capture

- Input: RTSP (CCTV, shelf cameras) or USB; phone video files for walk-throughs.
- Decode: GStreamer or PyAV with hardware decode on Jetson (NVDEC).
- Rate: 0.5-2 frames/s per camera.
- Output: `Frame{camera_id, ts, image (BGR uint8), frame_id}`.

### 3.2 Quality and change gate

- Blur: variance of Laplacian on grey image; reject if below a per-camera threshold (calibrate on 100 good frames; start at 0.5 x median).
- Glare: share of pixels with V > 245 in HSV inside each bay ROI; reject the bay if above 8%.
- Exposure: mean luminance outside [40, 220].
- Quality score `q` in [0,1] combining the three (e.g. product of clipped normalised scores).
- Change: frame differencing per bay ROI against the last processed frame (downscaled 160 px, Gaussian blur, absolute difference, share of changed pixels > 2%). Only changed bays go to heavy models. Force a full pass every 60 s.
- Output: `GatedFrame{frame, q, changed_bays: [bay_id]}`.

### 3.3 Visibility mask (person and cart)

- Model: YOLO-seg small (e.g. YOLO11n-seg or newer), COCO "person" works out of the box. Carts and baskets: fine-tune with pseudo-labels (Grounding DINO prompt "shopping cart . shopping basket", SAM 2 masks).
- Per slot: `v_s = 1 - |mask ∩ slot_polygon| / |slot_polygon|`.
- Output: `Visibility{slot_id: v_s}`.
- Privacy: the same mask blurs or blacks out people in every saved crop.

### 3.4 Pack and gap detector

- Model: Ultralytics YOLO n or s variant; two classes: `pack` (class-agnostic product facing) and `gap` (visible shelf back where product should be).
- Input: 1080p frame split into two 640 px tiles (with 10% overlap); merge with Soft-NMS or WBF.
- Important settings for dense shelves: `max_det` 1000 (SKU-110K averages 147 objects per image, some exceed 500), lower NMS IoU (0.5-0.6).
- Output: `Detections[{bbox_xyxy, cls, conf}]`.

### 3.5 Shelf geometry and slot coverage

- Shelf-edge detection: a small segmentation or line model for shelf edges, or Hough lines on the price-rail band; pseudo-labels can be derived from SKU-110K by clustering box bottoms into rows.
- Homography `H` per bay from four shelf-edge corner points (set once at install, refined automatically) maps image points to shelf coordinates in cm.
- Slot assignment: a pack belongs to the slot containing its bottom-centre after projection; width overlap breaks ties.
- Coverage of slot s:

$$c_s = \frac{\sum_{b \in \mathcal{B}} |b \cap s|}{|s|}, \qquad \tilde{\mathbf{p}}' = H\,\tilde{\mathbf{p}}$$

- Camera drift: ORB/SIFT feature matching to a reference frame; if reprojection error > 8 px, pause alerts for that camera and request recalibration.
- Output: `SlotObs{slot_id, ts, coverage c_s, packs:[crop, bbox_cm], v_s, q}`.

### 3.6 Product identity (catalogue retrieval with abstention)

- Backbone: DINOv2 ViT-S/14 (21M params), input 224 px crops.
- Head: linear projection to 256-d, L2-normalised, trained with ArcFace (additive angular margin; scale 30-64, margin 0.2-0.5).
- Index: FAISS inner-product (`IndexFlatIP` for under 50k vectors; `IndexHNSWFlat` above). 3-5 gallery images per SKU, both studio pack shots and shelf crops.
- Similarity and decision:

$$s_k = \cos\big(f(x), f(g_k)\big), \qquad \text{accept } k^{*} \iff s_{(1)} \ge 0.75 \;\wedge\; s_{(1)} - s_{(2)} \ge 0.15$$

  Otherwise return `unknown`. Thresholds are starting values; calibrate on a held-out open-set split (Section 7).
- OCR tie-break (only for ambiguous crops): PaddleOCR mobile models; compare token overlap of brand, flavour and weight (e.g. "70g", "500 ml") with catalogue text.
- Shelf-tag reading: barcode reader (e.g. ZXing or pyzbar) plus OCR of price labels, used to draft layouts.
- Unknown crops go to a manager inbox. Staff scan the barcode and take 3-5 photos; embeddings are pushed to the store index within a minute. No retraining needed.
- Output: `Identity{slot_id, sku_id | "unknown", s1, s2, ocr_used}`.

### 3.7 Planogram alignment

- Each shelf row is a sequence of expected slots (approved layout) and a sequence of observed packs (left to right).
- Needleman-Wunsch alignment with a position term:

$$D(i,j) = \min \begin{cases} D(i-1,j-1) + \alpha\,|x_i - \hat{x}_j| + \beta\,[\mathrm{id}_i \ne \widehat{\mathrm{id}}_j] \\ D(i-1,j) + \gamma_{\mathrm{gap}} \\ D(i,j-1) + \gamma_{\mathrm{ins}} \end{cases}$$

- Outcomes per slot: match, gap (restock), substitution (displacement), insertion (foreign item).
- Starting costs: α = 0.05 per cm, β = 1.0, γ_gap = 0.8, γ_ins = 0.8. Tune on annotated rows.
- Layout sources:
  - Imported: planogram CSV/JSON with `bay, shelf, x_cm, width_cm, sku_id, facings`.
  - Drafted: shelf-tag barcodes/OCR plus stable observations over the first days.
  - Approved: a manager confirms per bay; only approved versions drive alerts.
- Bay-level guard: if more than 40% of slots in a bay mismatch at once, raise one "layout changed?" question.

### 3.8 Temporal state engine (Built)

Each slot is treated like a cell in an occupancy grid (Moravec and Elfes). Evidence is the log-odds that the slot needs restock.

$$L_t = e^{-\lambda \Delta t} L_{t-1} + v_t\, q_t\, \ell(c_t)$$

$$\ell(c) = \begin{cases} +1.6, & c < 0.40 \text{ (looks low or empty)} \\ -2.2, & c > 0.45 \text{ (looks stocked)} \\ 0, & \text{otherwise} \end{cases}$$

$$\text{decision}_t = \begin{cases} \text{CANNOT SEE}, & v_t < 0.6 \text{ or } q_t < 0.5 \\ \text{ACT}, & L_t \ge \tau_{\text{act}} = 4.0 \\ \text{CLEAR}, & L_t \le \tau_{\text{clr}} = -1.0 \text{ (after a task)} \\ \text{WAIT}, & \text{otherwise} \end{cases}$$

| Parameter | Value | Meaning |
|---|---|---|
| `min_visibility` | 0.6 | Below this the view adds no evidence |
| quality floor | 0.5 | Below this the view adds no evidence |
| `empty_cov` | 0.40 | Coverage below this looks low or empty |
| `full_cov` | 0.45 | Coverage above this looks stocked |
| `llr_empty` | 1.6 | Evidence added by one clean low view |
| `llr_stocked` | 2.2 | Evidence removed by one clean stocked view |
| `decay_per_s` (λ) | 0.02 | Old evidence fades |
| `act_threshold` | 4.0 | About 3 clean confirming views |
| `clear_threshold` | -1.0 | Verifies a refill (hysteresis) |
| `unknown_after_s` | 20 | No clean view this long: show CANNOT SEE in the UI |
| evidence clamp | [-6, 8] | Old states can be overturned |

Engine outputs: ACT (open task), WAIT (change seen, evidence weak), UNKNOWN / CANNOT SEE (blocked or unusable; UI shows last trusted state and its age), CLEAR (slot verified full), OK.

Reference implementation (`engine.py`, Built):

```python
@dataclass
class EngineConfig:
    min_visibility: float = 0.6
    empty_cov: float = 0.40
    full_cov: float = 0.45
    llr_empty: float = 1.6
    llr_stocked: float = 2.2
    decay_per_s: float = 0.02
    act_threshold: float = 4.0
    clear_threshold: float = -1.0
    unknown_after_s: float = 20.0

class TemporalEngine:
    def update(self, slot_id, t, coverage, visibility, quality=1.0) -> str:
        # decay; skip if v < 0.6 or q < 0.5 (WAIT, or UNKNOWN after 20 s);
        # add w*1.6 if coverage < empty_cov, subtract w*2.2 if coverage > full_cov; clamp [-6, 8];
        # HEALTHY -> EMPTY on score >= 4.0 (ACT); EMPTY -> HEALTHY on score <= -1.0 (CLEAR)
```

Baselines in the same file: `SingleFrameBaseline(empty_cov, use_mask)` alerts on the first empty-looking frame; `MajorityVoteBaseline(n=5, k=3, use_mask=True)` alerts when 3 of the last 5 frames look empty.

Planned extension: learn `llr_empty`, `llr_stocked` and thresholds per store from recorded footage (logistic calibration of ℓ(c) against labelled episodes), and add a LOW state between HEALTHY and EMPTY.

### 3.9 Task service

- Adaptive low-stock threshold (units):

$$\theta_s = \hat{d}_s\,(T_{\text{resp}} + T_{\text{safe}})$$

  where d̂ is the demand rate (billing data if connected, otherwise the camera's depletion record), T_resp is staff response time, T_safe a safety margin. Fast movers warn earlier.

- Priority:

$$P_s = \hat{d}_s\,T_{\text{gap}}\,m_s\,p_{\text{lost}} + \beta\,\mathbf{1}[\text{critical}]$$

  d̂ demand rate, T_gap expected gap time, m unit margin, p_lost 0.3-0.5 per category (from shopper response data: 31% buy elsewhere + 9% do not buy = 40% lost to the store), critical = chilled item out of place, promoted item, top seller.

- Time-to-empty with a Gamma posterior on demand (Poisson depletion):

$$\widehat{T}_{\text{empty}} = \frac{u_s}{\hat{d}_s}, \qquad \hat{d}_s \sim \mathrm{Gamma}(a_0 + n_s,\; b_0 + t_s)$$

  u_s units left (estimated from coverage × facings × depth), n_s units removed over observed time t_s; report a range from posterior quantiles.

- Alert types and routing:

| Alert | Trigger | Routed to |
|---|---|---|
| Restock: low | Coverage below the slot threshold | Floor staff, batched per aisle |
| Restock: empty | No product in the slot | Backroom staff, immediately |
| Displacement | Product sits in another product's slot | Floor staff |
| Cold chain | Chilled item seen on an ambient shelf | Floor staff, top priority |
| Record check | Empty for hours while the stock system shows units | Inventory team |

- Task lifecycle:

| State | Entered when | Leaves when |
|---|---|---|
| Open | Engine outputs ACT | Staff tap "I am on it" |
| Acknowledged | Staff accept the task | Staff check in at the bay |
| In progress | Person at bay + app check-in | Staff mark done |
| Verifying | Staff mark done | Clean views show slot full (CLEAR) |
| Verified | Camera or photo confirms | Final; time-to-fix recorded |
| Reopened | Slot still empty after time limit | Returns to Open with a note |

- Alert-fatigue controls: empty and cold-chain tasks go at once; low-stock tasks batch every 15 minutes per aisle; cap on open tasks per team (extra tasks queue nearest-first); "snooze with reason" (e.g. delivery due at 16:00); unanswered tasks escalate after 10 minutes.
- Feedback: "Wrong alert" taps become labelled training data.

Task message example: "EMPTY · Aisle 4 · Shelf 2 · Slot 3 · Priority 1. Maggi 70g. Bring 12 packs from backroom rack B-14. Rack B-14 shows 2-3 cases. Refill by 18:45." With before/now crops and "Why this alert?": clean views confirming (3 views in 22 s), coverage (0%, was 85%), identity score of the product before the gap (0.91), share of views blocked (0%), layout version (v3, approved 28 Sep). Buttons: I am on it, Wrong alert, Snooze.

### 3.10 Interfaces

- Manager dashboard: Next.js; live shelf view, slot history, replay of any alert, unknown-product inbox, layout approval.
- Staff app: mobile web on store Wi-Fi, works offline with a local queue.
- Messaging: Telegram Bot API; WhatsApp Business (Cloud API).
- Languages: English, Hindi, Gujarati (final build).

---

## 4. Data model

PostgreSQL on the edge (and in the cloud twin, with row-level security per tenant). Masked crops go to an object store (local disk on edge; S3-compatible in cloud).

```sql
CREATE TABLE store   (store_id TEXT PRIMARY KEY, name TEXT, tz TEXT);
CREATE TABLE camera  (camera_id TEXT PRIMARY KEY, store_id TEXT, kind TEXT, -- cctv|shelf|phone
                      rtsp_url TEXT, ref_frame_uri TEXT, calib_json JSONB);
CREATE TABLE bay     (bay_id TEXT PRIMARY KEY, store_id TEXT, aisle TEXT, camera_id TEXT,
                      roi_polygon JSONB, homography JSONB, temperature_zone TEXT); -- ambient|chilled|frozen
CREATE TABLE sku     (sku_id TEXT PRIMARY KEY, barcode TEXT, name TEXT, brand TEXT, variant TEXT,
                      size_text TEXT, temperature_zone TEXT, unit_margin NUMERIC, category TEXT);
CREATE TABLE sku_gallery (id BIGSERIAL PRIMARY KEY, sku_id TEXT, image_uri TEXT, source TEXT, -- studio|shelf
                      embedding VECTOR(256) /* or BYTEA */, model_version TEXT);
CREATE TABLE layout_version (layout_id BIGSERIAL PRIMARY KEY, bay_id TEXT, version INT,
                      status TEXT, -- observed|draft|approved
                      approved_by TEXT, approved_at TIMESTAMPTZ);
CREATE TABLE slot    (slot_id TEXT PRIMARY KEY, layout_id BIGINT, shelf INT, x_cm REAL, width_cm REAL,
                      sku_id TEXT, facings INT, depth_units INT, low_threshold_units REAL);
CREATE TABLE observation (id BIGSERIAL PRIMARY KEY, slot_id TEXT, ts TIMESTAMPTZ, frame_id TEXT,
                      coverage REAL, visibility REAL, quality REAL, sku_pred TEXT, s1 REAL, s2 REAL,
                      crop_uri TEXT);
CREATE TABLE slot_state (slot_id TEXT PRIMARY KEY, state TEXT, score REAL, last_clean_ts TIMESTAMPTZ,
                      updated_ts TIMESTAMPTZ);
CREATE TABLE state_event (id BIGSERIAL PRIMARY KEY, slot_id TEXT, ts TIMESTAMPTZ, from_state TEXT,
                      to_state TEXT, score REAL, decision TEXT, frame_ids TEXT[], engine_version TEXT);
CREATE TABLE task    (task_id BIGSERIAL PRIMARY KEY, slot_id TEXT, kind TEXT, -- restock_low|restock_empty|displacement|cold_chain|record_check
                      priority REAL, status TEXT, assignee_team TEXT, created_ts TIMESTAMPTZ,
                      ack_ts TIMESTAMPTZ, done_ts TIMESTAMPTZ, verified_ts TIMESTAMPTZ,
                      backroom_location TEXT, reason_json JSONB);
CREATE TABLE feedback (id BIGSERIAL PRIMARY KEY, task_id BIGINT, user_role TEXT, kind TEXT, -- wrong_alert|snooze|label_unknown
                      note TEXT, ts TIMESTAMPTZ);
```

Event bus between modules: in-process queues for the prototype; MQTT or Redis Streams on the edge for the final build. Cloud sync is outbound only (HTTPS/mTLS), store-and-forward when offline.

---

## 5. Datasets

No organiser data is assumed. Check each licence before use; most academic datasets are for non-commercial research only, which is acceptable for the hackathon but must be reviewed for a product.

| Dataset | Content | Use in ShelfTwin | Access |
|---|---|---|---|
| SKU-110K (Goldman et al., CVPR 2019) | 11,762 dense shelf images; about 1.73M boxes; 147 objects per image on average; single class "object"; split 8,219 train / 588 val / 2,936 test | Pre-train the class-agnostic pack detector; dense-scene tests; source of real shelf photos for figures | github.com/eg4000/SKU110K_CVPR19. Ultralytics config `SKU-110K.yaml` downloads `http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz` (about 13.6 GB) and converts to YOLO format. Kaggle mirrors exist (search "SKU110K"); verify file counts. |
| Locount (Cai et al., AAAI 2021) | 50,394 images; about 1.9M instances; 140 classes; count labels | Counting and robustness tests; extra detector data | isrc.iscas.ac.cn/gitlab/research/locount-dataset |
| RP2K (Peng et al., 2020) | 2,388 SKUs; about 500k product crops from shelves | Metric learning for identity (ArcFace) | pinlandata.com/rp2k_dataset; arXiv:2006.12634 |
| Retail-786k (Lamm and Keuper, 2023) | 786k images; 3,298 product groups ("visual entities") | Look-alike pairs and hard negatives for the embedder | zenodo.org/records/7970567; arXiv:2309.17164 |
| Unitail (Chen et al., ECCV 2022) | Unitail-Det (quadrilateral boxes) and Unitail-OCR (1,454 products, about 30k text regions) | OCR fine-tuning and text-based matching | unitedretail.github.io. Det on academic request only. |
| CPGDet-129 (Neurolabs, 2022) | Synthetic rendered retail shelves, 129 products | Synthetic pre-training reference; rendering ideas | github.com/neurolaboratories/reshelf-detection |
| StandardSim (2022) | Synthetic store with change-detection labels | Ideas for change and gap simulation | arXiv:2202.02418 |
| COCO (person) | Persons with masks | Visibility mask, ready-trained weights | Through Ultralytics pretrained weights |
| Own store capture | Phone and camera video of local stores, with permission | Fine-tuning, catalogue gallery, real test sets, temporal episodes | Collected by the team (Section 5.1) |
| Rendered shelves | Shelves rendered from our pack photos with BlenderProc2 | Rare events: gaps, misplacements, glare, look-alikes | Generated (Section 5.3) |

### 5.1 Own data capture protocol

- Get written permission from the store manager. Avoid faces; mask people before storing anything beyond the session.
- Target: 3-5 stores, 60-80 Indian SKUs on a demo shelf, at least 3 camera types (fixed shelf-edge, ceiling CCTV, phone walk-through).
- Fixed cameras: record 2-4 hours per session at 1-2 fps, at different times of day, including busy periods (natural occlusion) and a refill event.
- Phone: walk each aisle at constant speed, 1080p, 30 fps; extract 2 fps.
- Catalogue: for each SKU, 3-5 studio shots (front, two angles, back) on a plain background, plus barcode scan.
- Scripted episodes for evaluation: remove units until empty; move a pack to a wrong slot; place a chilled item on an ambient shelf; block slots with a person or cart for 2-30 s; refill.
- Log ground truth with a simple timestamped sheet (or tap app): slot id, event, time.

### 5.2 Labelling without hand-drawing every box

1. Grounding DINO with prompts like "product package . bottle . box . packet" proposes boxes; SAM 2 refines masks.
2. A second teacher (SKU-110K-trained YOLO) also proposes boxes.
3. Keep labels where the two teachers agree (IoU ≥ 0.6); send disagreements to manual review (CVAT or Label Studio).
4. Hand-check a random 5% sample per batch; reject the batch if box precision is below 95%.
5. Gap labels: mark visible shelf back between packs. Generate extra gaps by erasing pack boxes and in-painting with shelf-back texture (Section 5.3).
6. Staff feedback in deployment ("wrong alert", unknown-pack labels) becomes new data.
7. Rendered and edited images are used for training only, never for testing.

### 5.3 Synthetic data

- BlenderProc2: model shelves (standard Indian gondola dimensions), place textured pack meshes (box, bottle, pouch primitives with our pack photos as textures), randomise lighting, camera pose, glare, facings, gaps and misplacements. Output COCO-format boxes and masks.
- Cheap 2D augmentation: cut-paste of pack crops onto real shelf backgrounds; erase-and-inpaint packs to create gaps (OpenCV `inpaint` or a diffusion in-painter); copy-paste persons from COCO for occlusion.
- Target mix for detector fine-tuning: about 60% real (SKU-110K + own), 40% synthetic; test on real only.

### 5.4 Test splits

- Seen sessions (same store, same day) for debugging only.
- Unseen days and lighting.
- Unseen stores (main reported number).
- Open-set SKU split for identity: 20% of SKUs never seen in training.
- Hard cases: look-alikes, multipacks, old vs new packs, half-hidden packs.

---

## 6. Training recipes

### 6.1 Compute options

| Platform | GPU | Notes |
|---|---|---|
| Kaggle Notebooks | T4 x2 (16 GB each) or P100 (16 GB) | Weekly GPU quota (about 30 h, check current limit); sessions up to about 12 h; `/kaggle/input` is read-only; `/kaggle/working` keeps up to about 20 GB of outputs; enable internet in settings (phone-verified account). Save checkpoints every epoch and resume across sessions. |
| AWS EC2 | g4dn.xlarge (T4 16 GB), g5.xlarge (A10G 24 GB), g6.xlarge (L4 24 GB) | Use Deep Learning AMI (PyTorch). Spot instances cut cost substantially; checkpoint to S3 every epoch. Check current on-demand prices for your region. |
| AWS SageMaker | ml.g4dn / ml.g5 training jobs | Managed spot training with checkpoint S3 URI; data from S3 via FastFile mode. |

General tips:

- Pin library versions in `requirements.txt`; log runs to Weights & Biases or MLflow (or CSV on Kaggle).
- Large datasets: on Kaggle add SKU-110K as an input dataset rather than downloading 13.6 GB each session. On AWS keep datasets in S3 in the same region; copy to local NVMe for training.
- Use mixed precision (AMP). Seed everything. Save `best.pt`, `last.pt`, config and metrics per run.

### 6.2 Notebook 01: pack detector pre-training on SKU-110K

```python
# pip install ultralytics
from ultralytics import YOLO
model = YOLO("yolo11s.pt")              # or the newest small YOLO; n for faster edge
model.train(
    data="SKU-110K.yaml",                # Ultralytics config; converts to YOLO labels on first run
    imgsz=1024,                          # dense scenes need resolution; try 640 tiles vs 1024 full
    epochs=60, batch=16, device=[0,1],   # Kaggle T4 x2; lower batch if OOM
    optimizer="AdamW", lr0=0.002, cos_lr=True,
    mosaic=1.0, close_mosaic=10, fliplr=0.5, degrees=0, perspective=0.0005,
    max_det=1000, workers=4, amp=True, patience=15,
    project="runs/sku110k", name="y11s_1024",
)
metrics = model.val(data="SKU-110K.yaml", split="test", max_det=1000, iou=0.6)
```

- Report mAP50 and mAP50-95 on the SKU-110K test split.
- Check time per epoch in the first session and set `epochs` to fit the quota; resume with `model.train(resume=True)`.

### 6.3 Notebook 02: two-class detector fine-tune (pack + gap)

- Data: own capture (labelled per Section 5.2) + synthetic gaps + SKU-110K subset relabelled as `pack`.
- Start from Notebook 01 weights; `nc=2`, names `[pack, gap]`.
- Train at 640 with tiling of 1080p frames (2 tiles, 10% overlap) to match deployment.
- 40-80 epochs, early stopping on val mAP of `gap`.
- Metrics: per-class AP; gap recall at fixed precision 0.9; slot-level coverage error (mean absolute error of c_s versus labels).

### 6.4 Notebook 03: visibility mask

- Start with pretrained YOLO-seg (COCO) for `person`.
- Add `cart` and `basket` with 300-1,000 pseudo-labelled frames (Grounding DINO + SAM 2), 30-50 epochs.
- Metric: mask IoU on persons and carts; slot visibility error.

### 6.5 Notebook 04: product embedder (DINOv2 + ArcFace)

```python
# pip install torch torchvision timm pytorch-metric-learning faiss-cpu
import torch, timm
from pytorch_metric_learning import losses, samplers
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")   # 384-d CLS
head = torch.nn.Sequential(torch.nn.Linear(384, 256))
loss_fn = losses.ArcFaceLoss(num_classes=N_TRAIN_SKUS, embedding_size=256, margin=28.6, scale=64)  # margin in degrees (0.5 rad)
# sampler: PK sampling, P=64 classes x K=4 images per batch
sampler = samplers.MPerClassSampler(labels, m=4, batch_size=256)
# optim: AdamW; backbone lr 1e-5 (unfreeze last 4 blocks), head + loss lr 1e-3; 20-30 epochs; AMP
# augment: RandomResizedCrop(224, scale=(0.6,1)), ColorJitter(0.3,0.3,0.3,0.05), small perspective,
#          random erasing (simulates partial occlusion); no horizontal flip (text on packs)
```

- Training data: RP2K + Retail-786k + own shelf crops and studio shots. Mine hard negatives from the same brand (sample batches by brand).
- Evaluation: Recall@1, Recall@5 and mAP@R on the open-set SKU split; plot the s1 and s1-s2 distributions for correct, wrong and unknown cases and pick thresholds that give at least 98% precision on accepted matches.
- Export: ONNX (dynamic batch), then TensorRT FP16 on Jetson. Build the FAISS index from gallery embeddings.

### 6.6 Notebook 05: OCR

- Use PaddleOCR pretrained mobile detection and recognition models; test on Unitail-OCR and own crops (English and Hindi pack text).
- Fine-tune recognition only if word accuracy on own crops is below 85%.
- Token matching: normalise text (lowercase, units such as g/kg/ml/l), Jaccard overlap with catalogue tokens.

### 6.7 Notebook 06: shelf geometry

- Shelf-edge model: YOLO-seg on shelf-edge strips (pseudo-labels from row clustering of SKU-110K boxes plus own manual labels on 200-300 frames).
- Homography from detected edge lines; evaluate reprojection error on 20 bays with measured slot widths.

### 6.8 Notebook 07: temporal engine calibration and simulation (Built, extend)

- Run the simulator (`python simulate.py 20`) to reproduce the prototype table.
- Replace simulated detector outputs with recorded real per-frame outputs from Notebooks 02-03 (coverage, visibility, quality per slot per frame), and the scripted ground-truth episodes.
- Grid-search `llr_empty`, `llr_stocked`, `act_threshold`, `decay_per_s` to minimise false alerts per camera-hour subject to recall ≥ 99% and p95 delay ≤ 30 s.

### 6.9 Notebook 08: export and benchmark

- Export each model: `model.export(format="onnx", dynamic=True, simplify=True)`; on Jetson build TensorRT engines (`format="engine", half=True` or INT8 with 300-500 calibration frames).
- Benchmark per-stage latency on the target device and fill the latency budget table (Section 9.2).

---

## 7. Evaluation protocol

### 7.1 Prototype simulation (Built)

Simulator assumptions (`simulate.py`):

- 1 camera, 24 slots, one frame every 2 s, 8 trading hours per run, 20 runs (seeds 0-19); 160 camera-hours.
- Each slot starts with 6-12 units; shoppers take units as a Poisson process at 0.5-6 units per slot per hour.
- Shoppers block each slot about 25 times per hour for 2-30 s (log-normal, mean log 1.8, sd 0.7, clipped).
- While blocked, the detector reports a gap 70% of the time. The person mask misses 5% of block events.
- Glare and blur: 1% of clean frames of a well-stocked slot read as a gap.
- Low stock (1-2 units) reads as coverage 0.05-0.35; empty reads as 0-0.06; 3% of empty frames misread as stocked.
- Ground truth: a slot "needs restock" when 2 or fewer units remain. An alert is correct if it fires while the slot needs restock.
- Staff refill a slot 10-40 minutes after it needs restock, identically for every engine.

Results (1,357 true restock events):

| Method | False alerts / camera-hour | Precision | Recall | Median delay | p95 delay |
|---|---|---|---|---|---|
| Single-frame | 1019.33 | 11.6% | 100% | 0 s | 2 s |
| Single-frame + person mask | 378.67 | 13.1% | 99.7% | 0 s | 4 s |
| Majority vote 3 of 5 + mask | 12.81 | 42.2% | 99.9% | 4 s | 10 s |
| ShelfTwin engine | 0.58 | 93.6% | 100% | 12 s | 20 s |

Scope: these numbers test the decision logic on simulated detector outputs. Real-camera accuracy must be measured in the final build. The engine trades about 12 s of extra delay for a 99.9% cut in false alerts versus single-frame.

### 7.2 Final build metrics

| Level | Metric |
|---|---|
| Detector | mAP50, mAP50-95 (pack, gap); gap recall at precision 0.9 |
| Coverage | MAE of c_s per slot |
| Identity | Recall@1, mAP@R; precision of accepted matches; correct "unknown" rate on new packs |
| Alignment | Slot-level accuracy of match/gap/substitution/insertion |
| Engine (episodes) | Alert precision and recall per episode; false alerts per camera-hour; detection delay (median, p95) |
| System | End-to-end notification delay; cameras per box at full speed; upload MB/day |
| Workflow | Time-to-acknowledge, time-to-fix, verified-closure rate |

Report every metric on unseen-store data, with the seen-session numbers only as a debug reference.

---

## 8. Repository layout and build plan

```
shelftwin/
  README.md
  requirements.txt
  configs/
    engine.yaml  detector.yaml  embedder.yaml  store_demo.yaml
  shelftwin/
    capture/        rtsp.py  video_file.py
    gate/           quality.py  change.py
    vision/         detector.py  visibility.py  geometry.py  identity.py  ocr.py  tiling.py
    planogram/      align.py  layouts.py  draft_from_tags.py
    engine/         temporal.py  baselines.py      # from prototype engine.py
    tasks/          priority.py  threshold.py  tte.py  service.py  lifecycle.py
    store/          db.py  models.sql  crops.py
    api/            server.py (FastAPI)  sync.py
    notify/         telegram.py  whatsapp.py  i18n/{en,hi,gu}.json
  dashboard/        Next.js app
  sim/              simulate.py  figures.py  results.json
  notebooks/
    01_sku110k_pretrain.ipynb
    02_pack_gap_finetune.ipynb
    03_visibility_mask.ipynb
    04_embedder_arcface.ipynb
    05_ocr_eval.ipynb
    06_shelf_geometry.ipynb
    07_engine_calibration.ipynb
    08_export_benchmark.ipynb
  data/             (not committed) raw/, labels/, synthetic/, catalogue/
  tests/            test_engine.py  test_align.py  test_priority.py  test_geometry.py
```

Build order:

1. Port `engine.py`, `simulate.py`, `figures.py` into the package; add unit tests.
2. Notebook 01 (detector pre-train) in parallel with own data capture.
3. Labelling pipeline (Grounding DINO + SAM 2 + agreement filter) and Notebook 02.
4. Notebook 04 (embedder) and the FAISS catalogue with the unknown-product inbox.
5. Geometry, alignment and the end-to-end pipeline on recorded video.
6. Task service, staff app, Telegram bot; then WhatsApp and Hindi/Gujarati strings.
7. Engine calibration on real episodes (Notebook 07).
8. Export, Jetson benchmark (Notebook 08), dashboard replay.

Final build demonstration targets:

- A shelf with 60-80 real Indian SKUs and 3-4 cameras.
- Shoppers block views; the baseline raises false alerts while ShelfTwin waits.
- A real low-stock alert in Hindi names the backroom rack.
- A refill is verified by the camera and the task closes.
- An unseen pack is flagged "unknown" and added in a minute.

---

## 9. Deployment

### 9.1 Architecture

| Component | Technique / model | Runs on | Rate |
|---|---|---|---|
| Frame capture | RTSP/USB readers; hardware decode (GStreamer) | Edge box | 0.5-2 fps per camera |
| Quality and change gate | Laplacian blur, glare ratio, frame differencing per bay | Edge box | Every frame |
| Visibility mask | Person/cart instance segmentation (YOLO-seg, small) | Edge box | Changed bays |
| Pack and gap detector | YOLO (n/s), two classes, 2 tiles per 1080p frame | Edge box | Changed bays |
| Shelf geometry | Shelf-edge lines, homography, slot assignment | Edge box | On camera move |
| Product identity | DINOv2 ViT-S/14 + ArcFace; FAISS k-NN; margin test | Edge box | Changed slots |
| Pack and tag text | PaddleOCR (mobile); barcode reader | Edge box | Ambiguous only |
| Planogram matcher | Needleman-Wunsch with position cost | Edge box | Per row update |
| Temporal state engine | Visibility-weighted log-odds, decay, hysteresis | Edge box | Every observation |
| Task service | Priority, routing, verification, escalation | Edge + cloud | Event-driven |
| Twin store and replay | PostgreSQL + object store for masked crops | Edge + cloud | Event-driven |
| Interfaces | Next.js dashboard; staff web app; Telegram/WhatsApp | Cloud + LAN | Real time |

Edge box: NVIDIA Jetson Orin-class (e.g. Orin Nano Super, 67 sparse INT8 TOPS, launched at USD 249 for the developer kit). Models in ONNX/TensorRT (FP16/INT8). Cameras on PoE to the box. Video stays in the store; only events and masked crops are sent out.

### 9.2 Latency budget (planning estimate, to measure on hardware)

| Stage | ms per frame |
|---|---|
| Decode + quality check | 4 |
| Person/cart mask | 6 |
| Pack + gap detection (2 tiles) | 18 |
| Slot mapping (homography) | 1 |
| Catalogue match (changed slots only) | 10 |
| OCR (ambiguous packs only) | 6 |
| State engine + task logic | 1 |
| **Total** | **about 46** |

Sizing: about 20 frames/s per box; 12-20 cameras per box at 0.5-1 fps per camera with change gating; upload under 200 MB per store per day; about 1 TB SSD for a 72 h replay buffer plus crops.

Camera options: existing CCTV (bay-level gaps, no hardware cost); shelf-edge cameras (product-level, real time, per-bay cost); phone walk-through (product-level snapshots, no hardware cost).

### 9.3 Scalability

- Compute scales per store, so head-office cost grows slowly.
- One shared product catalogue; each store loads only its own range.
- Models, catalogue, layouts and rules are versioned separately; staged rollout to 1%, 10%, then all stores, with automatic rollback.
- Multi-tenant cloud (row-level security) or private deployment for large chains.
- The same twin extends to pharmacies, dark stores and warehouses.

### 9.4 Costs (mid-size Indian supermarket, planning estimates)

| Item | Qty | Unit cost (Rs) | Total (Rs) |
|---|---|---|---|
| Edge GPU box, installed (Jetson Orin class, case, SSD, UPS) | 1 | 45,000-70,000 | 45,000-70,000 |
| Shelf-edge camera, installed (cabling, mount) | 20 | 6,000-12,000 | 1.2-2.4 lakh |
| PoE switch and network work | 1 | 20,000-35,000 | 20,000-35,000 |
| Existing CCTV reuse | - | 0 | 0 |
| **One-time hardware** | | | **about 1.9-3.5 lakh** |
| Cloud twin, storage, messaging | per year | | 15,000-30,000 |
| Software subscription and support | per year | | 1-2 lakh |
| Maintenance (10-15% of hardware) | per year | | 20,000-50,000 |

Worked example: store revenue Rs 10 crore/year; lost to out-of-stocks (3-4%) Rs 30-40 lakh; recovered (assume 25-50%) Rs 7.5-20 lakh; at 20-25% gross margin, Rs 1.5-5 lakh profit per year, plus staff audit hours saved. Payback about 6-24 months. Vendor case studies (e.g. Simbe with Wakefern) report on-shelf availability gains of 25-60%; a pilot must confirm this for Indian stores.

Cost levers: shelf cameras only on the 20-30% of bays carrying most sales; change gating to raise cameras per box; hardware leased inside the subscription.

### 9.5 Privacy and security

Privacy:

- Continuous video never leaves the store.
- People, hands, phones and badges are masked in every saved crop.
- No face recognition, no re-identification, no tracking of people.
- Crops kept 30 days by default; replay frames 72 hours.
- Staff metrics are team-level, not individual.
- Each deployment is reviewed under India's Digital Personal Data Protection Act 2023 and DPDP Rules 2025 (notified November 2025, obligations phased in to May 2027).

| Data | Leaves the store? | Retention | Who can see it |
|---|---|---|---|
| Live video | No | Seconds (memory only) | Nobody |
| Replay frames (people masked) | No | 72 hours | Store admin, on request |
| Slot crops (people masked) | Yes, with events | 30 days | Store and HQ roles |
| Event and task records | Yes | Configurable | Store and HQ roles |
| Model and system telemetry | Yes | 90 days | Operations team |

Security: edge box makes outbound connections only (no open ports); mutual TLS with per-device certificates; signed model and software updates; secure boot; encrypted disks; role-based access with every crop view logged; camera network on its own VLAN.

### 9.6 Limitations (version 1 does not claim)

- Exact unit counts or store-wide inventory.
- Automatic planogram changes or ordering.
- Perfect separation of near-identical packs (it abstains instead).
- Product-level identity from wide ceiling CCTV.
- Camera depletion is not called "sales" without billing data.

Main risks and mitigation: look-alike packs (abstain and ask staff); low-resolution CCTV (bay-level alerts only); camera moved (drift check pauses alerts); store lighting changes (quality gate, unseen-lighting test split); label noise from teachers (agreement filter and audits).

---

## 10. Existing approaches and how ShelfTwin differs

| Approach / work | What it does well | Limitation for this problem |
|---|---|---|
| Manual audits | No hardware; staff know the store | Slow, infrequent, error-prone; no time record |
| Shelf-scanning robots (e.g. Simbe Tally) | Whole-store scans several times a day | High cost; periodic, not continuous |
| Dense detection: SKU-110K | Finds every pack in crowded shelves | No product identity; single image |
| Counting: Locount | Localises and counts product groups | Single image; no expected layout |
| Text matching: Unitail | Reads pack text to match products | Recognition only; no alerts |
| Planogram compliance (Ou et al., YOLOv8, 7,000+ 7-Eleven stores) | Real deployment at scale | Needs a digital planogram; 471 fixed classes |
| Retail vision platforms (e.g. Everseen Evershelf, Everstock) | Store-wide cameras; shelf and loss products | ShelfTwin adds a trusted, verified slot layer |

ShelfTwin's contribution: decisions over time with explicit uncertainty instead of per-frame alerts; works from an approved layout when no digital planogram exists; an open product catalogue that abstains; a task loop closed by camera proof.

---

## 11. Useful tools and libraries

| Purpose | Tool | Link |
|---|---|---|
| Detection, segmentation, export | Ultralytics YOLO | github.com/ultralytics/ultralytics, docs.ultralytics.com |
| SKU-110K dataset docs | Ultralytics | docs.ultralytics.com/datasets/detect/sku-110k/ |
| Backbone | DINOv2 | github.com/facebookresearch/dinov2 |
| Metric learning losses and samplers | pytorch-metric-learning | github.com/KevinMusgrave/pytorch-metric-learning |
| Vector search | FAISS | github.com/facebookresearch/faiss |
| OCR | PaddleOCR | github.com/PaddlePaddle/PaddleOCR |
| Open-set teacher | Grounding DINO | github.com/IDEA-Research/GroundingDINO |
| Masks teacher | SAM 2 | github.com/facebookresearch/sam2 |
| Synthetic rendering | BlenderProc2 | github.com/DLR-RM/BlenderProc |
| Annotation | CVAT, Label Studio | cvat.ai, labelstud.io |
| Box fusion | ensemble-boxes (WBF) | github.com/ZFTurbo/Weighted-Boxes-Fusion |
| Edge inference | TensorRT, ONNX Runtime, JetPack | developer.nvidia.com |
| Video | GStreamer, PyAV, OpenCV | gstreamer.freedesktop.org |
| Backend | FastAPI, PostgreSQL, MQTT (Mosquitto) | fastapi.tiangolo.com |
| Messaging | Telegram Bot API; WhatsApp Business Cloud API | core.telegram.org/bots/api; developers.facebook.com/docs/whatsapp |
| Experiment tracking | Weights & Biases, MLflow | wandb.ai, mlflow.org |

---

## 12. References

1. T. W. Gruen, D. S. Corsten, S. Bharadwaj. *Retail Out-of-Stocks: A Worldwide Examination of Extent, Causes and Consumer Responses.* Grocery Manufacturers of America, 2002.
2. D. Corsten, T. Gruen. Desperately seeking shelf availability: an examination of the extent, the causes, and the efforts to address retail out-of-stocks. *International Journal of Retail & Distribution Management* 31(12), 2003.
3. E. Goldman, R. Herzig, A. Eisenschtat, J. Goldberger, T. Hassner. Precise Detection in Densely Packed Scenes. *CVPR* 2019. github.com/eg4000/SKU110K_CVPR19
4. Y. Cai et al. Rethinking Object Detection in Retail Stores (Locount). *AAAI* 2021.
5. F. Chen et al. Unitail: Detecting, Reading, and Matching in Retail Scene. *ECCV* 2022. unitedretail.github.io
6. J. Peng, C. Xiao, Y. Li. RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification. arXiv:2006.12634, 2020.
7. B. Lamm, J. Keuper. Retail-786k: a Large-Scale Dataset for Visual Entity Matching. arXiv:2309.17164, 2023.
8. T.-Y. Ou, A. Ponce, C. Lee, A. Wu. Real-time retail planogram compliance application using computer vision and virtual shelves. *Scientific Reports* 15, 43898, 2025.
9. M. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. *TMLR*, 2024.
10. G. Jocher et al. Ultralytics YOLO (YOLOv8, YOLO11, YOLO26). github.com/ultralytics/ultralytics.
11. J. Deng, J. Guo, N. Xue, S. Zafeiriou. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *CVPR* 2019.
12. J. Johnson, M. Douze, H. Jégou. Billion-scale similarity search with GPUs (FAISS). *IEEE Transactions on Big Data*, 2019.
13. S. B. Needleman, C. D. Wunsch. A general method applicable to the search for similarities in the amino acid sequence of two proteins. *J. Mol. Biol.* 48(3), 1970.
14. H. Moravec, A. Elfes. High resolution maps from wide angle sonar. *IEEE ICRA*, 1985.
15. S. Liu et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. *ECCV* 2024.
16. Y. Du et al. PP-OCR: A Practical Ultra Lightweight OCR System. arXiv:2009.09941, 2020.
17. Neurolabs. CPGDet-129: Synthetic Dataset for Object Detection in Retail. github.com/neurolaboratories/reshelf-detection, 2022.
18. M. Denninger et al. BlenderProc2: A Procedural Pipeline for Photorealistic Rendering. *JOSS* 8(82), 2023.
19. NVIDIA. Jetson Orin Nano Developer Kit Gets a Super Boost. NVIDIA Technical Blog, December 2024.
20. N. Ravi et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024.
21. Simbe Robotics. Wakefern Food Corp. customer story. simberobotics.com.
22. Everseen. Evershelf product page; expanded product suite including Everstock (press release, January 2023). everseen.com
23. Ministry of Electronics and IT, Government of India. Digital Personal Data Protection Rules, 2025 (notified November 2025). meity.gov.in
24. StandardSim: A Synthetic Dataset for Retail Environments. arXiv:2202.02418, 2022.

---

## 13. Open items

- Confirm current Kaggle GPU quota and AWS prices before planning runs.
- Confirm licences of RP2K, Retail-786k and Locount for the intended use.
- Measure the latency budget on real Jetson hardware.
- Collect store permissions and run the capture protocol.
- Replace the repository placeholder `github.com/<team>/shelftwin` with the real URL in the deck and README.
