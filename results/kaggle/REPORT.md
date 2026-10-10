# ShelfTwin training report

* **detector decision**: `{"model": "yolo11s.pt", "imgsz": 1024, "mosaic": 1.0, "nms_iou": 0.7}`
* **gap decision**: `{"p_synth": 0.6, "fill": [0.0, 1.0, 0.0]}`
* **embedder decision**: `{"unfreeze_blocks": 4}`
* **engine decision**: `{"llr_empty": 1.2, "llr_stocked": 2.2, "act_threshold": 5.0, "decay_per_s": 0.01}`

## Detector experiments

| name | model | imgsz | mosaic | epochs | split | mAP50 | mAP50_95 | precision |
|---|---|---|---|---|---|---|---|---|
| d_y11s_1024 | yolo11s.pt | 1024 | 1.0000 | 10 | test | 0.9171 | 0.5666 | 0.8868 |
| d_y11s_896 | yolo11s.pt | 896 | 1.0000 | 10 | test | 0.9092 | 0.5585 | 0.8844 |
| d_y11s_896_nomosaic | yolo11s.pt | 896 | 0.0000 | 10 | test | 0.8826 | 0.5427 | 0.8836 |
| d_y11n_1024 | yolo11n.pt | 1024 | 1.0000 | 10 | test | 0.8930 | 0.5371 | 0.8721 |

## Gap-label experiments

| name | p_synth | gap_AP50 | pack_AP50 | gap_recall_at_p90 | n_gap_gt |
|---|---|---|---|---|---|
| real_only | 0.0000 | 0.4615 | 0.8861 | 0.0000 | 2063 |
| mixed | 0.6000 | 0.4583 | 0.9046 | 0.0000 | 2063 |
| shelfback_only | 0.6000 | 0.4714 | 0.9059 | 0.0044 | 2063 |

## Embedder experiments

| name | unfreeze_blocks | lr_head | recall_at_1 | recall_at_5 | map_at_r | n_known_queries | n_unknown_queries | thr_sim |
|---|---|---|---|---|---|---|---|---|
| head_only | 0 | 0.0020 | 0.8789 | 0.9743 | 0.6221 | 545 | 577 | 0.2800 |
| last2 | 2 | None | 0.9211 | 0.9780 | 0.6942 | 545 | 577 | 0.7600 |
| last4 | 4 | None | 0.9523 | 0.9798 | 0.7761 | 545 | 577 | 0.7300 |

## Final results

```json
{
 "pack_detector": {
  "metrics": {
   "split": "test",
   "mAP50": 0.9496650204336342,
   "mAP50_95": 0.6024631473375399,
   "precision": 0.9161125753217194,
   "recall": 0.8885217192852535,
   "per_class_AP50": {
    "pack": 0.9496650204336342
   }
  },
  "error_analysis": {
   "images": 2920,
   "gt": 429411,
   "pred": 454528,
   "precision": 0.8675461137707688,
   "recall": 0.9182904024349633,
   "recall_by_size_quintile": [
    {
     "rel_size": "0.007-0.036",
     "recall": 0.7751242970087212
    },
    {
     "rel_size": "0.036-0.045",
     "recall": 0.9231055842764659
    },
    {
     "rel_size": "0.045-0.053",
     "recall": 0.9580825075975455
    },
    {
     "rel_size": "0.053-0.064",
     "recall": 0.9708214472841591
    },
    {
     "rel_size": "0.064-0.363",
     "recall": 0.9643235564663554
    }
   ],
   "recall_by_density": [
    {
     "boxes_per_image": "0-50",
     "images": 9,
     "recall": 0.7003484320557491
    },
    {
     "boxes_per_image": "50-100",
     "images": 101,
     "recall": 0.8645236703682058
    },
    {
     "boxes_per_image": "100-150",
     "images": 1778,
     "recall": 0.9355902106262527
    },
    {
     "boxes_per_image": "150-250",
     "images": 948,
     "recall": 0.9065184858367665
    },
    {
     "boxes_per_image": "250-10000",
     "images": 84,
     "recall": 0.8614904870924449
    }
   ],
   "count_error": {
    "mae": 13.842123287671233,
    "bias": 8.601712328767123,
    "mape": 0.10702407826251881
   },
   "false_positive_types": {
    "duplicate_or_misaligned": 21465,
    "background": 38739
   }
  }
 },
 "pack_gap_detector": {
  "metrics": {
   "split": "test",
   "mAP50": 0.7580598701208587,
   "mAP50_95": 0.4734725143373752,
   "precision": 0.7527983300455094,
   "recall": 0.721189902729545,
   "per_class_AP50": {
    "pack": 0.9423106021473191,
    "gap": 0.5738091380943984
   }
  },
  "gap_recall_at_p90": {
   "n_gt": 8314,
   "target_precision": 0.9,
   "recall_at_p": 0.07806110175607409,
   "conf_at_p": 0.7724609375,
   "images": 2000
  }
 },
 "e2e_demo": {
  "roles": {
   "A": "R3-P22",
   "B": "R1-P11",
   "C": "R1-P18",
   "D": "R1-P26"
  },
  "n_slots": 112,
  "single_frame_alerts": 53,
  "engine_alerts": 9,
  "engine_correct_alerts": 1,
  "engine_false_alerts": 8,
  "detection_delay_s": 32.0,
  "refill_verified": true,
  "displacement_task": true
 }
}
```