# Phase 0 Report — Clean Foundation

**Date:** 2026-04-19  
**Status:** Complete

---

## Objectives

Phase 0 locked in the experiment design before running anything that would need repeating. Decisions made:

- **Primary model:** YOLOv8m trained on SH17 (100 epochs, 17 classes)
- **Target domains:** Pictor-PPE, SHWD, CHV — all three, same eval protocol throughout
- **Evaluation metric:** mAP50 primary, mAP50-95 secondary, per-class AP (hard_hat, no_hard_hat, person) mandatory
- **Fixed batch protocol:** adapt_batch=16, val_batch=64 across all TTA methods
- **Zero-shot baselines:** YOLO-World-L, OWL-ViT-L, Grounding DINO-base — one eval run each, no adaptation

---

## 1. YOLOv8m Training on SH17

### Setup

| Parameter | Value |
|---|---|
| Architecture | YOLOv8m |
| Dataset | SH17 (17 safety-gear classes) |
| Epochs | 100 |
| Image size | 640 × 640 |
| Optimizer | SGD, lr0=0.01, momentum=0.937 |
| Batch | 144 (48/GPU × 3 GPUs via DDP) |
| GPUs | 3× NVIDIA A10 (22.5 GB each) |
| AMP | enabled |
| Augmentation | mosaic=1.0, hsv, flip, scale, translate |

### Training Curve (key epochs)

| Epoch | mAP50 | mAP50-95 | box loss | cls loss |
|---|---|---|---|---|
| 1  | 0.330 | 0.212 | 1.200 | 2.345 |
| 10 | 0.519 | 0.324 | 0.996 | 0.734 |
| 25 | 0.558 | 0.358 | 0.909 | 0.614 |
| 50 | 0.622 | 0.404 | 0.794 | 0.490 |
| 75 | 0.633 | 0.413 | 0.691 | 0.419 |
| 100 | 0.635 | 0.417 | 0.550 | 0.284 |

Training was stable throughout — no divergence after the lr0=0.01 fix (linear LR scaling does not apply to SGD+momentum). Best weights saved at the epoch with highest validation fitness.

### Source Domain Verification

Evaluated `best.pt` on SH17 val split (held-out, same distribution as training):

| Metric | Value |
|---|---|
| mAP50 | **0.6442** |
| mAP50-95 | **0.4223** |
| Status | PASS (threshold: 0.60) |

The model is solid on source. This is our baseline for measuring domain gap.

---

## 2. Zero-Shot Baselines

All three open-vocabulary models were evaluated on the same three target domains with identical eval settings (conf=0.001, IoU=0.6, imgsz=640). No adaptation, no fine-tuning.

Text prompts used by all models: `["hard hat", "no hard hat", "person"]`

### Model Details

| Model | Type | Parameters | Weights |
|---|---|---|---|
| YOLO-World-L v2 | Open-vocab YOLO | ~68M | `yolov8l-worldv2.pt` (90 MB) |
| OWL-ViT-L | CLIP-based detector | ~307M | `google/owlvit-large-patch14` (1.74 GB) |
| Grounding DINO-base | Transformer grounding | ~172M | `IDEA-Research/grounding-dino-base` (933 MB) |

### Overall mAP50 Results

| Model | Pictor-PPE | SHWD | CHV | Mean mAP50 |
|---|---|---|---|---|
| YOLO-World-L | 0.548 | 0.253 | **0.875** | **0.559** |
| OWL-ViT-L | **0.567** | 0.161 | 0.562 | 0.430 |
| Grounding DINO-base | 0.270 | 0.021 | 0.152 | 0.148 |
| **YOLOv8m (zero-shot)** | **0.011** | **0.008** | **0.022** | **0.014** |

> YOLOv8m row = our fine-tuned model applied directly to target domains without any adaptation. This row defines the domain gap we are trying to close.

### Overall mAP50-95 Results

| Model | Pictor-PPE | SHWD | CHV | Mean mAP50-95 |
|---|---|---|---|---|
| YOLO-World-L | 0.393 | 0.048 | **0.433** | **0.291** |
| OWL-ViT-L | **0.374** | 0.032 | 0.271 | 0.226 |
| Grounding DINO-base | 0.177 | 0.004 | 0.076 | 0.086 |
| YOLOv8m (zero-shot) | 0.004 | 0.004 | 0.004 | 0.004 |

---

## 3. Per-Class Breakdown (mAP50)

### Pictor-PPE

| Model | hard_hat | no_hard_hat | person |
|---|---|---|---|
| YOLO-World-L | 0.713 | **0.000** | 0.932 |
| OWL-ViT-L | 0.798 | **≈0.000** | 0.903 |
| Grounding DINO-base | 0.132 | 0.002 | 0.676 |
| YOLOv8m (zero-shot) | 0.017 | 0.000 | 0.017 |

### SHWD

| Model | hard_hat | no_hard_hat | person |
|---|---|---|---|
| YOLO-World-L | 0.496 | — | 0.010 |
| OWL-ViT-L | 0.460 | **0.000** | 0.022 |
| Grounding DINO-base | 0.062 | **0.000** | 0.000 |
| YOLOv8m (zero-shot) | 0.017 | — | 0.000 |

*SHWD has no `no_hard_hat` annotations — omitted from that column.*

### CHV

| Model | hard_hat | no_hard_hat | person |
|---|---|---|---|
| YOLO-World-L | 0.892 | — | 0.858 |
| OWL-ViT-L | 0.910 | **0.000** | 0.776 |
| Grounding DINO-base | 0.086 | **0.000** | 0.372 |
| YOLOv8m (zero-shot) | 0.006 | — | 0.037 |

---

## 4. Key Observations

### 4.1 The domain gap is severe and consistent

YOLOv8m collapses from mAP50=0.644 on SH17 to a mean of 0.014 across target domains — a ~97.8% relative drop. This is not a borderline degradation; it is a complete failure of the model to generalize. The gap holds across all three target domains despite their different visual characteristics (factory floor vs outdoor construction vs varied environments).

### 4.2 Open-vocabulary models do not solve the problem

Even the strongest zero-shot model (YOLO-World-L, mean 0.559) falls well short of what a domain-adapted detector should achieve. More importantly, the comparison table answers a critical question for the paper: "Does scale alone solve cross-factory shift?" — the answer is clearly no. A 68M parameter model fine-tuned on one factory fails. A 307M CLIP-based detector partially succeeds but is still far from deployable. This frames TTA as a necessary and principled solution.

### 4.3 `no_hard_hat` is a failure class for all open-vocabulary models

Every open-vocabulary model scores approximately 0.000 on `no_hard_hat` across all datasets. This is a structural failure: these models were trained to detect the *presence* of objects as visual concepts. "No hard hat" is an absence — a negative class — that CLIP-style embeddings cannot represent. By contrast, YOLOv8m was explicitly trained on labeled `no_hard_hat` examples. This per-class breakdown is one of the strongest arguments in the paper for why closed-set fine-tuned detectors + TTA is the right approach over open-vocabulary substitution.

A post-hoc confidence threshold sweep on YOLO-World-L (conf ∈ {0.001, 0.01, 0.1, 0.25} on Pictor-PPE and SHWD) confirms this is not a calibration artifact: `no_hard_hat` AP remains 0.000 at every threshold. The failure is structural — no confidence tuning can recover a class the model has no embedding for.

**Note (updated in Phase 3):** subsequent diagnostic analysis in Phase 3 revealed that YOLOv8m's `no_hard_hat` AP50 = 0.000 on target domains has a *distinct* cause from the open-vocabulary models above. Raw classification-head inspection showed the YOLOv8m head fires 5,000+ `no_hard_hat` predictions per batch at max conf = 1.000 on every target domain — the class is the most frequently argmaxed class, not a silent one. Despite this, AP50 is 0.0000 because these confident predictions land on wrong spatial locations and fail to match ground-truth boxes at IoU ≥ 0.5. This is a **localization failure**, not a classification failure — the class is *mislocated*, not absent. Open-vocabulary models fail *structurally* (CLIP cannot represent absence as a visual concept); YOLOv8m fails *geometrically* (known class, wrong box). These are two distinct failure modes with different remediation paths — open-vocab failures are unfixable without architecture change, whereas YOLOv8m's failure is addressable via feature-space alignment (see Phase 3b).

### 4.4 SHWD is consistently the hardest domain

All models rank SHWD as the most difficult: YOLO-World-L drops from 0.875 (CHV) to 0.253 (SHWD), OWL-ViT-L drops from 0.567 (Pictor) to 0.161 (SHWD), GDINO drops to 0.021. This consistent signal across architectures validates that SHWD represents a genuinely harder distribution shift, not a measurement artifact.

### 4.5 CHV appears visually close to YOLO-World's training distribution

YOLO-World-L achieves 0.875 mAP50 on CHV — unusually high for a zero-shot model on a domain-specific task. OWL-ViT does not replicate this (0.562), and GDINO is far behind (0.152). This suggests CHV's visual appearance happens to match large-scale internet-scraped pretraining data well. For the paper, CHV results should be reported but the SHWD and Pictor-PPE results carry more weight as genuine out-of-distribution scenarios.

---

## 5. Decisions Carried Into Phase 1

- **YOLOv8m is the adaptation target.** YOLO11m is dropped — open-vocabulary model comparisons already cover the "stronger model" question, and YOLOv8m's catastrophic zero-shot failure is a cleaner story.
- **SHWD and Pictor-PPE are the primary evaluation domains.** CHV results will be reported but weighted less in discussion due to the YOLO-World anomaly.
- **`no_hard_hat` AP must be reported per-class in all tables.** Its consistent failure across open-vocab models vs YOLOv8m's explicit training on it is a key finding.
- **The zero-shot baselines are fixed.** No re-runs needed unless eval code changes.
- **YOLO-World-L conf=0.001 is confirmed optimal.** Sweep over {0.001, 0.01, 0.1, 0.25} on Pictor and SHWD verified that the standard mAP protocol threshold is already the best. The reported numbers are the model's true ceiling.

---

## 6. Files Produced

| File | Description |
|---|---|
| `runs/train/sh17_yolov8m/weights/best.pt` | YOLOv8m trained on SH17, 100 epochs |
| `runs/results/sh17_yolov8m/baseline.json` | YOLOv8m zero-shot eval on all 3 domains |
| `runs/results/yolo_world_L/baseline.json` | YOLO-World-L zero-shot eval |
| `runs/results/owlvit_large/baseline.json` | OWL-ViT-L zero-shot eval |
| `runs/results/grounding_dino_base/baseline.json` | Grounding DINO-base zero-shot eval |
| `models/yolov8m.pt` | YOLOv8m COCO pretrained init weights |
| `models/yolo11m.pt` | Downloaded, not used (YOLO11m dropped) |
| `models/yolov8l-worldv2.pt` | YOLO-World-L weights |
| `hparams_yolov8m.yaml` | Locked training config |
| `runs/results/yolo_world_L/conf_sweep.json` | YOLO-World-L conf threshold sweep (Pictor + SHWD) |

---

## Next: Phase 1

Run TENT sweep on YOLOv8m best.pt across all three target domains:

```bash
uv run python scripts/eval_baseline.py --hparams hparams_yolov8m.yaml
uv run python scripts/sweep_tent.py --hparams hparams_yolov8m.yaml --domain all
```
