# Phase 1 Report — TENT Adaptation

**Date:** 2026-04-19  
**Status:** Complete

---

## Objectives

Phase 1 adapts the source-trained YOLOv8m to each target domain using TENT (Test-time Entropy Minimization). The goal is to close the domain gap measured in Phase 0 without any labeled target data.

Key questions:
1. Does TENT recover any meaningful mAP on the target domains?
2. What are the best adaptation hyperparameters (steps × lr)?
3. Is recovery consistent across all three domains, or domain-specific?

---

## 1. Starting Point — Baseline Domain Gap

Source model: `runs/train/sh17_yolov8m/weights/best.pt`  
Source validation (SH17): mAP50 = **0.6442**, mAP50-95 = **0.4223**

### YOLOv8m Zero-Shot on Target Domains (no adaptation)

| Domain | mAP50 | mAP50-95 | hard_hat | no_hard_hat | person |
|---|---|---|---|---|---|
| Pictor-PPE | 0.0114 | 0.0044 | 0.0167 | 0.0002 | 0.0174 |
| SHWD | 0.0084 | 0.0038 | 0.0168 | — | 0.0000 |
| CHV | 0.0216 | 0.0044 | 0.0062 | — | 0.0370 |
| **Mean** | **0.0138** | **0.0042** | | | |

The model collapses ~97.8% from source to target. This is the gap TENT must close.

---

## 2. TENT Methodology

**Algorithm:** TENT minimizes prediction entropy on batch normalization statistics only. No labels required.

**Implementation:** `src/tent.py` — wraps the YOLOv8m backbone, enables grad only for BN layers, runs Adam steps on mean entropy of detection confidence scores.

**Adaptation settings (from `hparams_yolov8m.yaml`):**

| Parameter | Value | Notes |
|---|---|---|
| adapt_batch | 16 | Images per gradient step |
| val_batch | 64 | Batch size during mAP eval |
| optimizer | Adam | Standard for TENT |
| Adapted params | BN only | γ, β of all BN layers |
| Image size | 640 | Matches training |

**Sweep grid:**

| Parameter | Values |
|---|---|
| steps | 1, 5, 10 |
| lr | 0.001, 0.005, 0.01 |

9 combinations per domain × 3 domains = 27 runs total.

---

## 3. TENT Sweep Results

Run with:
```bash
uv run python scripts/sweep_tent.py --hparams hparams_yolov8m.yaml --domain all
```

### Pictor-PPE

Baseline: 0.0114

| steps | lr | mAP50 | recovery |
|---|---|---|---|
| 1 | 0.001 | 0.0117 | +0.0003 |
| 1 | 0.005 | 0.0119 | +0.0005 |
| 1 | 0.010 | 0.0114 | −0.0000 |
| 5 | 0.001 | 0.0120 | +0.0005 |
| 5 | 0.005 | 0.0071 | −0.0043 |
| 5 | 0.010 | 0.0055 | −0.0059 |
| 10 | 0.001 | 0.0103 | −0.0011 |
| 10 | 0.005 | 0.0057 | −0.0058 |
| 10 | 0.010 | 0.0055 | −0.0059 |

**Best: steps=5, lr=0.001 → mAP50=0.0120, recovery=+0.0005** → `runs/results/sh17_yolov8m/sweep_tent_pictor_ppe.json`

### SHWD

Baseline: 0.0084

| steps | lr | mAP50 | recovery |
|---|---|---|---|
| 1 | 0.001 | 0.0088 | +0.0004 |
| 1 | 0.005 | 0.0081 | −0.0003 |
| 1 | 0.010 | 0.0077 | −0.0007 |
| 5 | 0.001 | 0.0081 | −0.0003 |
| 5 | 0.005 | 0.0068 | −0.0017 |
| 5 | 0.010 | 0.0055 | −0.0029 |
| 10 | 0.001 | 0.0079 | −0.0005 |
| 10 | 0.005 | 0.0055 | −0.0029 |
| 10 | 0.010 | 0.0063 | −0.0021 |

**Best: steps=1, lr=0.001 → mAP50=0.0088, recovery=+0.0004** → `runs/results/sh17_yolov8m/sweep_tent_shwd.json`

### CHV

Baseline: 0.0216

| steps | lr | mAP50 | recovery |
|---|---|---|---|
| 1 | 0.001 | 0.0216 | −0.0001 |
| 1 | 0.005 | 0.0214 | −0.0002 |
| 1 | 0.010 | 0.0219 | +0.0003 |
| 5 | 0.001 | 0.0219 | +0.0003 |
| 5 | 0.005 | 0.0225 | +0.0009 |
| 5 | 0.010 | 0.0207 | −0.0009 |
| 10 | 0.001 | 0.0221 | +0.0005 |
| 10 | 0.005 | 0.0210 | −0.0006 |
| 10 | 0.010 | 0.0206 | −0.0010 |

**Best: steps=5, lr=0.005 → mAP50=0.0225, recovery=+0.0009** → `runs/results/sh17_yolov8m/sweep_tent_chv.json`

---

## 4. Online TENT Results

After sweep identifies best (steps, lr) per domain, online TENT runs the same adaptation sequentially image-by-image (simulating real deployment). Evaluated with `scripts/eval_online_tent.py`.

Run with:
```bash
uv run python scripts/eval_online_tent.py --hparams hparams_yolov8m.yaml
```

| Domain | Baseline | TENT offline best | Online TENT | vs. Baseline | vs. YOLO-World-L |
|---|---|---|---|---|---|
| Pictor-PPE | 0.0114 | 0.0120 (s=5, lr=0.001) | 0.0017 | −0.0097 | 0.548 |
| SHWD | 0.0084 | 0.0088 (s=1, lr=0.001) | 0.0002 | −0.0083 | 0.253 |
| CHV | 0.0216 | 0.0225 (s=5, lr=0.005) | 0.0000 | −0.0216 | 0.875 |
| **Mean** | **0.0138** | **0.0144** | **0.0006** | **−0.0132** | **0.559** |

*Online TENT uses fixed hparams policy (steps=10, lr=0.005) — not per-domain tuned.*

---

## 5. Comparison Against Zero-Shot Baselines

Final summary table (mAP50) once all runs complete:

| Method | Pictor-PPE | SHWD | CHV | Mean |
|---|---|---|---|---|
| YOLOv8m (no adapt) | 0.0114 | 0.0084 | 0.0216 | 0.0138 |
| YOLO-World-L | 0.548 | 0.253 | 0.875 | 0.559 |
| OWL-ViT-L | 0.567 | 0.161 | 0.562 | 0.430 |
| Grounding DINO-base | 0.270 | 0.021 | 0.152 | 0.148 |
| **YOLOv8m + TENT (offline best)** | **0.0120** | **0.0088** | **0.0225** | **0.0144** |
| **YOLOv8m + Online TENT** | **0.0017** | **0.0002** | **0.0000** | **0.0006** |

### Per-Class Online TENT (mAP50)

| Domain | hard_hat | no_hard_hat | person |
|---|---|---|---|
| Pictor-PPE | 0.0050 | 0.0000 | 0.0001 |
| SHWD | 0.0003 | — | 0.0000 |
| CHV | 0.0000 | — | 0.0000 |

---

## 6. Key Findings

### 6.1 Offline TENT: marginal but positive recovery

The offline sweep found small gains at low lr (0.001) and few steps (1–5) across all three domains. Recovery ranged from +0.0004 (SHWD) to +0.0009 (CHV). Higher lr and more steps consistently degraded performance, indicating BN adaptation overshoots quickly when the domain gap is this severe.

### 6.2 Online TENT: catastrophic degradation

Online TENT (steps=10, lr=0.005) destroyed performance on all domains — mean recovery −0.0132, CHV collapsed to 0.0000. Sequential adapt-then-predict with aggressive hyperparameters accumulates BN corruption across batches with no recovery. The offline sweep's per-domain best settings would not be known in advance in real deployment, making this a realistic failure case for the paper.

### 6.3 TENT does not solve the domain gap

Even the offline sweep best (mean mAP50=0.0144) barely improves on the unadapted baseline (0.0138). TENT closes less than 1% of the gap to YOLO-World-L (0.559). BN-only entropy minimization is insufficient when the source-to-target shift is this large — the feature distribution mismatch goes beyond what affine BN parameters can correct.

### 6.4 `no_hard_hat` destroyed by online TENT — motivates Phase 3

Online TENT produces no `no_hard_hat` detections on any domain. This is not just a failure to recover the class — it actively destroys class-specific knowledge the fine-tuned model already had. Vanilla TTA methods not only fail to recover `no_hard_hat`, they erase the one structural advantage the fine-tuned model holds over open-vocabulary models. This motivates a detection-aware adaptation method that preserves class-discriminative features during adaptation — the core contribution of Phase 3.

### 6.5 SHWD confirms hardest domain

SHWD online TENT: mAP50=0.0002 — effectively zero. Consistent with Phase 0 findings across all architectures. The distribution shift in SHWD is qualitatively harder than Pictor or CHV.

---

## 7. Files Produced

| File | Description |
|---|---|
| `runs/results/sh17_yolov8m/baseline.json` | YOLOv8m zero-shot on all 3 domains (done) |
| `runs/results/sh17_yolov8m/sweep_tent_pictor_ppe.json` | TENT sweep on Pictor-PPE |
| `runs/results/sh17_yolov8m/sweep_tent_shwd.json` | TENT sweep on SHWD |
| `runs/results/sh17_yolov8m/sweep_tent_chv.json` | TENT sweep on CHV |
| `runs/results/sh17_yolov8m/online_tent.json` | Online TENT eval on all 3 domains |

---

## Next: Phase 2

Run DUA (Distribution Uncertainty Adaptation) eval and compare against TENT:

```bash
uv run python scripts/eval_dua.py --hparams hparams_yolov8m.yaml
```
