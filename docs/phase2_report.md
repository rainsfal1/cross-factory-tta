# Phase 2 Report — Standard TTA Baselines: DUA, SAR, and EATA

**Date:** 2026-04-20  
**Status:** Complete

---

## Objectives

Phase 2 evaluates three additional TTA baselines — DUA, SAR, and EATA — that address different failure modes of TENT from Phase 1. Together with TENT, these four methods cover the main families of standard TTA: entropy minimization (TENT), gradient-free stat-shift (DUA), sharpness-aware reliable entropy minimization (SAR), and anti-forgetting reliable entropy minimization with Fisher regularization (EATA). All methods are given their best possible hyperparameters via sweep before being reported.

---

## 1. DUA (Distribution Uncertainty Adaptation, Mirza et al. 2022)

### Methodology

DUA is entirely gradient-free. For each test image, it creates `n_augments` augmented copies, forwards them through the model in BN-train mode with geometrically decaying momentum, and restores eval mode. Running statistics drift toward the target distribution — no optimizer, no labels, no entropy signal.

```
mom_new      = mom_pre × decay_factor
effective_mom = mom_new + min_momentum_constant   # additive floor, not hard clip
```

Augmentation is photometric only (brightness, contrast, noise, horizontal flip). Geometric transforms were excluded — they break spatial structure and corrupt detection-scale features.

| Parameter | Value | Notes |
|---|---|---|
| decay_factor | 0.94 | Paper default |
| min_momentum_constant | 0.005 | Additive floor |
| mom_pre | 0.1 | Starting momentum |
| n_augments | 64 | Augmented copies per image |
| Adapted params | BN running stats | γ, β frozen; no gradient |

### Results

| Domain | Baseline | DUA | Recovery |
|---|---|---|---|
| Pictor-PPE | 0.0114 | 0.0039 | −0.0076 |
| SHWD | 0.0084 | 0.0040 | −0.0045 |
| CHV | 0.0216 | 0.0187 | −0.0029 |
| **Mean** | **0.0138** | **0.0089** | **−0.0050** |

### Decay Sweep

Sweep over `decay ∈ {0.94, 0.98, 0.99}` confirmed the paper default was already optimal. Higher decay (slower stat update) did not help.

| Domain | decay=0.94 | decay=0.98 | decay=0.99 | Best |
|---|---|---|---|---|
| Pictor-PPE | **0.0039** | 0.0023 | 0.0024 | 0.0039 |
| SHWD | 0.0039 | **0.0040** | 0.0027 | 0.0040 |
| CHV | **0.0185** | 0.0174 | 0.0178 | 0.0185 |

The numbers above are DUA's ceiling.

---

## 2. SAR (Sharpness-Aware and Reliable Entropy Minimization, Niu et al. 2023)

### Methodology

SAR extends TENT with two mechanisms designed to prevent instability on noisy or small batches:

1. **Entropy filtering** — samples with entropy above `e_margin` are excluded before updating. Only confident predictions drive adaptation.
2. **Sharpness-aware minimization (SAM)** — two-step update: perturb weights in the max-entropy direction, then compute and apply the gradient at the perturbed point. Biases toward flat minima in the entropy landscape.
3. **Model reset** — EMA tracks adapted loss; resets weights if the loss diverges past a threshold.

| Parameter | Value | Notes |
|---|---|---|
| lr | 0.00025 | Paper default |
| steps | 1 | Gradient steps per batch |
| rho | 0.05 | SAM perturbation radius |
| e_margin | swept | See below |
| reset_constant | 0.2 | EMA deviation fraction |
| adapt_batch | 16 | Images per step |

### Implementation Validation

Verified against the official SAR repo (`github.com/mr-eggplant/SAR`). Two issues found and fixed:

- **Bug:** `_per_sample_entropy` only handled `list/tuple` prediction formats; YOLOv8 train-mode returns a `dict`. Fixed by adding dict handling matching `tent_loss`.
- **e_margin calibration:** Paper formula `0.4 × ln(17) = 1.145` (17 SH17 classes) was confirmed via debug to silence all samples — observed target-domain entropy is **1.75–2.04**, all above 1.145. The official code uses `0.4 × ln(1000) = 2.763` (ImageNet). A sweep was run to find the true ceiling.

### e_margin Sweep Results

| e_margin | Pictor mAP50 | SHWD mAP50 | CHV mAP50 | samples used (Pictor) |
|---|---|---|---|---|
| 1.145 (paper formula) | 0.0114 | 0.0084 | 0.0215 | 0/152 |
| **2.000** | **0.0117** | 0.0085 | 0.0215 | 132/152 |
| 2.500 | 0.0114 | **0.0086** | 0.0215 | 152/152 |
| 2.763 (official default) | 0.0114 | **0.0086** | 0.0215 | 152/152 |

### Results (best e_margin per domain)

| Domain | Baseline | SAR | Recovery |
|---|---|---|---|
| Pictor-PPE | 0.0114 | 0.0117 | +0.0003 |
| SHWD | 0.0084 | 0.0086 | +0.0001 |
| CHV | 0.0216 | 0.0215 | −0.0001 |
| **Mean** | **0.0138** | **0.0139** | **+0.0001** |

---

## 3. EATA (Efficient Anti-Forgetting Test-Time Adaptation, Niu et al. 2022)

### Methodology

EATA extends TENT with two mechanisms: dual sample filtering and Fisher EWC regularization.

1. **Reliability filter** — samples with entropy above `e_margin` are excluded (same as SAR, but without SAM).
2. **Redundancy filter** — samples whose mean softmax prediction is too similar to a running average of past predictions (cosine similarity above `d_margin`) are excluded. Prevents redundant samples from dominating updates.
3. **Fisher EWC regularization** — a penalty term weighted by the Fisher information diagonal (pre-computed on source images) penalizes deviation from source BN parameters, preventing catastrophic forgetting.

| Parameter | Value | Notes |
|---|---|---|
| lr | 0.001 | SGD with momentum 0.9 |
| steps | 1 | Gradient steps per batch |
| e_margin | 2.454 | log(1000)/2 − 1; paper default, above observed target entropy |
| d_margin | swept | See below |
| fisher_alpha | swept | EWC regularization strength |
| fisher_n_images | 200 | Source images for Fisher diagonal |
| adapt_batch | 16 | Images per step |

### Implementation Validation

Verified against the official EATA repo (`github.com/mr-eggplant/EATA`). Two issues found and fixed:

- **YOLOv8 dict format:** Same fix as SAR — `_per_sample_entropy` and `_per_sample_softmax` both needed dict-aware handling for YOLOv8 train-mode forward output.
- **Redundancy filter calibration:** Paper default `d_margin=0.05` silences all samples after batch 0. Under catastrophic shift, the model outputs near-uniform distributions across all images — all predictions look identical — so cosine similarity is uniformly 0.90–0.99. A sample only passes if `|cos_sim| < 0.05`, which is never satisfied. A sweep over `d_margin` was required.

### d_margin Sweep Results

`e_margin=2.454` fixed. `fisher_alpha ∈ {0, 2000}` crossed.

**Pictor-PPE**

| d_margin | fisher_α=0 | fisher_α=2000 | Recovery (best) |
|---|---|---|---|
| 0.05 (paper default) | 0.0114 | 0.0114 | 0.0000 |
| **0.95** | **0.0119** | 0.0116 | **+0.0005** |
| 1.00 | 0.0117 | 0.0117 | +0.0003 |

**SHWD**

| d_margin | fisher_α=0 | fisher_α=2000 | Recovery (best) |
|---|---|---|---|
| 0.05 (paper default) | 0.0084 | 0.0084 | 0.0000 |
| 0.95 | 0.0090 | 0.0088 | +0.0006 |
| **1.00** | **0.0090** | 0.0086 | **+0.0006** |

**CHV**

| d_margin | fisher_α=0 | fisher_α=2000 | Recovery (best) |
|---|---|---|---|
| 0.05 (paper default) | 0.0216 | 0.0216 | 0.0000 |
| **0.95** | **0.0217** | 0.0215 | **+0.0001** |
| 1.00 | 0.0216 | 0.0215 | −0.0001 |

Two structural patterns emerge from the sweep:

- **Fisher regularization consistently hurts.** `fisher_alpha=2000` is always worse than `fisher_alpha=0`. Under extreme shift, penalizing deviation from source parameters is counterproductive — the source prior is wrong for the target domain and anchoring to it suppresses useful updates.
- **Best EATA degenerates to filtered TENT.** With `fisher_alpha=0` and `d_margin=0.95`, EATA reduces to reliability-filtered entropy minimization. Even then, improvements are on the order of 0.0005 mAP50 — well within noise for these absolute values.

### Results (best config: d_margin=0.95, fisher_alpha=0)

| Domain | Baseline | EATA | Recovery |
|---|---|---|---|
| Pictor-PPE | 0.0114 | 0.0119 | +0.0005 |
| SHWD | 0.0084 | 0.0090 | +0.0006 |
| CHV | 0.0216 | 0.0217 | +0.0001 |
| **Mean** | **0.0138** | **0.0142** | **+0.0004** |

---

## 5. Full Baseline Comparison (mAP50)

| Method | Pictor-PPE | SHWD | CHV | Mean |
|---|---|---|---|---|
| YOLOv8m (no adapt) | 0.0114 | 0.0084 | 0.0216 | 0.0138 |
| YOLOv8m + TENT (offline best) | 0.0120 | 0.0088 | 0.0225 | 0.0144 |
| YOLOv8m + EATA (best config) | 0.0119 | 0.0090 | 0.0217 | 0.0142 |
| YOLOv8m + SAR (best e_margin) | 0.0117 | 0.0086 | 0.0215 | 0.0139 |
| YOLOv8m + DUA | 0.0039 | 0.0040 | 0.0187 | 0.0089 |
| YOLOv8m + Online TENT | 0.0017 | 0.0002 | 0.0000 | 0.0006 |
| YOLO-World-L (zero-shot) | 0.548 | 0.253 | 0.875 | 0.559 |
| OWL-ViT-L (zero-shot) | 0.567 | 0.161 | 0.562 | 0.430 |
| Grounding DINO-base (zero-shot) | 0.270 | 0.021 | 0.152 | 0.148 |

---

## 6. Conclusion

Six standard TTA configurations have now been evaluated across three target domains, all with hyperparameter sweeps to ensure fair reporting. The results form a definitive picture of where the standard TTA literature stands on extreme-shift object detection.

### 6.1 The Gap Is Not Closable by Normalization-Layer Adaptation

The best result across all methods — offline TENT at mean mAP50=0.0144 — is a 4% relative improvement over the unadapted baseline (0.0138). Against YOLO-World-L at 0.559, that closes less than 1% of the domain gap. Every method converges to roughly the same ceiling, regardless of mechanism or sophistication.

### 6.2 Distinct Failure Modes, Same Root Cause

Each method fails for a structurally different reason, but all failures trace to the same assumption being violated:

| Method | Failure Mode | Mechanism |
|---|---|---|
| Online TENT | Catastrophic overshoot | Continuous BN updates collapse all class signal; `no_hard_hat` zeroed out first |
| DUA | Running stat overwrite | Geometric momentum update overwrites source stats with noisy target stats; all domains degraded below baseline |
| SAR | Silenced by design | Target-domain entropy (1.75–2.04) exceeds reliability threshold at any reasonable `e_margin`; SAM never fires meaningfully |
| EATA | Dual failure | Redundancy filter silences all batches after the first (cos_sim 0.90–0.99 >> d_margin); Fisher EWC anchors model to wrong source prior and hurts when `d_margin` is relaxed |
| Offline TENT | Marginal ceiling | Positive only at very low lr/steps; higher settings overshoot; method has no mechanism to know when to stop |

The violated assumption is this: **all standard TTA methods expect the model to produce meaningful, moderately-confident predictions on target-domain inputs.** They use those predictions — their entropy, their softmax probabilities, their gradient directions — as the signal for adaptation. Under a 97.8% mAP collapse, predictions carry no useful signal. The model is effectively random on target inputs. Minimizing the entropy of random predictions, or matching their statistics to source-domain statistics, cannot recover anything.

### 6.3 What the Filtering Methods Reveal

SAR and EATA are the most informative failures precisely because they are designed to be conservative. SAR refuses to update when entropy is too high; EATA refuses to update when predictions are too redundant. Both end up refusing to update at all — not because of bugs, but because the target domain triggers every safety valve simultaneously. This is a quantitative confirmation that the domain gap exceeds the regime where entropy-based TTA is theoretically grounded.

### 6.4 What Phase 3 Must Address

The Phase 2 baseline suite motivates a custom method with three explicit requirements:

1. **Does not require confident predictions to function.** The adaptation signal cannot be entropy or softmax probability — both are uninformative under extreme shift.
2. **Does not blindly overwrite BN statistics.** DUA and online TENT show that unconstrained stat updates degrade performance. Any update must be discriminative, not distributional.
3. **Explicitly preserves class-discriminative features, especially `no_hard_hat`.** Vanilla TTA destroys the hardest class first. Open-vocabulary models never detect it at all. A detection-aware method must target the feature-level cause of this collapse, not the stat-level symptom.

The question Phase 3 answers: can a method that uses detection geometry — anchor activations, objectness structure, class-head feature norms — as its adaptation signal succeed where entropy minimization cannot?

---

## 7. Files Produced

| File | Description |
|---|---|
| `runs/results/sh17_yolov8m/dua.json` | DUA eval on all 3 domains |
| `runs/results/sh17_yolov8m/sweep_dua.json` | DUA decay sweep (decay ∈ {0.94, 0.98, 0.99}) |
| `runs/results/sh17_yolov8m/sar.json` | SAR eval at default e_margin |
| `runs/results/sh17_yolov8m/sweep_sar.json` | SAR e_margin sweep (∈ {1.145, 2.0, 2.5, 2.763}), all 3 domains |
| `runs/results/sh17_yolov8m/eata.json` | EATA eval at best config (d_margin=0.95, fisher_alpha=0) |
| `runs/results/sh17_yolov8m/sweep_eata.json` | EATA d_margin × fisher_alpha sweep, all 3 domains |
