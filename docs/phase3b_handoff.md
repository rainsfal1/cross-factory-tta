# Phase 3b Handoff — Prototype Alignment (paused)

**Last updated:** 2026-04-20
**Status:** Design locked, implementation not started.

---

## Where we left off

Phase 3a is complete. **DATTA-S2 (conf_threshold=0.15, lr=0.01, steps=5)** beats every Phase 2 baseline on every target domain:

| Metric | Pictor | SHWD | CHV | Mean |
|---|---|---|---|---|
| DATTA-S2 @ 0.15 | 0.0126 | 0.0090 | 0.0227 | **0.0148** |
| Prior best (TENT) | 0.0120 | 0.0088 | 0.0225 | 0.0144 |

Secondary goal (non-zero `no_hard_hat` AP50) is **not yet met** — still 0.0000 on SHWD/CHV, 0.0002 on Pictor. This is what 3b is designed to fix.

## The decisive diagnostic (why 3b is worth building)

Full output lives in `runs/results/sh17_yolov8m/datta_s2.json` run logs. Summary:

| Domain | `no_hard_hat` argmax count | max conf | mean conf | final AP50 |
|---|---|---|---|---|
| Pictor-PPE | 5,222 / 134,400 | 1.000 | 0.423 | 0.0002 |
| SHWD | 5,137 / 134,400 | 1.000 | 0.399 | 0.0000 |
| CHV | 5,373 / 134,400 | 1.000 | 0.417 | 0.0000 |

The class head fires `no_hard_hat` more than any other class at full confidence — but none of the firings land on GT boxes at IoU ≥ 0.5. **This is a spatial/geometric failure, not a classification failure.** Prototype alignment pulls target-domain `no_hard_hat` features toward their source-domain prototype, which is expected to correct the geometry.

## What was built in 3a (do not rebuild)

| File | Purpose |
|---|---|
| `src/datta.py` | DATTA class — Stage 1 (soft BN warm-up) + Stage 2 (detection-aware entropy min). Already has `_debug` hook with per-class argmax diagnostic. |
| `scripts/eval_datta.py` | Single-config eval. Supports `--s1-only`, `--s2-only`, `--alpha`, `--conf-threshold`, `--debug`. |
| `scripts/sweep_datta.py` | Full ablation sweep: S1 / S2 / full. |
| `hparams_yolov8m.yaml` | `datta:` section — currently lr=0.01, steps=5, alpha=0.9, conf_threshold=0.05. **Note:** conf_threshold=0.15 is the actual winning value but wasn't promoted to default. |

## Documents updated in 3a

- `docs/phase0_report.md` §4.3 — added "Note (updated in Phase 3)" distinguishing open-vocab structural failure from YOLOv8m geometric failure.
- `docs/phase3_report.md` — full Phase 3a writeup, and §5 motivation for 3b reframed from "resurrect dead class" to "correct spatial misalignment of active class."

---

## What to build in 3b (detailed plan)

### Decisions already locked
- **Pooling strategy: Option A** — use Ultralytics anchor-grid cells directly; each anchor already has an associated feature vector at its grid cell. No RoI interpolation. Fast, matches existing loss code structure.
- **Forward hook target:** `model.model.model[-1]` (the `Detect` module at `model.22`).
- **Prototype scope:** per-class mean feature vector, source = SH17 val (200 images should suffice for initial test, can scale up).

### Architecture facts already verified

```
Detect module: model.22
  nc=17, reg_max=16, nl=3, no=81, stride=[8, 16, 32]

Forward-hook input shapes (3-level PAFPN pyramid):
  level 0: (B, 192, 80, 80)     —  6,400 anchors, stride 8
  level 1: (B, 384, 40, 40)     —  1,600 anchors, stride 16
  level 2: (B, 576, 20, 20)     —    400 anchors, stride 32
  total: 8,400 anchors per image (matches the diagnostic)
```

**Feature dimensions are different per level (192/384/576).** Two sensible options:
1. Keep one prototype *per level* → 3 prototypes × 3 classes = 9 vectors (simplest, no projection needed)
2. Project all levels to a common dim via 1×1 conv → 1 prototype × 3 classes = 3 vectors (cleaner but adds a learnable projection we'd have to freeze)

**Go with (1)** — per-level prototypes, 3 vectors per class per level. No projection. Cosine distances computed per level and averaged.

### Implementation steps

**Step 1 — `src/prototypes.py`**

```python
def extract_prototypes(model, source_img_dir, device, imgsz=640, n_images=200):
    """
    Returns: dict {class_idx: {level: feature_vector}}
      where class_idx ∈ {4, 9, 16} (SH17 indices for hard_hat, no_hard_hat, person)
    Pipeline:
      1. Register forward_hook on Detect module, capture 3-level feature pyramid.
      2. For each image with GT boxes:
         - Letterbox to imgsz
         - Read YOLO-format labels
         - For each GT box: assign to the level whose stride matches the box size
           (small→level 0, medium→level 1, large→level 2 — use sqrt(w*h) thresholds)
         - Bilinear-sample the feature map at the grid cell containing the box center
           → feature vector for that GT instance
      3. Accumulate per-class, per-level mean.
    """
```

Rough level assignment: area thresholds 64² (level 0), 256² (level 1), else level 2. Or simpler: assign to the level where `box_center / stride` lies within the grid and the box fits in ≥1 stride unit.

**Step 2 — `scripts/extract_prototypes.py`**

One-shot runner. Saves `runs/results/sh17_yolov8m/prototypes.pt` with structure:
```python
{
  4: {0: tensor[192], 1: tensor[384], 2: tensor[576]},   # hard_hat
  9: {0: tensor[192], 1: tensor[384], 2: tensor[576]},   # no_hard_hat
  16: {0: tensor[192], 1: tensor[384], 2: tensor[576]},  # person
}
```
CLI:
```bash
uv run python scripts/extract_prototypes.py --hparams hparams_yolov8m.yaml --n-images 200
```

**Step 3 — extend `src/datta.py`**

Add to `DATTA.__init__`:
```python
prototypes: dict | None = None,      # loaded via torch.load
lambda_proto: float = 0.5,           # prototype loss weight
```

Register forward hook on `Detect` to capture `feat_list` (3-level pyramid) on every forward pass.

Add helper `_prototype_loss(feats, preds, mask)`:
- For each level, get its anchor positions for anchors in `mask` with argmax class ∈ {4, 9, 16}
- Gather the grid-cell feature vector for each such anchor
- Cosine distance to the prototype of its predicted class at that level
- Return mean

Modify `adapt()`:
```python
loss_entropy = detection_aware_loss(...)
loss_proto = prototype_loss(...)  # if prototypes loaded
loss = loss_entropy + lambda_proto * loss_proto
```

**Step 4 — extend `scripts/eval_datta.py`**

- `--use-prototypes` flag (loads `prototypes.pt` automatically from `runs/results/<run_name>/`)
- `--lambda-proto FLOAT`
- `--proto-only` (disables entropy loss, runs only prototype alignment — ablation row)

**Step 5 — extend `scripts/sweep_datta.py`**

Add λ ∈ {0.1, 0.5, 1.0} as a sweep axis when `--use-prototypes` is set. Keep conf_threshold fixed at 0.15 for the sweep to isolate the prototype contribution.

### Target ablation table for the paper

| Variant | Entropy loss | Prototype align | Expected no_hh AP50 |
|---|---|---|---|
| Baseline | ✗ | ✗ | 0.0000 |
| DATTA-S2 (current best) | ✓ | ✗ | 0.0000 |
| DATTA-Proto | ✗ | ✓ | >0 if method works |
| DATTA-Full | ✓ | ✓ | highest expected |

### Success criteria for 3b

- **Hard success:** DATTA-Full achieves `no_hard_hat` AP50 ≥ 0.002 on at least one of SHWD/Pictor, and maintains the mAP50 gains from DATTA-S2.
- **Partial success:** DATTA-Full matches DATTA-S2 on mean mAP50 but lifts `no_hard_hat` measurably (even to 0.0005) — proves the mechanism works, paper has a qualitative finding.
- **Failure:** `no_hard_hat` stays 0.0000 — we accept DATTA-S2 as the final method and frame the paper around the mAP win + the diagnostic reframe (localization failure, not classification failure) as the key insight.

---

## Starting the work later

Pick up with:

```
Read docs/phase3b_handoff.md, then build step 1 (src/prototypes.py).
Option A (anchor-grid pooling) is locked. Per-level prototypes, not projected.
```

The Ultralytics probe command in §"Architecture facts" confirms the feature shapes — no need to re-probe.
