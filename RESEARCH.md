# Cross-Factory PPE Detection — Research Summary

## What We Are Building

A test-time adaptation (TTA) framework for PPE (Personal Protective Equipment) detection across industrial environments. The core problem: a detector trained in one factory fails when deployed in a different factory due to domain shift — different lighting, camera angles, worker demographics, equipment styles.

---

## Current Setup

### Model
- **Architecture**: YOLOv8m (25.8M params, 78.7 GFLOPs)
- **Training domain**: SH17 — 17-class safety helmet dataset, 6,479 train / 1,620 val images
- **Hardware**: Tesla V100 32GB, 128-core CPU, 377GB RAM
- **Training config**: 50 epochs, batch 64, imgsz 640, SGD lr=0.02, AMP enabled

### Evaluation Domains (Target — cross-factory)
| Dataset | Images | Notes |
|---|---|---|
| Pictor-PPE | 152 test | Chinese factory images, strong visual shift |
| SHWD | 1,517 test | Mixed construction/industrial |
| CHV | 133 test | Construction helmet, varied environments |

### Label Space
SH17 has 17 classes. Target domains only annotate 3 canonical classes:
- `hard_hat` → SH17 index 4
- `no_hard_hat` → SH17 index 9
- `person` → SH17 index 16

Evaluation remaps target labels into SH17 index space to allow fair comparison.

### TTA Method Implemented: TENT
- Adapts only BatchNorm affine parameters (gamma/beta) at test time
- Minimizes prediction entropy on incoming batches — no ground truth needed
- Current config: `steps=1, lr=0.001`
- This is intentionally minimal — serves as the lower bound for TTA methods

### Evaluation Pipeline
```
eval_baseline.py
  1. Source domain verification  — SH17 val (expected mAP50 >= 0.70)
  2. Zero-shot cross-domain      — Pictor, SHWD, CHV (expected mAP50 drop)

eval_tent.py
  1. Same baseline on Pictor
  2. TENT adaptation on Pictor
  3. Recovery delta (TENT - baseline)
```

---

## Expected Results

### Source Domain (SH17 val)
- mAP50: **~0.75–0.88** — model trained here, should be strong
- Anything below 0.70 indicates a data or weight issue

### Cross-Domain Baseline (zero-shot)
- Pictor-PPE: **~0.25–0.45** — largest expected gap, most visually different
- SHWD: **~0.35–0.55** — moderate gap
- CHV: **~0.30–0.50** — moderate gap

### TENT Adaptation (Pictor-PPE)
- Expected recovery: **+0.02 to +0.06 mAP50**
- With steps=1 and 152 images the adaptation signal is weak — modest recovery is the realistic outcome
- Documenting that TENT partially but not fully closes the gap *is* the finding — it motivates stronger methods

---

## Why This Is Not Yet Publishable

The current experiment is a baseline section, not a paper. It answers "does TENT help a little?" — yes, probably, marginally. That's not a contribution.

---

## Directions Toward Publication

### Direction 1 — Benchmark Contribution
**What**: Frame the unified multi-dataset PPE eval framework as the contribution itself. No existing work cleanly benchmarks cross-factory PPE TTA across SH17 + Pictor + SHWD + CHV with a common label space.

**What to add**:
- 1-2 more datasets (e.g. HARD Hat Workers Dataset from Roboflow, PISC)
- Run 4-5 TTA methods on the same framework (TENT, DUA, CoTTA, LAME, TTT)
- Report per-class results, not just mAP
- Include a t-SNE feature visualization showing domain gaps

**Venue**: IEEE Access, WACV, or an ECCV/ICCV workshop on industrial vision

**Effort**: Medium. Infrastructure is almost done. Mostly running experiments and writing.

---

### Direction 2 — Novel Method: Class-Imbalance Aware TTA
**What**: TENT's entropy minimization collapses on dominant classes (person >> hard_hat >> no_hard_hat). The model becomes confidently wrong about minority classes. Propose a reweighted entropy loss that upweights underrepresented classes during adaptation.

**Core idea**:
```
standard TENT:    L = mean entropy over all predictions
proposed:         L = sum( w_c * entropy_c )
                  where w_c = 1 / (frequency of class c in predictions)
```

**What to add**:
- Implement the reweighted loss variant
- Ablation: standard entropy vs. reweighted vs. class-conditional entropy
- Compare against TENT, DUA, CoTTA on all three target domains
- Show per-class AP improvement, especially for `no_hard_hat` (rare, safety-critical)

**Venue**: IEEE Transactions on Industrial Informatics, CVPR workshop

**Effort**: Medium-high. Novel enough to stand alone.

---

### Direction 3 — Continual Cross-Factory Adaptation
**What**: Factories don't shift once — they shift continuously (seasons, new equipment, personnel changes, line reconfigurations). Frame as an online/continual TTA problem where the model adapts to a sequence of domains without forgetting earlier ones.

**Core idea**: Simulate sequential domain shift: train on SH17, then adapt to Pictor → SHWD → CHV in sequence. Measure forward transfer and catastrophic forgetting. Compare methods that handle continual shift (CoTTA, ETA) vs. ones that don't (vanilla TENT).

**What to add**:
- Sequential eval loop across domains
- Forgetting metric: re-eval on earlier domains after adapting to later ones
- Memory replay or EWC-style regularization to prevent forgetting

**Venue**: IEEE Transactions on Neural Networks and Learning Systems, ICCV/ECCV

**Effort**: High. Most novel, most work, strongest venue.

---

### Direction 4 — Source-Free Domain Adaptation Comparison
**What**: Compare TTA (test-time, no source data needed at inference) against source-free DA methods (SHOT, NRC, AaD) that still use source data during adaptation. Quantify the accuracy-privacy tradeoff — in real deployments, sharing source factory data may be restricted for IP reasons.

**Framing**: "How much performance do you lose by not sharing source data?" — directly relevant to industrial deployment.

**Venue**: IEEE Transactions on Industrial Electronics, BMVC

**Effort**: Medium. Mostly running existing methods, framing is the contribution.

---

## Recommended Path

1. **Finish current experiments** — get baseline + TENT numbers on all three datasets
2. **Add 2-3 more TTA methods** (DUA and CoTTA are highest priority — both have clean implementations)
3. **Decide on novel angle** — Direction 2 (class-imbalance) is the most self-contained and achievable; Direction 1 (benchmark) is lowest risk
4. **t-SNE visualization** — run once training is done, shows domain gap clearly in figures
5. **Write** — with clean numbers across 4+ methods and 3 datasets, the paper writes itself

---

## Repo Structure (current)

```
cross-factory-tta/
  config.py                  — paths, class mappings, SH17 index remapping
  hparams.yaml               — all hyperparameters in one place
  scripts/
    prepare_data.py          — sets up data/prepared/ from raw datasets
    train.py                 — trains YOLOv8m on SH17
    eval_baseline.py         — source domain verification + zero-shot cross-domain eval
    eval_tent.py             — baseline vs TENT on Pictor-PPE
  src/
    data_setup.py            — dataset preparation logic
    eval.py                  — eval functions (source domain, baseline, TENT)
    tent.py                  — TENT implementation (BN adaptation + entropy loss)
    train.py                 — training logic with resume support
    hparams.py               — hparams loader
  data/ -> /mnt/data/...     — symlink to mounted data disk (raw + prepared)
  runs/                      — checkpoints and eval outputs
```
