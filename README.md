# Cross-Factory Test-Time Adaptation (TTA) for PPE Detection

This repository implements **Test-Time Adaptation (TTA)** using **TENT** for object detection, applied to safety PPE detection across industrial environments.

The goal is to bridge the performance gap between a **source factory (SH17)** and a **target factory (Pictor-PPE, SHWD, CHV)** without retraining on labeled target data.

## Project Structure

```text
cross-factory-tta/
│
├── notebooks/
│   ├── kaggle/                    # self-contained experiment notebooks (run on Kaggle)
│   │   └── tta_experiment.ipynb
│   └── scratch/                   # local throwaway analysis (gitignored)
│
├── models/
│   ├── runs.md                    # log: checkpoint → Kaggle run → results
│   └── *.pt                       # downloaded model weights (gitignored)
│
├── data/                          # datasets (gitignored except metadata)
│   ├── sh17/
│   ├── pictor_ppe/
│   ├── shwd/
│   ├── CHV_dataset/
│   └── datasets.md
│
└── docs/                          # write-ups and visual examples
```

## Workflow

All training and evaluation runs on Kaggle with GPU. The notebook in `notebooks/kaggle/` is self-contained — it handles data prep, training, TTA, and evaluation end to end.

1. Open the notebook on Kaggle and attach the relevant dataset inputs
2. Run with **Save & Run All** (version history acts as the experiment log)
3. Download the output `.pt` checkpoint
4. Drop it in `models/` locally and log the run in `models/runs.md`

## Datasets

| Dataset | Role | Classes |
|---------|------|---------|
| SH17 | Source (train) | 17-class PPE |
| Pictor-PPE | Target (eval) | hard_hat / no_hard_hat / person |
| SHWD | Target (eval) | hard_hat / person |
| CHV | Target (eval) | hard_hat / no_hard_hat / person |

Canonical label mapping used across all target datasets:

```
0 → hard_hat
1 → no_hard_hat
2 → person
```

## Methodology

**YOLOv8m** trained on SH17 (source). At test time, **TENT** adapts BatchNorm statistics on the target domain without any labels, improving cross-factory detection performance.

---
**Author**: rainsfal1
