# Cross-Factory Test-Time Adaptation (TTA) for PPE Detection

This repository implements a robust approach for **Test-Time Adaptation (TTA)** using **TENT** for object detection, specifically adapted for safety PPE detection in varying industrial environments. 

The goal is to bridge the performance gap between a **Source Factory (SH17 dataset)** and a **Target Factory (Pictor-PPE dataset)** without requiring retraining on labeled target data.

## Project Structure

```text
cross-factory-tta/
│
├── data/
│   ├── sh17/                  # source domain (Baseline Training)
│   └── pictor_ppe/            # target domain (Zero-shot evaluation)
│
├── configs/
│   ├── train_sh17.yaml        # training hyperparameters
│   └── tent_config.yaml       # TTA hyperparameters (lr, steps)
│
├── src/
│   ├── train.py               # train YOLOv8m on source domain
│   ├── evaluate.py            # zero-shot evaluation on target
│   ├── tent.py                # TENT implementation for YOLOv8
│   ├── adapt.py               # run TTA and evaluate
│   ├── download_data.py       # Helper script to pull datasets
│   └── utils/
│       ├── class_remap.py     # align class indices across datasets
│       └── metrics.py         # mAP computation and logging
│
├── notebooks/
│   └── results_analysis.ipynb # visualize results, generate table
│
├── results/                   # log outputs and metrics
└── README.md
```

## Getting Started

### 1. Prerequisites
Ensure you have `uv` installed.

```bash
# Install dependencies
uv sync
```

### 2. Prepare Data
Prepare the `data/` folders in the repo format expected by training/evaluation.

This repo includes a Kaggle-friendly preparer that copies your Kaggle-mounted datasets into place:

```bash
uv run python src/prepare_data.py --sh17-dir <SH17_DATASET_DIR> --pictor-dir <PICTOR_DATASET_DIR>
```

Notes:
- On Kaggle, datasets are mounted under `/kaggle/input/<dataset-name>/...`.
- The preparer supports either YOLO-style layouts (`images/train`, `labels/train`, etc.) or the original formats expected by `src/setup_data.py`.

```bash
uv run python src/prepare_data.py --sh17-dir data/sh17 --pictor-dir data/pictor_ppe
```

### 3. Usage
- **Train on Source**: `uv run python src/train.py --data data/sh17/sh17.yaml --model yolov8m.pt --epochs 50 --device mps`
- **Zero-Shot Evaluation**: `uv run python src/evaluate.py --weights runs/train/best.pt --data data/pictor_ppe/pictor.yaml`
- **TENT Adaptation**: `uv run python src/adapt.py --weights runs/train/best.pt --data data/pictor_ppe/pictor.yaml --lr 0.001 --steps 1`

### Kaggle One-Liner
After cloning into `/kaggle/working/cross-factory-tta`:

```bash
python -u src/prepare_data.py \
  --sh17-dir /kaggle/input/<YOUR_SH17_DATASET_DIR> \
  --pictor-dir /kaggle/input/<YOUR_PICTOR_DATASET_DIR>

python -u src/train.py \
  --data data/sh17/sh17.yaml \
  --model yolov8m.pt \
  --epochs 50 \
  --device 0 \
  --project /kaggle/working/runs/train
```

## Methodology
The implementation utilizes **YOLOv8m** for its robustness and clean integration of **Batch Normalization** layers, which are central to the TENT adaptation strategy. By adapting BatchNorm parameters on-the-fly, we improve detection performance in "unseen" manufacturing environments.

---
**Author**: rainsfall
