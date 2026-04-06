# Cross-Factory Test-Time Adaptation (TTA) for PPE Detection

Implements **Test-Time Adaptation (TTA)** using **TENT** for object detection, applied to safety PPE detection across industrial environments.

Goal: bridge the performance gap between a **source factory (SH17)** and **target factories (Pictor-PPE, SHWD, CHV)** without retraining on labeled target data.

## Project Structure

```
cross-factory-tta/
├── config.py                   # all paths, hyperparams, class mappings — edit this
├── pyproject.toml              # uv project + dependencies
│
├── src/                        # reusable modules
│   ├── data_setup.py           # dataset preparation + YAML generation
│   ├── tent.py                 # TENT class + helpers
│   ├── train.py                # training logic
│   └── eval.py                 # baseline + TENT evaluation
│
├── scripts/                    # CLI entry points (SSH / local use)
│   ├── prepare_data.py
│   ├── train.py
│   ├── eval_baseline.py
│   └── eval_tent.py
│
├── notebooks/
│   ├── kaggle/                 # self-contained notebook for Kaggle runs
│   │   └── tta_experiment.ipynb
│   └── scratch/                # local throwaway analysis (gitignored)
│
├── models/
│   ├── runs.md                 # log: checkpoint → run → results
│   └── *.pt                    # model weights (gitignored)
│
├── data/                       # datasets (gitignored except metadata)
│   └── datasets.md
│
└── docs/                       # write-ups and visual examples
```

## Workflows

### SSH / Local (GPU machine)

Raw datasets live on the mounted data disk. All working output (prepared data, checkpoints, eval results) goes to `/mnt/data/cross-factory-tta/`.

**1. Configure paths**

Edit [config.py](config.py) — set `SH17_DIR`, `PICTOR_DIR`, etc. to where your raw data is.

**2. Set up environment**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

**3. Run**

```bash
uv run python scripts/prepare_data.py   # organise datasets + write YAMLs
uv run python scripts/train.py          # train YOLOv8m on SH17
uv run python scripts/eval_baseline.py  # zero-shot eval on all target domains
uv run python scripts/eval_tent.py      # TENT adaptation on Pictor-PPE
```

### Kaggle

The notebook in `notebooks/kaggle/` is fully self-contained — no imports from `src/`. Edit the config section at the top of the notebook for paths and platform, then run with **Save & Run All**.

## Datasets

| Dataset    | Role            | Classes                              |
|------------|-----------------|--------------------------------------|
| SH17       | Source (train)  | 17-class PPE                         |
| Pictor-PPE | Target (eval)   | hard_hat / no_hard_hat / person      |
| SHWD       | Target (eval)   | hard_hat / person                    |
| CHV        | Target (eval)   | hard_hat / no_hard_hat / person      |

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
