"""
Train YOLOv8m on SH17.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --hparams hparams_fast.yaml
"""
import argparse
import os
import random
import sys
from pathlib import Path

os.environ["NCCL_P2P_DISABLE"] = "1"   # GPUs on different PCIe root complexes — P2P unsupported
os.environ["NCCL_IB_DISABLE"] = "1"    # no InfiniBand on this machine
os.environ["NCCL_DEBUG"] = "WARN"      # keep warnings only

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from src.data_setup import data_ready
from src.hparams import load
from src.train import run_training

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, help="Path to hparams yaml (default: hparams_yolov8m.yaml)")
args = parser.parse_args()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    for i in range(n):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if not data_ready():
    print("[ERROR] Data not prepared. Run scripts/prepare_data.py first.")
    sys.exit(1)

hp = load(args.hparams)
run_training(hp)
