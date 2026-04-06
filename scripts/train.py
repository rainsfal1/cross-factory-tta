"""
Train YOLOv8m on SH17.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --hparams hparams_fast.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data_setup import data_ready
from src.hparams import load
from src.train import run_training

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, help="Path to hparams yaml (default: hparams.yaml)")
args = parser.parse_args()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if not data_ready():
    print("[ERROR] Data not prepared. Run scripts/prepare_data.py first.")
    sys.exit(1)

hp = load(args.hparams)
run_training(hp)
