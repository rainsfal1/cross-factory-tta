"""
Prepare all datasets for training and evaluation.

Usage:
    uv run python scripts/prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_setup import data_ready, setup_sh17, setup_pictor, setup_shwd, setup_chv, write_yamls

if data_ready():
    print("Data already prepared.")
else:
    setup_sh17()
    setup_pictor()
    setup_shwd()
    setup_chv()
    write_yamls()
    print("\nData preparation complete.")
