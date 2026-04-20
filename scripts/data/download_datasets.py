"""
Download all raw datasets to data/.

Handles:
  - SH17       : Kaggle (kagglehub, built-in resume + integrity)
  - SHWD        : Kaggle (zxy000/shwd-dataset) → data/raw/shwd/
  - Pictor-PPE  : Kaggle (zyanahmed/pictor-ppe) → data/raw/pictor_ppe_source/
  - CHV         : Kaggle (zyanahmed/chv-dataset) → data/chv/

A .complete marker is written after each dataset finishes.
Re-running skips already-complete datasets unless --force is passed.

Usage:
    uv run python scripts/data/download_datasets.py
    uv run python scripts/data/download_datasets.py --force
    uv run python scripts/data/download_datasets.py --only sh17
    uv run python scripts/data/download_datasets.py --only chv
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

DATA_DIR = config._ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"

DATASETS = {
    "sh17": {
        "type": "kaggle",
        "id":   "mugheesahmad/sh17-dataset-for-ppe-detection",
        "dest": DATA_DIR / "sh17",
    },
    "shwd": {
        "type": "kaggle",
        "id":   "zxy000/shwd-dataset",
        "dest": RAW_DIR / "shwd",
    },
    "pictor_ppe": {
        "type": "kaggle",
        "id":   "zyanahmed/pictor-ppe",
        "dest": RAW_DIR / "pictor_ppe_source",
    },
    "chv": {
        "type": "kaggle",
        "id":   "zyanahmed/chv-dataset",
        "dest": DATA_DIR / "chv",
    },
}

MAX_RETRIES = 10
RETRY_DELAY = 15


# ── helpers ───────────────────────────────────────────────────────────────────

def marker(dest: Path) -> Path:
    return dest / ".complete"

def is_complete(dest: Path) -> bool:
    return marker(dest).exists()

def mark_complete(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    marker(dest).touch()

def retry(fn, label: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fn()
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  [FAIL] {label} failed after {MAX_RETRIES} attempts: {e}")
                raise
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {label} — {e} — waiting {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)


# ── downloaders ───────────────────────────────────────────────────────────────

def download_kaggle(ds: dict):
    import kagglehub

    dest: Path = ds["dest"]
    print(f"  Downloading from Kaggle: {ds['id']}")
    print("  (kagglehub handles resume and integrity internally)")

    def _dl():
        cached = Path(kagglehub.dataset_download(ds["id"]))
        if dest.exists() and dest != cached:
            shutil.rmtree(dest)
        if cached != dest:
            shutil.copytree(cached, dest, dirs_exist_ok=True)

    retry(_dl, "kaggle download")
    mark_complete(dest)
    print(f"  {ds['id'].split('/')[-1]} -> {dest}")


# ── main ──────────────────────────────────────────────────────────────────────

HANDLERS = {
    "kaggle": download_kaggle,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if already complete")
    parser.add_argument("--only",  nargs="+", choices=list(DATASETS), metavar="DATASET",
                        help="Download only specific datasets")
    args = parser.parse_args()

    targets = args.only or list(DATASETS)

    W = 50
    print()
    print(f"  Dataset Download")
    print(f"  {'=' * W}")
    print(f"  Target: {DATA_DIR}")
    print(f"  {'=' * W}")

    failed = []
    for name in targets:
        ds = DATASETS[name]
        dest = ds["dest"]

        if not args.force and is_complete(dest):
            print(f"\n  [{name}] already complete, skipping  (--force to re-download)")
            continue

        print(f"\n  [{name}]")
        print(f"  {'-' * W}")
        try:
            HANDLERS[ds["type"]](ds)
            print(f"  [{name}] done")
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            failed.append(name)

    print()
    print(f"  {'=' * W}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
        print(f"  Re-run to retry failed datasets.")
        sys.exit(1)
    else:
        print(f"  All datasets ready.")
    print(f"  {'=' * W}")
    print()


if __name__ == "__main__":
    main()
