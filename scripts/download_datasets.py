"""
Download all raw datasets to data/.

Handles:
  - SH17       : Kaggle  (kagglehub, built-in resume + integrity)
  - SHWD        : Google Drive zip   (gdown, resume + zip verify)
  - Pictor-PPE  : Google Drive folder (gdown, per-file resume)
  - CHV         : Google Drive zip   (gdown, resume + zip verify)

A .complete marker is written after each dataset finishes.
Re-running skips already-complete datasets unless --force is passed.

Usage:
    uv run python scripts/download_datasets.py
    uv run python scripts/download_datasets.py --force
    uv run python scripts/download_datasets.py --only sh17 pictor_ppe
"""

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

RAW_DIR  = config._ROOT / "data" / "raw"
DATA_DIR = config._ROOT / "data"

DATASETS = {
    "sh17": {
        "type": "kaggle",
        "id":   "mugheesahmad/sh17-dataset-for-ppe-detection",
        "dest": DATA_DIR / "sh17",
    },
    "shwd": {
        "type": "gdrive_zip",
        "id":   "1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX",
        "zip":  RAW_DIR / "shwd.zip",
        "dest": DATA_DIR / "shwd",
    },
    "pictor_ppe": {
        "type": "gdrive_folder",
        "id":   "1akhyTNVrkqMMcIFUQCEbW5ehfmG0CdYH",
        "dest": DATA_DIR / "pictor_ppe",
    },
    "chv": {
        "type": "gdrive_zip",
        "id":   "1fdGn67W0B7ShpBDbbQpUF0ScPQa4DR0a",
        "zip":  RAW_DIR / "chv.zip",
        "dest": DATA_DIR / "chv",
    },
}

MAX_RETRIES    = 10
RETRY_DELAY    = 15   # seconds between retries (flat — gdown handles its own backoff)


# ── helpers ───────────────────────────────────────────────────────────────────

def marker(dest: Path) -> Path:
    return dest / ".complete"

def is_complete(dest: Path) -> bool:
    return marker(dest).exists()

def mark_complete(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    marker(dest).touch()

def verify_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                print(f"  [ERROR] Corrupt entry in zip: {bad}")
                return False
        return True
    except zipfile.BadZipFile as e:
        print(f"  [ERROR] Bad zip file: {e}")
        return False

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

    # kagglehub downloads to its own cache; we copy from there
    def _dl():
        cached = kagglehub.dataset_download(ds["id"])
        cached = Path(cached)
        if dest.exists() and dest != cached:
            shutil.rmtree(dest)
        if cached != dest:
            shutil.copytree(cached, dest, dirs_exist_ok=True)

    retry(_dl, "kaggle download")
    mark_complete(dest)
    print(f"  SH17 -> {dest}")


def download_gdrive_zip(ds: dict):
    import gdown

    zip_path: Path = ds["zip"]
    dest: Path     = ds["dest"]
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Download with resume
    def _dl():
        print(f"  Downloading {zip_path.name} from Google Drive (resume enabled)...")
        gdown.download(
            id=ds["id"],
            output=str(zip_path),
            resume=True,
            quiet=False,
            fuzzy=True,
        )

    retry(_dl, f"gdrive download {zip_path.name}")

    # Verify
    print(f"  Verifying {zip_path.name}...")
    if not verify_zip(zip_path):
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Zip verification failed for {zip_path.name}. File deleted, re-run to retry.")

    # Extract
    print(f"  Extracting to {dest}...")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)

    mark_complete(dest)
    print(f"  Extracted -> {dest}")


def download_gdrive_folder(ds: dict):
    import gdown

    dest: Path = ds["dest"]
    dest.mkdir(parents=True, exist_ok=True)

    # gdown downloads each file individually — naturally resumable at file level
    def _dl():
        print(f"  Downloading Google Drive folder -> {dest}")
        print("  (each file is downloaded individually; interrupted runs resume from last file)")
        gdown.download_folder(
            id=ds["id"],
            output=str(dest),
            quiet=False,
            resume=True,
        )

    retry(_dl, "gdrive folder download")
    mark_complete(dest)
    print(f"  Folder -> {dest}")


# ── main ──────────────────────────────────────────────────────────────────────

HANDLERS = {
    "kaggle":        download_kaggle,
    "gdrive_zip":    download_gdrive_zip,
    "gdrive_folder": download_gdrive_folder,
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
