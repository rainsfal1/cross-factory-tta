import shutil
from pathlib import Path

import config


def setup_sh17():
    print("\n=== Setting up SH17 ===")

    layout_a = (config.SH17_DIR / "images" / "train").exists()
    layout_b = (config.SH17_DIR / "train_files.txt").exists()

    if not layout_a and not layout_b:
        raise RuntimeError(
            f"[SH17] Unrecognised layout at {config.SH17_DIR}. "
            "Expected either images/train/ (pre-split) or train_files.txt (public dataset)."
        )

    for split in ("train", "val"):
        out_images = config.DATA_DIR / "sh17" / "images" / split
        out_labels = config.DATA_DIR / "sh17" / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        if layout_a:
            src_images = config.SH17_DIR / "images" / split
            src_labels = config.SH17_DIR / "labels" / split
            if not src_images.exists():
                raise RuntimeError(f"[SH17] images/{split} not found: {src_images}")
            if not src_labels.exists():
                raise RuntimeError(f"[SH17] labels/{split} not found: {src_labels}")
            shutil.copytree(src_images, out_images, dirs_exist_ok=True)
            shutil.copytree(src_labels, out_labels, dirs_exist_ok=True)
        else:
            split_file = config.SH17_DIR / f"{split}_files.txt"
            filenames = [f.strip() for f in split_file.read_text().splitlines() if f.strip()]
            for fname in filenames:
                src_img = config.SH17_DIR / "images" / fname
                src_lbl = config.SH17_DIR / "labels" / (Path(fname).stem + ".txt")
                if src_img.exists():
                    shutil.copy2(src_img, out_images / fname)
                if src_lbl.exists():
                    shutil.copy2(src_lbl, out_labels / src_lbl.name)

        n = len(list(out_images.glob("*")))
        print(f"  SH17 {split}: {n} images")


def _copy_split(src_dir: Path, out_dir: Path, label: str):
    src_images = src_dir / "images" / "test"
    src_labels = src_dir / "labels" / "test"

    if not src_images.exists():
        raise RuntimeError(f"[{label}] images not found: {src_images}")
    if not src_labels.exists():
        raise RuntimeError(f"[{label}] labels not found: {src_labels}")

    out_images = out_dir / "images" / "test"
    out_labels = out_dir / "labels" / "test"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    shutil.copytree(src_images, out_images, dirs_exist_ok=True)
    shutil.copytree(src_labels, out_labels, dirs_exist_ok=True)
    print(f"  {label} test: {len(list(out_images.glob('*')))} images")


def setup_pictor():
    print("\n=== Setting up Pictor-PPE ===")
    _copy_split(config.PICTOR_DIR, config.DATA_DIR / "pictor_ppe", "Pictor")


def setup_shwd():
    print("\n=== Setting up SHWD ===")
    _copy_split(config.SHWD_DIR, config.DATA_DIR / "shwd", "SHWD")


def setup_chv():
    print("\n=== Setting up CHV ===")
    _copy_split(config.CHV_DIR, config.DATA_DIR / "chv", "CHV")


def write_yamls():
    print("\n=== Writing YAML configs ===")

    (config.DATA_DIR / "sh17" / "sh17.yaml").write_text(f"""path: {config.DATA_DIR / 'sh17'}
train: images/train
val: images/val

nc: {len(config.SH17_CLASSES)}
names: {config.SH17_CLASSES}
""")

    for ds_name, yaml_name in [
        ("pictor_ppe", "pictor.yaml"),
        ("shwd",       "shwd.yaml"),
        ("chv",        "chv.yaml"),
    ]:
        (config.DATA_DIR / ds_name / yaml_name).write_text(f"""path: {config.DATA_DIR / ds_name}
train: images/test
val: images/test

nc: {len(config.CANONICAL_CLASSES)}
names: {config.CANONICAL_CLASSES}
""")
        print(f"  {yaml_name} written")


def data_ready() -> bool:
    checks = [
        config.DATA_DIR / "sh17"       / "sh17.yaml",
        config.DATA_DIR / "pictor_ppe" / "pictor.yaml",
        config.DATA_DIR / "shwd"       / "shwd.yaml",
        config.DATA_DIR / "chv"        / "chv.yaml",
    ]
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        print(f"Missing: {missing}")
    return len(missing) == 0
