from pathlib import Path
import yaml

_DEFAULT = Path(__file__).parent.parent / "hparams_yolov8m.yaml"


def load(path: Path | str | None = None) -> dict:
    p = Path(path) if path else _DEFAULT
    with open(p) as f:
        return yaml.safe_load(f)
