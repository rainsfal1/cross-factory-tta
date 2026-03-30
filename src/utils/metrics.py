"""mAP logging utilities and pretty results table."""

import json
from pathlib import Path


def load_results(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def print_comparison_table(baseline_path: str | Path, tent_path: str | Path):
    """Print the results table for the proposal."""
    b = load_results(baseline_path)
    t = load_results(tent_path)

    src_map = b.get("source_mAP50", "N/A")
    baseline_map = b["mAP50"]
    tent_map = t["tent_mAP50"]
    drop = tent_map - baseline_map

    header = f"{'Method':<35} {'Source mAP50':>13} {'Target mAP50':>13} {'Drop':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    print(f"{'Source-only (no adaptation)':<35} {str(src_map):>13} {baseline_map:>13.4f} {'N/A':>8}")
    print(f"{'TENT (TTA, unlabeled)':<35} {'':>13} {tent_map:>13.4f} {drop:>+8.4f}")
    print(sep)
