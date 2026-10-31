"""Resolve the independently frozen deployment used by external/field studies.

Internal calibration revalidation must not silently change an already unblinded
external evaluation or the identity-blind queues selected for annotation.
"""
from pathlib import Path


def deployment_comparison_dir(base: Path) -> Path:
    snapshot = Path(base) / "frozen_deployment_v1"
    if (snapshot / "deployment_snapshot.json").is_file():
        return snapshot / "comparison_19_v1"
    return Path(base) / "comparison_19_v1"


def deployment_dreams_calibration_path(dev_dir: Path) -> Path:
    snapshot = Path(dev_dir).parent / "frozen_deployment_v1"
    if (snapshot / "deployment_snapshot.json").is_file():
        return snapshot / "dreams_head_probability_calibration.csv"
    return Path(dev_dir) / "dreams_head_probability_calibration.csv"
