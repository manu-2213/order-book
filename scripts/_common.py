"""Shared helpers for the experiment scripts."""

import json
from pathlib import Path

from order_book.execution.calibration import load as load_calibration
from scripts import OUTPUTS

CALIBRATION_PATH = OUTPUTS / "calibration.json"
WEIGHTS_PATH = OUTPUTS / "detectors" / "learned.npz"


def calibration_or_default():
    if CALIBRATION_PATH.exists():
        print(f"[config] calibration from {CALIBRATION_PATH}")
        return load_calibration(CALIBRATION_PATH)
    print("[config] no outputs/calibration.json: using default Calibration (run scripts/calibrate.py)")
    return None


def learned_weights_or_none():
    if WEIGHTS_PATH.exists():
        print(f"[config] learned detector from {WEIGHTS_PATH}")
        return str(WEIGHTS_PATH)
    print("[config] no outputs/detectors/learned.npz: A3 observer skipped (run scripts/train_detector.py)")
    return None


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"saved {path}")
