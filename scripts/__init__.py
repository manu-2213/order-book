"""Command-line entry points; run from the repo root as ``python -m scripts.<name>``."""

from pathlib import Path

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
OUTPUTS.mkdir(exist_ok=True)
