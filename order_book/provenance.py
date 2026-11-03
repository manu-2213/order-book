"""Where a result came from: the simulator's behaviour version, the code and the run's configuration.

``SIMULATOR_VERSION`` names the simulator's behaviour. It changes only when a run's output changes, and
``tests/test_simulator_version.py`` enforces that: it holds a fingerprint per version, so a change in behaviour
fails the suite until the version is bumped and the new fingerprint recorded. A paper records the version
it ran on and can stay on that tag while the market keeps changing for the next paper.

``write_manifest`` appends one entry per invocation to ``manifest.json`` next to a run's results, and refuses
to resume results produced by another simulator version or with another configuration, which would silently
mix two markets or label old runs as the new setting (the result logs identify a run by its seed, strategy,
size, horizon, channels and regime only).

The cluster has no ``.git`` (it is excluded from the sync), so ``hpc/push.sh`` writes the commit to
``.source_version``, which is read when git is unavailable.
"""

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

SIMULATOR_VERSION = "1"

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = ROOT / ".source_version"


def source_version() -> dict:
    """{"commit", "dirty", "origin"}: from git when there is a repository, else from what ``push.sh`` recorded."""
    try:
        run = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        commit = run("rev-parse", "HEAD")
        return {"commit": commit, "dirty": bool(run("status", "--porcelain", "--", "order_book", "papers", "scripts")),
                "origin": "git"}
    except (OSError, subprocess.CalledProcessError):
        pass
    if SOURCE_FILE.exists():
        return {**json.loads(SOURCE_FILE.read_text()), "origin": str(SOURCE_FILE.name)}
    return {"commit": None, "dirty": None, "origin": "unknown"}


def _jsonable(config) -> object:
    return asdict(config) if is_dataclass(config) else config


def write_manifest(out_dir: Path, config=None, resume: bool = False) -> dict:
    """Record this run in ``out_dir/manifest.json``. With ``resume``, refuse a different simulator version or configuration."""
    import numpy

    path = Path(out_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(path.read_text()) if path.exists() else []
    config = json.loads(json.dumps(_jsonable(config), default=str))  # as the file will hold it, so that it compares
    if resume and entries and entries[-1]["simulator_version"] != SIMULATOR_VERSION:
        raise RuntimeError(f"{out_dir} holds results from simulator version {entries[-1]['simulator_version']}; this is "
                           f"version {SIMULATOR_VERSION}. Resuming would mix two markets: use a new output directory.")
    if resume and entries and entries[-1]["config"] != config:
        raise RuntimeError(f"{out_dir} holds results of a run with other settings; resuming would label them as this run's. "
                           f"Use a new output directory.")
    entry = {"simulator_version": SIMULATOR_VERSION, **source_version(), "resumed": bool(resume and entries),
             "argv": sys.argv, "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
             "python": platform.python_version(), "numpy": numpy.__version__, "config": config}
    if not resume:
        entries = []  # a fresh run replaces the results, so the record of the old ones goes with them
    path.write_text(json.dumps(entries + [entry], indent=1, default=str))
    return entry


def shard_versions(out: Path) -> set[str]:
    """The simulator versions recorded by a run's shards."""
    versions = set()
    for path in (Path(out) / "shards").glob("*/manifest.json"):
        versions.update(entry["simulator_version"] for entry in json.loads(path.read_text()))
    return versions
