"""Running many jobs: the process pool and the resumable result log every experiment shares.

``iter_jobs`` yields results in job order, from a process pool when ``workers > 1``.

``load_results`` reads back a JSONL log so that a job killed at its walltime can continue: a record is
written and flushed as soon as its run finishes, so at most the last line is damaged, and that line is
dropped so the file can be appended to again.

A grid can also be split into short jobs, one seed per Slurm array task, each with its own log in
``out/shards/<task>/``. Seeds are the natural unit: runs paired on a seed share a burn-in snapshot, and no
statistic crosses a seed until the merge, which refuses to summarise an incomplete or mixed set of shards.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def iter_jobs(fn, jobs: list, workers: int = 1, chunksize: int = 1):
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            yield from pool.map(fn, jobs, chunksize=chunksize)
    else:
        yield from map(fn, jobs)


def load_results(path: Path, make=lambda record: record) -> list:
    """The records already in a JSONL log, each passed through ``make``; a truncated last line is removed."""
    path = Path(path)
    if not path.exists():
        return []
    results, kept = [], 0
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            results.append(make(json.loads(line)))
            kept += len(line)
    if kept != path.stat().st_size:
        with open(path, "r+b") as f:
            f.truncate(kept)
    return results


# ----- splitting a grid into short jobs, one seed each ------------------------------------------- #

def shard_dir(out: Path, shard: int) -> Path:
    """Where shard ``shard`` (1-based, a Slurm array task id) of a run keeps its log and manifest."""
    return Path(out) / "shards" / f"{shard:04d}"


def seed_of_shard(seeds, shard: int) -> int:
    if not 1 <= shard <= len(seeds):
        raise ValueError(f"shard {shard} is out of range: this grid has {len(seeds)} seeds, so shards 1..{len(seeds)}")
    return seeds[shard - 1]


def load_shards(out: Path, make=lambda record: record, log: str = "episodes.jsonl") -> list:
    """The records of every shard, refusing shards written by different simulator versions."""
    from order_book.provenance import shard_versions

    versions = shard_versions(out)
    if len(versions) > 1:
        raise RuntimeError(f"{out} holds shards from simulator versions {sorted(versions)}; they cannot be merged")
    records = []
    for path in sorted((Path(out) / "shards").glob(f"*/{log}")):
        records += load_results(path, make)
    return records


def missing_shards(expected: dict, have: set, seeds) -> list[int]:
    """Shards with work left. ``expected`` maps each run's key to its seed; ``have`` holds the finished keys."""
    unfinished = {seed for key, seed in expected.items() if key not in have}
    return [i for i, seed in enumerate(seeds, 1) if seed in unfinished]


class IncompleteShards(RuntimeError):
    """A merge found runs missing; the message says which array tasks to submit again."""

    def __init__(self, missing: list[int], out: Path):
        self.missing = missing
        out = Path(out)
        job = os.environ.get("SLURM_JOB_NAME", out.name).removesuffix("_merge")
        super().__init__(f"{out}: {len(missing)} shards unfinished. Resubmit them; finished runs are kept:\n"
                         f"    ARRAY={','.join(map(str, missing))} bash hpc/submit.sh {out.parent.name} {job}")
