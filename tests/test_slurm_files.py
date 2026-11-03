"""Job files follow the cluster's guide (docs.hpc.qmul.ac.uk/using/submittingjobs) and run what exists."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS = sorted(ROOT.glob("papers/*/slurm/*.slurm"))


def directive(text: str, flag: str) -> str | None:
    match = re.search(rf"^#SBATCH {flag}[ =](\S+)", text, re.M)
    return match.group(1) if match else None


def test_there_are_jobs_and_every_paper_has_a_smoke_test():
    assert JOBS
    for paper in {job.parent.parent for job in JOBS}:
        assert (paper / "slurm" / "smoke_test.slurm").exists()


@pytest.mark.parametrize("job", JOBS, ids=lambda p: f"{p.parents[1].name}/{p.stem}")
def test_a_job_requests_what_the_guide_asks_and_its_name_matches_its_file(job):
    text = job.read_text()
    assert text.startswith("#!/bin/bash\n")
    assert directive(text, "-J") == job.stem
    tag = "%A.%a" if directive(text, "-a") else "%j"
    assert directive(text, "-o") == f"logs/%x.o{tag}" and directive(text, "-e") is None, "one log per job, with its errors in it"
    assert directive(text, "-N") == "1", "workers are processes on one machine"
    assert int(directive(text, "-n")) >= 1
    assert re.fullmatch(r"\d+[MG]", directive(text, "--mem-per-cpu")), "the guide wants an integer amount per core"
    assert directive(text, "-t") == "1:0:0", "1-hour jobs qualify for the short queue and start at once"
    assert directive(text, "-p") == "compute,computeshort"


@pytest.mark.parametrize("job", JOBS, ids=lambda p: f"{p.parents[1].name}/{p.stem}")
def test_a_job_uses_tasks_and_only_runs_modules_that_exist(job):
    text = job.read_text()
    assert "SLURM_CPUS_PER_TASK" not in text and "cpus-per-task" not in text, "-n sets SLURM_NTASKS, and -c is unset under it"
    assert "gres" not in text and "--account" not in text
    commands = re.findall(r"python -u -m (\S+)", text)
    assert commands
    for module in commands:
        assert (ROOT / (module.replace(".", "/") + ".py")).exists(), module
    for line in text.splitlines():
        if line.startswith("python -u -m papers"):
            assert "--workers $SLURM_NTASKS" in line


ARRAYS = [job for job in JOBS if directive(job.read_text(), "-a")]


def command(job: Path) -> str:
    return next(line for line in job.read_text().splitlines() if line.startswith("python -u -m"))


def test_the_grids_are_split_into_arrays():
    assert {job.stem for job in GRIDS} >= {"e8_deep_cover", "e8_low_cover", "e8_horizons", "protocol", "detection_curve", "replications", "e8_rl"}
    assert {job.stem for job in SWEEPS} == {"train_rl", "train_rl_large_tick"}


def test_submit_chains_a_job_on_itself_when_asked():
    text = (ROOT / "hpc/submit.sh").read_text()
    assert "CHAIN" in text and "--dependency=afterany" in text


GRIDS = [job for job in ARRAYS if "--shard" in command(job)]
SWEEPS = [job for job in ARRAYS if "--shard" not in command(job)]


@pytest.mark.parametrize("job", SWEEPS, ids=lambda p: p.stem)
def test_a_sweep_array_runs_one_setting_per_task_and_survives_the_hour(job):
    text = command(job)
    assert "--sweep $SLURM_ARRAY_TASK_ID" in text
    assert "--resume" in text and re.search(r"--hours 0\.[0-9]+", text), "a training task must checkpoint before the walltime so the chain can resume it"


@pytest.mark.parametrize("job", GRIDS, ids=lambda p: p.stem)
def test_an_array_has_one_task_per_seed_and_a_merge_with_the_same_grid(job):
    text = job.read_text()
    seeds = int(re.search(r"--seeds (\d+)", command(job)).group(1))
    assert directive(text, "-a") == f"1-{seeds}"
    assert "--shard $SLURM_ARRAY_TASK_ID" in command(job)
    merge = job.with_name(f"{job.stem}_merge.slurm")
    assert merge.exists() and not directive(merge.read_text(), "-a")
    strip = lambda line: re.sub(r" --(shard \S+|merge)", "", line)
    assert strip(command(job)) == strip(command(merge)), "the merge must expect exactly the grid the array runs"


def test_submit_all_names_every_job_of_the_paper_and_nothing_else():
    text = (ROOT / "papers/decomposition/slurm/submit_all.sh").read_text()
    named = {name for group in re.findall(r"^\w+=\((.*)\)$", text, re.M) for name in group.split()}
    jobs = {job.stem for job in JOBS if job.parents[1].name == "decomposition" and job.stem != "smoke_test" and not job.stem.endswith("_merge")}
    assert named == jobs
    assert "final)" in text and "rl_heldout" in text, "the final run also reruns the first-generation policies' held-out evaluation"
