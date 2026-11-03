"""The training script: the sweep matches its array, and a quick run trains, evaluates and resumes end to end."""

import json
import re
from pathlib import Path

import pytest

from papers.decomposition import OUT
from papers.decomposition.scripts import rl_eval, train_rl

HERE = Path(__file__).resolve().parents[1]


def test_the_sweep_has_one_row_per_array_task_and_the_decomposition_names_every_policy():
    array = re.search(r"^#SBATCH -a 1-(\d+)", (HERE / "slurm" / "train_rl.slurm").read_text(), re.M)
    assert int(array.group(1)) == len(train_rl.SWEEP)
    names = [row["name"] for row in train_rl.SWEEP]
    assert len(set(names)) == len(names)
    decomposition = (HERE / "slurm" / "e8_rl.slurm").read_text()
    for name in names:
        assert f"rl@outputs/decomposition/rl/{name}/best_policy.pt" in decomposition
    assert "--first-seed 1020" in decomposition, "the decomposition scores the policies on seeds nobody trained or selected on"


def test_every_sweep_row_parses_into_settings():
    for i, row in enumerate(train_rl.SWEEP, 1):
        args = train_rl.settings(train_rl.build_parser().parse_args(["--sweep", str(i)]))
        assert args.name == row["name"]
        for key, value in row.items():
            if key != "name":
                assert getattr(args, key) == value


@pytest.mark.parametrize("method", ["ppo", "grpo"])
def test_a_quick_run_trains_resumes_and_evaluates(method):
    out = OUT / "rl_quick" / f"smoke_{method}"
    for path in list(out.glob("*")) if out.exists() else []:
        path.unlink()
    train_rl.main(["--quick", "--name", out.name, "--method", method, "--iterations", "2", "--hours", "0"])  # one iteration, then the walltime
    history = json.loads((out / "history.json").read_text())
    assert history["status"] == "paused" and history["iteration"] == 1 and history["config"]["method"] == method
    train_rl.main(["--quick", "--name", out.name, "--method", method, "--iterations", "2", "--resume"])
    history = json.loads((out / "history.json").read_text())
    assert history["status"] == "done" and history["iteration"] == 2 and [r["iteration"] for r in history["history"]] == [1, 2]
    assert history["best"] is not None and (out / "best_policy.pt").exists() and (out / "policy.pt").exists()
    rl_eval.main(["--quick", "--policies", str(out / "best_policy.pt"), "--workers", "2"])
    rows = json.loads((OUT / "rl_eval_quick" / "summary.json").read_text())
    assert {r["seeds"] for r in rows} == {"validation", "unseen"} and all("gap_is" in r and "exposure" in r and r["method"] == method for r in rows)


def test_the_results_jobs_are_queued_behind_the_training_chain():
    text = (HERE / "slurm" / "submit_all.sh").read_text()
    assert "AFTER=" in text and "rl_results" in text
    submit = (HERE.parents[1] / "hpc" / "submit.sh").read_text()
    assert 'id="${AFTER:-}"' in submit
