"""The training script: the sweep matches its array, and a quick run trains, evaluates and resumes end to end."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from papers.decomposition import OUT
from papers.decomposition.scripts import rl_eval, train_rl

HERE = Path(__file__).resolve().parents[1]


JOBS = {"base": ("train_rl", "e8_rl"), "large_tick": ("train_rl_large_tick", "e8_rl_large_tick")}


@pytest.mark.parametrize("sweep_set", sorted(train_rl.SWEEPS))
def test_each_sweep_has_one_row_per_array_task_and_its_decomposition_names_every_policy(sweep_set):
    train_job, decomposition_job = JOBS[sweep_set]
    array = re.search(r"^#SBATCH -a 1-(\d+)", (HERE / "slurm" / f"{train_job}.slurm").read_text(), re.M)
    assert int(array.group(1)) == len(train_rl.SWEEPS[sweep_set])
    names = [row["name"] for row in train_rl.SWEEPS[sweep_set]]
    assert len(set(names)) == len(names)
    decomposition = (HERE / "slurm" / f"{decomposition_job}.slurm").read_text()
    for name in names:
        assert f"rl@outputs/decomposition/rl/{name}/best_policy.pt" in decomposition
    assert "--first-seed 1020" in decomposition, "the decomposition scores the policies on seeds nobody trained or selected on"
    assert f"--regime {train_rl.SWEEP_MARKET[sweep_set]}" in decomposition
    all_names = [row["name"] for rows in train_rl.SWEEPS.values() for row in rows]
    assert len(set(all_names)) == len(all_names), "a run name is a folder under outputs/decomposition/rl/"


def test_every_sweep_row_parses_into_settings():
    for sweep_set, rows in train_rl.SWEEPS.items():
        for i, row in enumerate(rows, 1):
            args = train_rl.settings(train_rl.build_parser().parse_args(["--sweep-set", sweep_set, "--sweep", str(i)]))
            assert args.name == row["name"] and args.gamma == 1.0
            for key, value in row.items():
                if key != "name":
                    assert getattr(args, key) == value


@pytest.mark.parametrize("method, regime", [("ppo", "low_cover_capital10"), ("grpo", "low_cover_large_tick")])
def test_a_quick_run_trains_resumes_and_evaluates(method, regime):
    out = OUT / "rl_quick" / f"smoke_{method}"
    for path in list(out.glob("*")) if out.exists() else []:
        path.unlink()
    train_rl.main(["--quick", "--name", out.name, "--method", method, "--regimes", regime, "--iterations", "2", "--hours", "0"])  # one iteration, then the walltime
    history = json.loads((out / "history.json").read_text())
    assert history["status"] == "paused" and history["iteration"] == 1 and history["config"]["method"] == method
    train_rl.main(["--quick", "--name", out.name, "--method", method, "--regimes", regime, "--iterations", "2", "--resume"])
    history = json.loads((out / "history.json").read_text())
    assert history["status"] == "done" and history["iteration"] == 2 and [r["iteration"] for r in history["history"]] == [1, 2]
    assert history["best"] is not None and (out / "best_policy.pt").exists() and (out / "policy.pt").exists()
    rl_eval.main(["--quick", "--policies", str(out / "best_policy.pt"), "--workers", "2"])
    rows = json.loads((OUT / "rl_eval_quick" / "summary.json").read_text())
    assert {r["seeds"] for r in rows} == {"validation", "unseen"} and all("gap_is" in r and "exposure" in r and r["method"] == method for r in rows)
    analysis = json.loads((OUT / "rl_eval_quick" / "analysis.json").read_text())
    row = analysis["policies"][out.name]
    assert row["is"]["p_holm"] >= row["is"]["p_two_sided"] and len(row["drift_intercept"]) == 3 and 0 <= row["hold"] <= 1 and len(analysis["cells"]) == 3


def test_the_analysis_recovers_a_planted_policy():
    """A policy that always saves 1 bp net of drift and pays 0.5 in drift: pooled gaps, a flat drift slope, and GRPO against PPO."""
    rng = np.random.default_rng(0)
    episodes = []
    for seed in range(1000, 1080):
        drift = rng.normal(0, 5)
        for size in (0.1, 0.4):
            base = dict(regime="low_cover_capital10", size=size, seed=seed, exposure=0.1, aggressive_share=1.0, hold=0.0, forced=0, quantity=100)
            noise = rng.normal(0, 0.1)
            episodes.append(dict(base, name="twap_market", cost_is=3 + drift + noise, cost_net=3 + noise, drift=drift))
            episodes.append(dict(base, name="ppo", cost_is=3 + drift + noise, cost_net=3 + noise, drift=drift))
            episodes.append(dict(base, name="grpo", cost_is=2.5 + drift + noise, cost_net=2 + noise, drift=drift + 0.5, forced=10, hold=0.5))
    analysis = rl_eval.analyse(episodes)
    row = analysis["policies"]["grpo"]
    assert row["is"]["mean"] == pytest.approx(-0.5) and row["net"]["mean"] == pytest.approx(-1.0) and row["drift"]["mean"] == pytest.approx(0.5)
    assert row["drift_slope"] == pytest.approx(0, abs=1e-9) and row["drift_intercept"][0] == pytest.approx(0.5)
    assert row["forced"] == pytest.approx(0.1) and row["hold"] == pytest.approx(0.5) and row["is"]["p_holm"] < 0.05
    assert analysis["policies"]["ppo"]["is"]["mean"] == pytest.approx(0)
    assert analysis["grpo_minus_ppo"]["grpo"]["cost_is"]["mean"] == pytest.approx(-0.5)


def test_the_ceiling_script_runs_and_its_summary_pairs_by_seed():
    from papers.decomposition.scripts import rl_ceiling
    rl_ceiling.main(["--quick", "--regime", "low_cover_large_tick", "--workers", "2"])
    summary = json.loads((OUT / "rl_ceiling_large_tick_quick" / "summary.json").read_text())
    assert summary["market"] == "low_cover_large_tick"
    assert set(summary["ceiling"]) == {"twap_passive", "signal@+0.5", "hold@0", "hold@-1"} and set(summary["adversary"]) == {"medium", "lenient", "lenient_x3", "lenient_x3_0.6"}
    row = summary["ceiling"]["hold@0"]
    assert row["pooled"]["n"] == 2 and "p_holm" in row["pooled"] and 0 <= row["passive"] <= 1
    assert all(len(v["detection_predation"]) == 3 for v in summary["adversary"].values())
    assert all(isinstance(rl_ceiling.strategy(n).name, str) for n in rl_ceiling.HEURISTICS)


def test_policies_are_compared_with_the_best_rule_seed_by_seed_in_its_market():
    from papers.decomposition.scripts.make_paper import against_best_rule
    seeds, sizes = range(1000, 1020), (0.2, 0.4)
    noise = lambda s: (s * 37 % 11) - 5  # large and shared by seed, so only a paired difference cancels it
    rule = lambda name, gain: [{"part": "ceiling", "strategy": name, "size": z, "seed": s, "cost_is": 5 + noise(s) + gain} for z in sizes for s in seeds]
    rules = rule("best", -1.0) + rule("worse", -0.5) + rule("twap_market", 0.0) + [{"part": "adversary"}]
    ceiling = {"market": "m", "ceiling": {"best": {"pooled": {"mean": -1.0}}, "worse": {"pooled": {"mean": -0.5}}}}
    policy = lambda name, regime, gain: [{"name": name, "regime": regime, "size": z, "seed": s, "cost_is": 5 + noise(s) + gain} for z in (0.1, *sizes) for s in seeds]
    episodes = policy("a", "m", -3.0) + policy("b", "m", -1.0) + policy("elsewhere", "other", -9.0) + policy("twap_market", "m", 0.0)
    out = against_best_rule(episodes, rules, ceiling)
    assert out["rule"] == "best" and out["sizes"] == [0.2, 0.4] and set(out["policies"]) == {"a", "b"}
    assert out["policies"]["a"]["mean"] == pytest.approx(-2.0) and out["policies"]["a"]["high"] == pytest.approx(-2.0)
    assert out["policies"]["b"]["mean"] == pytest.approx(0.0) and out["policies"]["a"]["p_holm"] < 0.05


def test_the_results_jobs_are_queued_behind_the_training_chain():
    text = (HERE / "slurm" / "submit_all.sh").read_text()
    assert "AFTER=" in text and "rl_results" in text
    submit = (HERE.parents[1] / "hpc" / "submit.sh").read_text()
    assert 'id="${AFTER:-}"' in submit
