"""Training inside the paired worlds: the reward's accounting, parallel rollouts, resumption and seed hygiene."""

import pytest
import torch

from order_book.counterfactual import FULL, run_paired
from order_book.execution import PolicyNet, PolicyStrategy, make_strategy, new_policy
from order_book.execution.features import N_FEATURES
from order_book.execution.ppo import PPOConfig
from order_book.experiments import CAPITAL_REGIMES, RegimeSpec
from order_book.training import (
    FIRST_TEST_SEED, PairedExecutionEnv, PairedTrainer, RewardSpec, Rollouts, TrainingConfig, VALIDATION_SEEDS, cpu_state_dict, with_reward,
)

LOW = next(r for r in CAPITAL_REGIMES if r.name == "low_cover_capital10")
QUICK = dict(horizon=200, burn_in=400, configs=((LOW.name, LOW.base()),), sizes=(0.1, 0.3))


def quick_training(**changes) -> TrainingConfig:
    ppo = PPOConfig(iterations=3, episodes_per_update=3, epochs=1, minibatch=64, train_seeds=tuple(range(0, 20)))
    return TrainingConfig(ppo=ppo, validation_seeds=(800, 801), eval_every=2, checkpoint_every=1, **changes)


# ----- the reward's accounting ---------------------------------------------------------------- #

def test_a_baseline_episode_is_exactly_the_decompositions_paired_run():
    env = PairedExecutionEnv(**QUICK)
    _, record = env.rollout(make_strategy("twap_market"), 1000, 0.3)
    expected = run_paired(env.spec(1000, 0.3, strategy="twap_market"))
    assert record.quantity == expected.quantity and record.executed == expected.executed == record.quantity
    assert record.shortfall_bps == pytest.approx(expected.shortfall_bps)
    assert record.cost_is == pytest.approx(expected.shortfall_bps)
    assert record.drift == pytest.approx(expected.drift_bps), "cost net of drift is measured against the same Arm-0 mids as the decomposition"


def test_rewards_sum_to_minus_the_cost_net_of_drift_and_the_charges():
    env = with_reward(PairedExecutionEnv(**QUICK), detection_penalty=0.05, impact_penalty=0.5)
    transitions, r = env.rollout(PolicyStrategy(new_policy(seed=0), deterministic=False), 3, 0.3)
    assert transitions and r.decisions == len(transitions) and transitions[-1].done
    assert r.ret == pytest.approx(-(r.cost_net + r.impact_charge + r.detection_charge + r.incomplete_charge), abs=1e-9)
    assert r.detection_charge == pytest.approx(0.05 * r.exposure * r.decisions, rel=0.05) or r.exposure == 0  # charged per active step
    assert 0.0 <= r.exposure <= 1.0


def test_the_plain_shortfall_reward_is_available_and_the_charges_are_exact_nulls_when_off():
    env = PairedExecutionEnv(**QUICK, reward=RewardSpec(benchmark="arrival"))
    _, r = env.rollout(PolicyStrategy(new_policy(seed=0), deterministic=False), 4, 0.3)
    assert r.detection_charge == 0.0 and r.impact_charge == 0.0
    assert r.ret == pytest.approx(-(r.cost_is + r.incomplete_charge), abs=1e-9)


def test_the_market_is_built_as_the_decomposition_builds_it():
    env = PairedExecutionEnv(**QUICK)
    spec = env.spec(7, 0.2)
    assert spec.regime == "low_cover_capital10" and spec.rng == "counter" and spec.channels == FULL
    assert spec.config().endowment.speculator == 10 * RegimeSpec("low_cover", 1.0, 1).base().endowment.speculator


# ----- rollouts ------------------------------------------------------------------------------- #

def test_parallel_rollouts_equal_serial_ones_exactly():
    env = PairedExecutionEnv(**QUICK)
    weights = cpu_state_dict(new_policy(seed=3))
    jobs = [(weights, False, 5, 0.1, 0), (weights, False, 6, 0.3, 0), ("twap_market", True, 5, 0.1, 0)]
    serial = Rollouts(env, 1).run(jobs)
    parallel = Rollouts(env, 2)
    try:
        results = parallel.run(jobs)
    finally:
        parallel.close()
    for (ts, rs), (tp, rp) in zip(serial, results):
        assert rs == rp
        assert [(t.action, t.reward) for t in ts] == [(t.action, t.reward) for t in tp]


def test_a_net_trained_before_a_feature_was_appended_still_runs():
    env = PairedExecutionEnv(**QUICK)
    old = PolicyNet(n_features=N_FEATURES - 1)
    _, r = env.rollout(PolicyStrategy(old, deterministic=True), 8, 0.1)
    assert r.executed == r.quantity


# ----- the trainer ---------------------------------------------------------------------------- #

def test_training_resumes_from_a_checkpoint_to_the_same_weights(tmp_path):
    env = PairedExecutionEnv(**QUICK)
    straight = PairedTrainer(env, quick_training(), workers=1, out=tmp_path / "straight")
    assert straight.train(verbose=False) == "done"
    paused = PairedTrainer(env, quick_training(), workers=1, out=tmp_path / "paused")
    assert paused.train(hours=0.0, verbose=False) == "paused" and paused.iteration == 1  # one iteration, then the walltime stops it
    resumed = PairedTrainer(env, quick_training(), workers=1, out=tmp_path / "paused")
    assert resumed.resume() and resumed.iteration == 1
    assert resumed.train(verbose=False) == "done" and resumed.iteration == 3
    for a, b in zip(straight.net.parameters(), resumed.net.parameters()):
        assert torch.allclose(a, b, atol=1e-7)
    assert [row["iteration"] for row in resumed.history] == [1, 2, 3]
    assert [e["iteration"] for e in resumed.evals] == [2, 3] and resumed.best is not None
    assert (tmp_path / "paused" / "policy.pt").exists() and (tmp_path / "paused" / "best_policy.pt").exists()


def test_validation_pairs_the_policy_with_twap_on_the_same_episodes(tmp_path):
    env = PairedExecutionEnv(**QUICK)
    trainer = PairedTrainer(env, quick_training(), workers=1)
    ev = trainer.validate()
    assert len(trainer.twap_validation) == len(env.sizes) * 2
    assert ev["gap_net"] == pytest.approx(ev["cost_net"] - ev["twap_cost_net"], abs=1e-9)
    assert ev["gap_net_low"] <= ev["gap_net"] <= ev["gap_net_high"]


def test_training_and_validation_never_touch_a_test_seed():
    assert max(VALIDATION_SEEDS) < FIRST_TEST_SEED and max(TrainingConfig().ppo.train_seeds) < min(VALIDATION_SEEDS)
    with pytest.raises(ValueError):
        TrainingConfig(ppo=PPOConfig(train_seeds=(0, 1000)))
    with pytest.raises(ValueError):
        TrainingConfig(ppo=PPOConfig(train_seeds=(800,)))
