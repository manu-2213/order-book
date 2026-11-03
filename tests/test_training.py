"""Training inside the paired worlds: the reward's accounting, parallel rollouts, resumption and seed hygiene."""

import pytest
import torch

from order_book.counterfactual import FULL, run_paired
from order_book.execution import PolicyNet, PolicyStrategy, make_strategy, new_policy
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


def test_rewards_sum_to_minus_the_shortfall_plus_the_control_variate_minus_the_charges():
    env = with_reward(PairedExecutionEnv(**QUICK), detection_penalty=0.05, impact_penalty=0.5)
    transitions, r = env.rollout(PolicyStrategy(new_policy(seed=0), deterministic=False), 3, 0.3)
    assert transitions and r.decisions == len(transitions)
    assert r.ret == pytest.approx(-(r.cost_is - r.control + r.impact_charge + r.detection_charge + r.incomplete_charge), abs=1e-9)
    assert r.detection_charge == pytest.approx(0.05 * r.exposure * r.decisions, rel=0.05) or r.exposure == 0  # charged per active step
    assert 0.0 <= r.exposure <= 1.0


def test_the_control_variate_does_not_depend_on_the_policy():
    env = PairedExecutionEnv(**QUICK)
    _, a = env.rollout(PolicyStrategy(new_policy(seed=0), deterministic=False), 3, 0.3)
    _, b = env.rollout(make_strategy("twap_market"), 3, 0.3)
    _, c = env.rollout(make_strategy("immediate"), 3, 0.3)  # done long before the horizon, and still credited the whole schedule
    assert a.control == pytest.approx(b.control) == pytest.approx(c.control) and a.cost_is != pytest.approx(b.cost_is)
    assert a.control == pytest.approx(b.drift, abs=0.5), "a uniform schedule's drift is close to what TWAP's fills paid along Arm 0's mid"


def test_the_net_of_drift_reward_is_available_and_the_charges_are_exact_nulls_when_off():
    env = PairedExecutionEnv(**QUICK, reward=RewardSpec(benchmark="net"))
    _, r = env.rollout(PolicyStrategy(new_policy(seed=0), deterministic=False), 4, 0.3)
    assert r.detection_charge == 0.0 and r.impact_charge == 0.0 and r.control == 0.0
    assert r.ret == pytest.approx(-(r.cost_net + r.incomplete_charge), abs=1e-9)
    with pytest.raises(ValueError):
        RewardSpec(benchmark="arrival")


def test_the_market_is_built_as_the_decomposition_builds_it():
    env = PairedExecutionEnv(**QUICK)
    spec = env.spec(7, 0.2)
    assert spec.regime == "low_cover_capital10" and spec.rng == "counter" and spec.channels == FULL
    assert spec.config().endowment.speculator == 10 * RegimeSpec("low_cover", 1.0, 1).base().endowment.speculator


# ----- rollouts ------------------------------------------------------------------------------- #

def test_parallel_rollouts_equal_serial_ones_exactly():
    env = PairedExecutionEnv(**QUICK)
    weights = cpu_state_dict(new_policy(seed=3))
    jobs = [(weights, False, 5, 0.1, 0, 0), (weights, False, 6, 0.3, 0, 0), ("twap_market", True, 5, 0.1, 0, 0), (weights, False, 5, 0.1, 0, 1)]
    serial = Rollouts(env, 1).run(jobs)
    parallel = Rollouts(env, 2)
    try:
        results = parallel.run(jobs)
    finally:
        parallel.close()
    for (ts, rs), (tp, rp) in zip(serial, results):
        assert rs == rp
        assert [(t.action, t.reward) for t in ts] == [(t.action, t.reward) for t in tp]
    assert [t.action for t in serial[0][0]] != [t.action for t in serial[3][0]], "another sample of the same seed is another episode"


def test_a_greedy_rollout_is_the_agent_the_decomposition_scores(tmp_path):
    env = PairedExecutionEnv(**QUICK)
    net = new_policy(seed=2)
    net.save(tmp_path / "policy.pt")
    _, record = env.rollout(PolicyStrategy(net, deterministic=True), 11, 0.3)  # records transitions, and must still take the argmax
    expected = run_paired(env.spec(11, 0.3, strategy=f"rl@{tmp_path / 'policy.pt'}"))
    assert record.cost_is == pytest.approx(expected.shortfall_bps)
    assert record.drift == pytest.approx(expected.drift_bps)
    _, sampled = env.rollout(PolicyStrategy(net, deterministic=False), 11, 0.3)
    assert sampled.cost_is != pytest.approx(record.cost_is)


def test_a_net_trained_before_a_feature_or_action_was_appended_still_runs():
    env = PairedExecutionEnv(**QUICK)
    old = PolicyNet(n_features=14, n_actions=15)
    _, direct = env.rollout(PolicyStrategy(old, deterministic=True), 8, 0.1)
    assert direct.executed == direct.quantity
    for workers in (1, 2):  # through the workers too, as rl_eval runs it, alongside a policy of the current shape
        rollouts = Rollouts(env, workers)
        try:
            results = rollouts.run([(cpu_state_dict(old), True, 8, 0.1, 0, 0), (cpu_state_dict(new_policy(seed=1)), True, 8, 0.1, 0, 0)])
        finally:
            rollouts.close()
        assert results[0][1] == direct and results[1][1].executed == results[1][1].quantity


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


def test_grpo_centres_each_group_on_its_own_mean_and_trains_without_a_critic(tmp_path):
    env = PairedExecutionEnv(**QUICK)
    ppo = PPOConfig(iterations=2, episodes_per_update=4, epochs=1, minibatch=64, train_seeds=tuple(range(0, 20)))
    trainer = PairedTrainer(env, TrainingConfig(ppo=ppo, method="grpo", group_size=2, validation_seeds=(800,), eval_every=2), workers=1, out=tmp_path)
    assert trainer.cfg.value_coef == 0.0
    episodes = trainer.episodes()
    assert [e[:3] for e in episodes[::2]] == [e[:3] for e in episodes[1::2]] and [e[3] for e in episodes] == [0, 1, 0, 1]
    batch, stats = trainer.collect()
    adv = batch["advantages"].numpy()
    assert stats["transitions"] == len(adv) and abs(adv).max() > 0
    assert len(set(adv.round(6))) <= 4, "one advantage per episode, its return against the group's mean"
    assert trainer.train(verbose=False) == "done" and trainer.iteration == 2 and trainer.best is not None
    with pytest.raises(ValueError):
        TrainingConfig(ppo=ppo, method="grpo", group_size=3)
    with pytest.raises(ValueError):
        TrainingConfig(ppo=ppo, method="reinforce")


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
