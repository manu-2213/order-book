"""Phase 3: the learned executor -- features, environment, policy and PPO (plan section 5)."""

import math

import numpy as np
import pytest
import torch

from order_book.execution import (
    ACTIONS, ExecutionEnv, N_ACTIONS, N_FEATURES, PolicyNet, PolicyStrategy, PPOConfig, PPOTrainer,
    decision, features, make_strategy, new_policy, phase3_config,
)
from order_book.execution.task import ExecState


def fake_state(remaining=100, executed=0, decisions_left=10, side="bid"):
    """A minimal ExecState with a stub public object for testing the encoder."""
    class _Pub:
        momentum = [0.01, -0.02]
        var_fast = 1.0
        attention = 1.2
    return ExecState(
        step=1000, elapsed=500, horizon=2000, decision_interval=5, decisions_left=decisions_left, side=side,
        quantity=executed + remaining, remaining=remaining, executed=executed, aggressive_executed=0,
        arrival_mid=10_000.0, mid=10_001.0, best_bid=9_999, best_ask=10_001, spread=2, imbalance=0.1,
        public_volume_since_start=500, own_resting=0, own_resting_orders=[], expected_volume_per_step=5.0,
        own_flow_share=0.2, public=_Pub(),
    )


def test_feature_vector_shape_and_finite():
    vec = features(fake_state())
    assert vec.shape == (N_FEATURES,)
    assert np.all(np.isfinite(vec))


def test_feature_sign_flips_with_side():
    buy = features(fake_state(side="bid"))
    sell = features(fake_state(side="ask"))
    # imbalance feature (index 5) is sign-adjusted, so it flips with the side
    assert buy[5] == pytest.approx(-sell[5])


def test_decision_hold_and_market_and_passive():
    s = fake_state(remaining=100, decisions_left=10)
    # find a mult=0 action -> hold, a market action, and a passive action
    hold = next(i for i, (m, p) in enumerate(ACTIONS) if m == 0.0)
    market = next(i for i, (m, p) in enumerate(ACTIONS) if m == 1.0 and p == "market")
    passive = next(i for i, (m, p) in enumerate(ACTIONS) if m == 1.0 and p == "passive_best")
    assert decision(hold, s).market_qty == 0 and decision(hold, s).limit_qty == 0
    d_m = decision(market, s)
    assert d_m.market_qty == 10 and d_m.limit_qty == 0      # remaining/decisions_left = 10
    d_p = decision(passive, s)
    assert d_p.limit_qty == 10 and d_p.market_qty == 0


def test_rollout_reward_equals_negative_shortfall():
    env = ExecutionEnv(size_fraction=0.2, horizon=300, burn_in=400)
    strat = PolicyStrategy(new_policy(seed=0), deterministic=False)
    transitions, info = env.rollout(strat, seed=1000)
    assert transitions and info.decisions == len(transitions)
    assert info.ret == pytest.approx(-info.shortfall_bps, abs=1e-6)
    assert sum(t.reward for t in transitions) == pytest.approx(info.ret, abs=1e-6)


def test_rollout_is_deterministic_for_a_seed():
    env = ExecutionEnv(size_fraction=0.2, horizon=300, burn_in=400)
    net = new_policy(seed=1)
    a = env.rollout(PolicyStrategy(net, deterministic=False), seed=1000)[1]
    b = env.rollout(PolicyStrategy(net, deterministic=False), seed=1000)[1]
    assert a.shortfall_bps == pytest.approx(b.shortfall_bps)
    assert a.executed == b.executed


def test_policy_save_load_round_trip(tmp_path):
    net = new_policy(seed=2)
    path = tmp_path / "policy.pt"
    net.save(path)
    loaded = PolicyNet.load(path)
    x = torch.randn(4, N_FEATURES)
    with torch.no_grad():
        l0, v0 = net(x)
        l1, v1 = loaded(x)
    assert torch.allclose(l0, l1) and torch.allclose(v0, v1)


def test_make_strategy_loads_rl_policy(tmp_path):
    path = tmp_path / "policy.pt"
    new_policy(seed=3).save(path)
    strat = make_strategy(f"rl@{path}")
    assert isinstance(strat, PolicyStrategy)
    assert isinstance(decision(0, fake_state()), type(decision(0, fake_state())))


def test_ppo_update_runs_and_changes_weights():
    env = ExecutionEnv(size_fraction=0.2, horizon=300, burn_in=400)
    cfg = PPOConfig(iterations=1, episodes_per_update=3, epochs=2, minibatch=64, seed=0,
                    train_seeds=tuple(range(0, 20)))
    trainer = PPOTrainer(env, cfg)
    before = [p.detach().clone() for p in trainer.net.parameters()]
    trainer.train(verbose=False)
    after = list(trainer.net.parameters())
    assert len(trainer.history) == 1
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
    row = trainer.history[0]
    assert math.isfinite(row["policy_loss"]) and math.isfinite(row["value_loss"])


def test_phase3_config_has_single_institution_and_detecting_makers():
    cfg = phase3_config(institutions=1)
    assert cfg.population.institutions == 1
    assert sum(1 for m in cfg.population.market_makers if m.detector) == 2


def test_sweep_job_tag_and_ranking(tmp_path, monkeypatch):
    import json
    from scripts import sweep_rl
    monkeypatch.setattr(sweep_rl, "OUTPUTS", tmp_path)
    job = sweep_rl.SweepJob(lr=1e-4, entropy=0.01, epochs=4, size=0.20,
                            iterations=2, episodes=2, institutions=1, horizon=200,
                            burn_in=200, eval_every=2, eval_seeds=2, seed=0)
    assert job.tag == "lr1e-4_ent0.01_ep4_sz0.20"
    assert not job.is_done()
    job.out_dir.mkdir(parents=True)
    new_policy(seed=0).save(job.out_dir / "policy.pt")
    (job.out_dir / "history.json").write_text(json.dumps({
        "config": {"size": 0.20},
        "baseline_twap": {"shortfall": 2.82},
        "evals": [{"iteration": 2, "shortfall": 2.40, "completion": 0.99}],
        "history": [{"iteration": 1}, {"iteration": 2}],
    }))
    assert job.is_done()
    ranking = sweep_rl.rank_sweep(tmp_path / "rl" / "sweep")
    assert "0.20" in ranking and ranking["0.20"][0]["tag"] == job.tag
    assert "lr1e-4_ent0.01_ep4_sz0.20" in sweep_rl.format_ranking(ranking)
