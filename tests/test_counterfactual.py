"""The two-world lockstep: identity without an executor, independence of the channels, exact nulls."""

from dataclasses import replace

import numpy as np
import pytest

from order_book.agents import MarketMaker, OrderAnticipator
from order_book.book import OrderBook
from order_book.config import MarketMakerConfig, OrderAnticipatorConfig, PopulationConfig, SimConfig, market_makers
from order_book.counterfactual import (
    ALL_CHANNELS, ARMS, CHARTISTS_BLIND, FULL, FUNDAMENTALISTS_BLIND, GHOST, MAKERS_BLIND, PLAN_LABEL_ORDER, PREDATOR_CONTROL,
    Channels, PairedRun, PairedSpec, burned_in, flow_coupling, run_paired,
)
from order_book.detectors import FlowDetector
from order_book.execution import ExecutionTask, ExecutorAgent, make_strategy
from order_book.experiments import EpisodeSpec, run_episode
from order_book.market import Market
from order_book.simulator import Simulator

QUICK = dict(horizon=200, burn_in=400, after=60)
NO_PREDATION = [c for c in ALL_CHANNELS if not c.predation] + [CHARTISTS_BLIND, MAKERS_BLIND, FUNDAMENTALISTS_BLIND]


def blob_of(spec: PairedSpec) -> tuple:
    return burned_in(spec.seed, spec.config(), spec.burn_in)


# ----- the structure of the channels -------------------------------------------------------- #

def test_channel_keys_span_the_factorial_and_the_arms_are_nested():
    assert len(ALL_CHANNELS) == 16 and len({c.key for c in ALL_CHANNELS}) == 16
    assert {c.members for c in ALL_CHANNELS} == {frozenset(s) for s in (
        "", "I", "P", "D", "A", "IP", "ID", "IA", "PD", "PA", "DA", "IPD", "IPA", "IDA", "PDA", "IPDA")}
    assert GHOST.key == "----" and FULL.key == "IPDA"
    for chain in (ARMS, PLAN_LABEL_ORDER):
        assert chain[1] == GHOST and chain[5] == FULL
        for k in range(2, 6):
            assert chain[k - 1].members < chain[k].members and len(chain[k].members - chain[k - 1].members) == 1
    # The plan's rows, by what each one does.
    assert ARMS[2] == Channels(price=True) and ARMS[3] == Channels(inventory=True, price=True)
    assert CHARTISTS_BLIND.key == "-P--c" and MAKERS_BLIND.key == "-P--m" and FUNDAMENTALISTS_BLIND.key == "-P--f"
    assert PREDATOR_CONTROL.key == "IPD-b"
    assert all(c.members == ARMS[2].members for c in (CHARTISTS_BLIND, MAKERS_BLIND, FUNDAMENTALISTS_BLIND))


# ----- identity: with nothing to react to, every configuration is Arm 0 -------------------- #

@pytest.mark.parametrize("rng", ["counter", "stateful"])
@pytest.mark.parametrize("channels", NO_PREDATION, ids=lambda c: c.key)
def test_with_no_executor_every_configuration_is_exactly_arm_0(channels, rng):
    blob, _ = blob_of(PairedSpec(1000, "twap_market", 0.2, burn_in=400, rng=rng))
    run = PairedRun(blob, channels)
    for _ in range(300):
        run.step()
        assert run.market.public.mid == run.reference.market.public.mid
        assert run.market.public.imbalance == run.reference.market.public.imbalance
    for side in ("bid", "ask"):
        assert run.market.book.levels(side) == run.reference.market.book.levels(side)
    if run.ghost is not None:
        assert not any(run.ghost.stats.values())               # the matching rule never had to intervene
    assert run.flow.signal == run.reference.detectors["flow"].signal


def test_the_reference_world_never_learns_the_executor_exists():
    """Arm 0 inside a paired run, in any arm, is the path of the same seed run with no executor at all."""
    spec = PairedSpec(1001, "twap_market", 0.4, **QUICK)
    blob, volume = blob_of(spec)
    alone = Simulator.restore(blob)
    expected = []
    for _ in range(spec.window):
        alone.step()
        expected.append(alone.market.public.mid)
    for channels in (GHOST, ARMS[2], FULL, Channels(True, False, True, True)):
        run = PairedRun(blob, channels)
        run.add_executor(ExecutionTask(side="ask", size_fraction=0.4, horizon=spec.horizon), make_strategy("twap_market"), volume)
        path = []
        for _ in range(spec.window):
            run.step()
            path.append(run.reference.market.public.mid)
        assert path == expected
        assert not run.reference.market.hidden_aggressors


@pytest.mark.parametrize("rng", ["counter", "stateful"])
def test_the_full_a1_arm_is_exactly_a_single_world_run_of_the_same_market(rng):
    """Lockstep adds nothing: the treated world of Arm 4 equals a plain Simulator with the same executor."""
    spec = PairedSpec(1002, "twap_passive", 0.3, ARMS[4], rng=rng, **QUICK)
    blob, volume = blob_of(spec)
    single = Simulator.restore(blob)
    task = ExecutionTask(side=spec.resolved_side, size_fraction=spec.size_fraction, horizon=spec.horizon)
    executor = single.add_external(ExecutorAgent, 50_000 * single.config.book.mid, 50_000, task=task,
                                   strategy=make_strategy(spec.strategy), start=single.market.step, expected_volume=volume)
    path = []
    for _ in range(spec.window):
        single.step()
        path.append(single.market.public.mid)
    paired = run_paired(replace(spec, record_path=True))
    assert paired.shortfall_bps == executor.result.shortfall_bps and paired.executed == executor.result.executed
    assert paired.orders_sent == executor.result.orders_sent
    single.check_conservation()


def test_paired_runs_are_deterministic_and_keep_the_same_task_in_every_world():
    spec = PairedSpec(1003, "twap_random", 0.3, **QUICK)
    results = [run_paired(replace(spec, channels=c)) for c in (GHOST, ARMS[3], FULL, FULL)]
    assert results[2].shortfall_bps == results[3].shortfall_bps and results[2].impact_peak == results[3].impact_peak
    assert len({r.quantity for r in results}) == 1 and all(r.side == "ask" for r in results)
    assert all(r.executed == r.quantity for r in results)
    assert all(r.q_over_v > 0 and np.isfinite(r.drift_bps) for r in results)


# ----- exact nulls: a channel with nothing in it contributes exactly nothing ---------------- #

def _world_values(base: SimConfig, strategy: str = "twap_market", size: float = 0.4, **kw) -> dict:
    spec = PairedSpec(1000, strategy, size, base_config=base, **QUICK, **kw)
    return {c.members: run_paired(replace(spec, channels=c)).shortfall_bps for c in ALL_CHANNELS}


def test_detection_contributes_exactly_zero_when_no_maker_reads_the_detector():
    values = _world_values(SimConfig(), detecting=0)
    for coalition, v in values.items():
        assert values[coalition | {"D"}] == v


def test_inventory_contributes_exactly_zero_when_makers_do_not_price_inventory():
    cfg = replace(SimConfig(), population=replace(SimConfig().population, market_makers=market_makers(4, 2, base=MarketMakerConfig(horizon=0))))
    values = _world_values(cfg)
    for coalition, v in values.items():
        assert values[coalition | {"I"}] == v


def test_predation_contributes_exactly_zero_when_the_anticipator_cannot_trade():
    cfg = replace(SimConfig(), anticipator=OrderAnticipatorConfig(rate=0.0))
    values = _world_values(cfg)
    for coalition, v in values.items():
        assert values[coalition | {"A"}] == v


def test_an_order_filled_before_anyone_can_react_costs_the_same_in_every_world():
    values = _world_values(SimConfig(), "immediate", 0.2)
    assert len(set(values.values())) == 1 and next(iter(values.values())) > 0


def test_hiding_from_detectors_changes_no_decision_of_the_executor():
    """Under A0 nobody reads a detector, so visible and invisible must match exactly for any strategy. Before
    the true-flow view this failed for footprint, which read its own hidden flow and made E3's A0 leakage nonzero."""
    common = dict(horizon=200, burn_in=500, after=50)
    for strategy in ("footprint", "twap_market"):
        visible = run_episode(EpisodeSpec(1000, strategy, 0.4, "A0", "visible", **common))
        invisible = run_episode(EpisodeSpec(1000, strategy, 0.4, "A0", "invisible", **common))
        assert visible.shortfall_bps == invisible.shortfall_bps and visible.orders_sent == invisible.orders_sent


# ----- each switch does what it says -------------------------------------------------------- #

def _maker_market():
    book = OrderBook(20, 10_000, 1)
    market = Market(book, analysis=False)
    return market


def _make_maker(market, cid, cfg=None, detector=None, shares=1000):
    return MarketMaker(cid, market, np.random.default_rng(cid), shares * 10_000, shares, cfg=cfg, detector=detector)


def test_inventory_freeze_points_the_skew_at_another_account_but_not_the_risk_limit():
    cfg = MarketMakerConfig(max_inventory=50, size=(5, 5))

    def bid_quote(inventory, frozen):
        market = _maker_market()
        maker = _make_maker(market, 0, cfg)
        maker.account.shares += inventory
        if frozen:
            flat = Market(OrderBook(20, 10_000, 1), analysis=False).open_account(99, 0, maker.target)
            maker.skew_account = flat
        maker.act()
        orders = {o["side"]: o["price"] for o in market.book.get_client_orders(0)}
        return orders

    flat = bid_quote(0, False)
    long_live = bid_quote(40, False)
    long_frozen = bid_quote(40, True)
    assert long_live["ask"] < flat["ask"] and long_live["bid"] < flat["bid"]    # long: leans down to sell
    assert long_frozen == flat                                                 # frozen: no lean at all
    assert "bid" not in bid_quote(60, True)                                     # the real holding still pulls the side


def test_price_channel_decides_whose_state_the_background_lives_on():
    spec = PairedSpec(1000, "twap_market", 0.3, **QUICK)
    blob, _ = blob_of(spec)
    ghost = PairedRun(blob, GHOST)
    assert ghost.treated is None and ghost.ghost is not None and ghost.reference.market.mirror is ghost.ghost
    live = PairedRun(blob, ARMS[2])
    assert live.treated is not None and live.reference.market.mirror is None
    makers = [(t, r) for t, r in zip(live.treated.agents, live.reference.agents) if isinstance(t, MarketMaker)]
    assert makers and all(t.skew_account is r.account for t, r in makers)        # I off
    live_inventory = PairedRun(blob, ARMS[3])
    assert all(a.skew_account is None for a in live_inventory.treated.agents if isinstance(a, MarketMaker))  # I on
    assert PairedRun(blob, ARMS[2]).flow.include_hidden is False                 # D off
    assert PairedRun(blob, ARMS[4]).flow.include_hidden is True                  # D on
    blind = PairedRun(blob, CHARTISTS_BLIND)
    assert blind.treated.speculators.momentum_public is blind.reference.market.public
    assert blind.treated.speculators.belief_public is None
    blind = PairedRun(blob, FUNDAMENTALISTS_BLIND)
    assert blind.treated.speculators.belief_public is blind.reference.market.public
    assert blind.treated.speculators.momentum_public is None
    blind = PairedRun(blob, MAKERS_BLIND)
    assert all(a.public_view is blind.reference.market.public for a in blind.treated.agents if isinstance(a, MarketMaker))
    assert all(a.public_view is None for a in live.treated.agents if isinstance(a, MarketMaker))


def test_the_makers_and_the_predator_read_detectors_that_switch_independently():
    spec = PairedSpec(1000, "twap_market", 0.3, **QUICK)
    blob, _ = blob_of(spec)

    def anticipator(run):
        pool = run.treated.agents if run.treated is not None else run._treated_agents
        return next(a for a in pool if isinstance(a, OrderAnticipator))

    both = PairedRun(blob, FULL)
    assert anticipator(both).active and anticipator(both).detector.include_hidden
    assert anticipator(both).detector is not both.flow
    only_predator = PairedRun(blob, Channels(price=True, predation=True))
    assert only_predator.flow.include_hidden is False and anticipator(only_predator).detector.include_hidden is True
    control = PairedRun(blob, PREDATOR_CONTROL)
    assert anticipator(control).active and anticipator(control).detector.include_hidden is False
    dormant = PairedRun(blob, ARMS[4])
    assert not anticipator(dormant).active
    # In the ghost the anticipator is a live trader in the copy, and Arm 0's stays dormant.
    ghost = PairedRun(blob, Channels(predation=True))
    assert anticipator(ghost).market is ghost.ghost
    assert anticipator(ghost).account is ghost.ghost.accounts[anticipator(ghost).cid]
    assert not next(a for a in ghost.reference.agents if isinstance(a, OrderAnticipator)).active


def test_a_dormant_anticipator_keeps_the_schedule_of_all_arms_identical():
    spec = PairedSpec(1000, "twap_market", 0.3, burn_in=400)
    blob, _ = blob_of(spec)
    r1, r2 = PairedRun(blob, GHOST), PairedRun(blob, FULL)
    assert len(r1.reference.agents) == len(r2.treated.agents)
    for _ in range(50):
        r1.step()
        r2.step()
    assert (r1.reference.schedule() == r2.reference.schedule()).all()
    assert (r2.treated.schedule() == r2.reference.schedule()).all()


def test_flow_detector_reads_the_flow_it_is_told_to():
    class Public:
        imbalance = 0.0
        imbalance_all = 1.0

    blind, sighted = FlowDetector(), FlowDetector(include_hidden=True)
    for _ in range(100):
        blind.update(Public())
        sighted.update(Public())
    assert blind.signal == 0.0 and sighted.signal == pytest.approx(1.0)


def test_shade_is_the_identity_with_the_makers_own_terms_and_follows_the_formula_otherwise():
    market = _maker_market()
    detector = FlowDetector()
    maker = _make_maker(market, 0, MarketMakerConfig(), detector)
    detector.signal = 0.4
    own_inventory = maker.account.shares - maker.target
    assert maker.shade("ask", 10_010, 10, own_inventory, 0.4) == (10_010, 10)
    assert maker.shade("bid", 9_990, 10, own_inventory, 0.4) == (9_990, 10)
    # A long maker leans down on both sides.
    assert maker.shade("ask", 10_010, 10, own_inventory + 100, 0.4)[0] < 10_010
    assert maker.shade("bid", 9_990, 10, own_inventory + 100, 0.4)[0] < 9_990
    # A stronger buy signal lifts the reservation, widens the ask and cuts the ask's size; the bid only moves.
    ask_price, ask_size = maker.shade("ask", 10_010, 10, own_inventory, 0.9)
    bid_price, bid_size = maker.shade("bid", 9_990, 10, own_inventory, 0.9)
    assert ask_price - 10_010 > bid_price - 9_990 > 0 and ask_size < 10 and bid_size == 10
    # A full cut removes the quote.
    cut = _make_maker(_maker_market(), 1, MarketMakerConfig(detect_size_cut=1.0), detector)
    assert cut.shade("ask", 10_010, 10, 0, 1.0) is None


def test_in_the_ghost_channels_only_shade_replayed_quotes_and_never_create_orders():
    spec = PairedSpec(1000, "twap_market", 0.4, horizon=600, burn_in=1_500, after=60)
    ghost, inventory, detection = (run_paired(replace(spec, channels=c)) for c in
                                   (GHOST, Channels(inventory=True), Channels(detection=True)))
    assert all(v.ghost_stats is not None for v in (ghost, inventory, detection))
    assert inventory.shortfall_bps != ghost.shortfall_bps and detection.shortfall_bps != ghost.shortfall_bps
    assert all(v.executed == v.quantity for v in (ghost, inventory, detection))
    # Shading moves quotes, so the rule has to clamp them, but the arms still trade the same order.
    assert inventory.ghost_stats["clamped"] > ghost.ghost_stats["clamped"]
    assert detection.ghost_stats["clamped"] > ghost.ghost_stats["clamped"]


def test_controls_run_on_the_same_task_as_their_arm():
    spec = PairedSpec(1000, "twap_market", 0.4, **QUICK)
    arm2 = run_paired(replace(spec, channels=ARMS[2]))
    for control in (CHARTISTS_BLIND, MAKERS_BLIND, FUNDAMENTALISTS_BLIND):
        result = run_paired(replace(spec, channels=control))
        assert result.channels == control.key and result.quantity == arm2.quantity and result.executed == result.quantity
    arm4 = run_paired(replace(spec, channels=ARMS[4]))
    blind = run_paired(replace(spec, channels=PREDATOR_CONTROL))
    assert blind.channels == "IPD-b" and blind.quantity == arm4.quantity


# ----- every executor, including the learned one --------------------------------------------- #

@pytest.mark.parametrize("strategy", ["twap_market", "twap_passive", "twap_random", "pov", "footprint", "mimic",
                                      "limit_market", "obizhaeva_wang", "almgren_chriss_1e-06", "immediate",
                                      "pov@0.05", "twap@size=0.5,timing=0.3,passive=0.5"])
def test_every_baseline_runs_to_completion_in_the_ghost_and_in_the_full_market(strategy):
    for channels in (GHOST, FULL):
        r = run_paired(PairedSpec(1000, strategy, 0.2, channels, **QUICK))
        assert r.executed == r.quantity and r.incomplete == 0 and r.shortfall_bps is not None


def test_the_learned_executor_runs_as_the_thirteenth(tmp_path):
    from order_book.execution import new_policy

    path = tmp_path / "policy.pt"
    new_policy(seed=0).save(path)
    for channels in (GHOST, FULL):
        r = run_paired(PairedSpec(1000, f"rl@{path}", 0.2, channels, **QUICK))
        assert r.executed == r.quantity and r.shortfall_bps is not None


def test_randomised_executors_draw_the_same_decisions_in_every_world():
    """The fixed key makes schedule randomness identical across arms, also between A1 and A4 populations."""
    spec = PairedSpec(1000, "mimic", 0.3, **QUICK)
    styles = set()
    for channels in (GHOST, ARMS[4], FULL):
        blob, volume = blob_of(replace(spec, channels=channels))
        run = PairedRun(blob, channels)
        strategy = make_strategy("mimic")
        run.add_executor(ExecutionTask(side="bid", size_fraction=0.3, horizon=spec.horizon), strategy, volume)
        run.step()
        styles.add((strategy.style, strategy.rate))
    assert len(styles) == 1


def test_coupling_is_perfect_without_divergence_and_keyed_draws_keep_more_of_it():
    idle = dict(horizon=200, burn_in=400, after=50)
    counter = flow_coupling(PairedSpec(1000, "twap_market", 0.6, ARMS[3], rng="counter", **idle))
    stateful = flow_coupling(PairedSpec(1000, "twap_market", 0.6, ARMS[3], rng="stateful", **idle))
    assert all(0.0 <= x <= 1.0 for x in counter + stateful)
    assert np.mean(counter) > np.mean(stateful)
    with pytest.raises(ValueError):
        flow_coupling(PairedSpec(1000, "twap_market", 0.2, GHOST, **idle))
