"""The dialled TWAP and the detection curve."""

import json
import numpy as np
import pytest

from papers.decomposition.detection_curve import (
    DetectionCurveConfig, format_detection_curve, run_detection_curve, score_episode, summarise_detection_curve,
)
from order_book.execution import Pov, make_strategy
from order_book.execution.strategies import DialTwap
from order_book.experiments import EpisodeSpec, run_episode

COMMON = dict(horizon=300, burn_in=500, after=60, rng="counter")
LONG = dict(horizon=800, burn_in=2_000, after=60, rng="counter")


def test_dials_at_zero_are_exactly_twap_market():
    a = run_episode(EpisodeSpec(1000, "twap_market", 0.2, "A1-medium", **COMMON))
    b = run_episode(EpisodeSpec(1000, "twap@", 0.2, "A1-medium", **COMMON))
    assert a.shortfall_bps == b.shortfall_bps and a.orders_sent == b.orders_sent


def test_make_strategy_parses_the_parametric_names():
    dial = make_strategy("twap@size=0.5,timing=0.3,passive=0.25")
    assert isinstance(dial, DialTwap) and (dial.size, dial.timing, dial.passive) == (0.5, 0.3, 0.25)
    assert make_strategy("twap@timing=0.4").size == 0.0
    pov = make_strategy("pov@0.05")
    assert isinstance(pov, Pov) and pov.rate == 0.05


def test_the_passive_dial_runs_from_twap_market_to_twap_passive():
    a = run_episode(EpisodeSpec(1000, "twap_passive", 0.2, "A1-medium", **LONG))
    b = run_episode(EpisodeSpec(1000, "twap@passive=1", 0.2, "A1-medium", **LONG))
    assert a.shortfall_bps == b.shortfall_bps and a.orders_sent == b.orders_sent and a.aggressive_share == b.aggressive_share


def test_timing_jitter_keeps_the_schedule_on_pace():
    """Skipped slices are made up in size, so the dial changes regularity and not how late the order trades."""
    from order_book.execution import ExecutionTask, ExecutorAgent
    from order_book.simulator import Simulator

    def share_done_at_midpoint(name, seed):
        sim = Simulator(seed, None, analysis=True)
        sim.run(1_000)
        task = ExecutionTask(side="bid", size_fraction=0.3, horizon=1_000)
        executor = sim.add_external(ExecutorAgent, 50_000 * 10_000, 50_000, task=task, strategy=make_strategy(name),
                                    start=sim.market.step, expected_volume=3.0)
        sim.run(1_000 + 60)
        result = executor.result
        assert result.executed == result.quantity and result.forced <= 0.1 * result.quantity
        return sum(f[2] for f in result.fills if f[0] < executor.start + 500) / result.quantity

    # Against the linear schedule, averaged because one seed's lumpy orders are noisy. Plain TWAP itself runs a little
    # ahead of linear, since small slices round up.
    seeds = range(1000, 1008)
    for name in ("twap@", "twap@timing=0.4", "twap@timing=0.8"):
        assert np.mean([share_done_at_midpoint(name, seed) for seed in seeds]) == pytest.approx(0.5, abs=0.07)
    assert np.mean([share_done_at_midpoint("twap_random", seed) for seed in seeds]) < 0.43   # the baseline back-loads


def test_each_dial_moves_only_its_own_behaviour():
    run = lambda name: run_episode(EpisodeSpec(1000, name, 0.3, "A1-medium", **LONG))
    plain, timing, half, full, size = (run("twap@"), run("twap@timing=0.6"), run("twap@passive=0.5"), run("twap@passive=1"),
                                       run("twap@size=1"))
    assert plain.aggressive_share == 1.0 == timing.aggressive_share == size.aggressive_share
    assert 0.0 < full.aggressive_share <= half.aggressive_share < 1.0   # more passive attempted, more filled passively
    assert timing.orders_sent < plain.orders_sent               # skips decisions
    assert size.shortfall_bps != plain.shortfall_bps
    for r in (plain, timing, half, full, size):
        assert r.executed == r.quantity                         # every dial still finishes the order
        assert r.forced <= 0.05 * r.quantity                    # by trading, not by a block at the horizon


def test_score_episode_reads_direction_latency_and_duty():
    spec = EpisodeSpec(1000, "twap@", 0.2, "A1-medium", horizon=10, after=0, burn_in=0)

    class Result:
        executor_window = (0, 10)
        side = "bid"
        shortfall_bps = 1.0
        quantity = 5
        aggressive_share = 1.0
        signals = {"up": [0, 0, 0.6, 0.6, 0, 0, 0, 0, 0, 0], "down": [-0.9] * 10, "weak": [0.4] * 10}

    scores = score_episode(spec, Result(), 0.5)["scores"]
    assert scores["up"] == {"fired": True, "latency": 2, "duty": 0.2, "signal": pytest.approx(0.12)}
    assert scores["down"]["signal"] == pytest.approx(-0.9)                         # signed by the executor's side
    assert scores["down"]["fired"] is False and scores["down"]["latency"] == 10    # the other direction does not count
    assert scores["weak"]["fired"] is False and scores["weak"]["duty"] == 0.0      # below the operating threshold
    Result.side = "ask"
    assert score_episode(spec, Result(), 0.5)["scores"]["down"]["latency"] == 0


def test_config_points_and_spec_counts():
    cfg = DetectionCurveConfig(thresholds=("medium",), participation=(0.1, 0.2), timing_jitter=(0.0, 0.5), size_jitter=(0.0, 0.5),
                               passive_share=(0.0, 0.5), base_size=0.2, seeds=(1, 2), observers=("flow_medium",))
    points = cfg.points()
    assert set(points) == {"participation", "timing_jitter", "size_jitter", "passive_share"}
    # The plain TWAP at 0.2 is shared by every axis, the rest are one each: 0.1 plus three dials.
    unique = {(s, q) for axis in points.values() for _, s, q in axis}
    assert len(unique) == 5
    assert len(cfg.specs()) == 2 * (1 + 5 * 2)
    assert all(s.rng == "counter" and s.observers == ("flow_medium",) for s in cfg.specs())


def test_end_to_end_curve_is_well_formed_and_participation_raises_detection(tmp_path):
    cfg = DetectionCurveConfig(thresholds=("lenient", "medium"), participation=(0.02, 1.0), timing_jitter=(0.0, 0.5),
                               size_jitter=(0.0, 0.5), passive_share=(0.0, 0.5), seeds=(1000, 1001, 1002, 1003),
                               observers=("flow_lenient", "flow_medium"), horizon=400, burn_in=800, after=60)
    records = run_detection_curve(cfg, workers=1, verbose=False, path=tmp_path / "episodes.jsonl")
    assert len(records) == len(cfg.specs()) == sum(1 for _ in open(tmp_path / "episodes.jsonl"))
    summary = summarise_detection_curve(records, cfg)
    assert set(summary) == {"lenient", "medium"}
    rows = summary["medium"]["axes"]["participation"]
    assert [r["value"] for r in rows] == [0.02, 1.0]
    for row in rows:
        for d in row["detectors"].values():
            assert 0.0 <= d["p_fire"][0] <= 1.0 and 0.0 <= d["duty"][0] <= 1.0
            assert -1.0 <= d["signal"][0] <= 1.0 and -1.0 <= d["signal_arm0"] <= 1.0
            assert 0 <= d["latency_median"] <= cfg.horizon + 50 and d["p_fire_arm0"] == d["p_fire_arm0"]
        assert row["episodes"] == 4 and "leakage" in row
    low, high = rows[0]["detectors"]["flow_medium"], rows[1]["detectors"]["flow_medium"]
    assert high["duty"][0] >= low["duty"][0]                     # trading more of the flow fires the detector more
    value = summary["medium"]["randomisation_value"]
    assert set(value) >= {"mean", "low", "high", "p_two_sided"}
    json.dumps(summary)
    text = format_detection_curve(summary)
    assert "timing_jitter" in text and "randomisation value" in text


def test_learned_detector_is_scored_when_weights_exist(tmp_path):
    from order_book.detectors.features import FEATURE_NAMES
    from order_book.detectors.learned import LearnedDetectorParams

    d = len(FEATURE_NAMES)
    path = tmp_path / "learned.npz"
    LearnedDetectorParams(np.zeros((d, 3)), np.zeros(3), np.zeros(d), np.ones(d)).save(path)
    cfg = DetectionCurveConfig(thresholds=("medium",), participation=(0.2,), timing_jitter=(0.0,), size_jitter=(0.0,),
                               passive_share=(0.0,), seeds=(1000,), observers=("flow_medium", "learned"), horizon=100, burn_in=300, after=60)
    records = run_detection_curve(cfg, weights=str(path), verbose=False)
    assert all("learned" in r["scores"] for r in records)


def test_a_curve_killed_part_way_resumes_without_repeating_episodes(tmp_path):
    cfg = DetectionCurveConfig(thresholds=("medium",), participation=(0.2,), timing_jitter=(0.0,), size_jitter=(0.0,),
                               passive_share=(0.0,), seeds=(1000, 1001), observers=("flow_medium",), horizon=100, burn_in=300, after=60)
    path = tmp_path / "episodes.jsonl"
    whole = run_detection_curve(cfg, verbose=False)
    run_detection_curve(cfg, verbose=False, path=path)
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:2]) + lines[2][:10])           # the job dies half way through the third record
    resumed = run_detection_curve(cfg, verbose=False, path=path, resume=True)
    assert len(resumed) == len(whole) == sum(1 for _ in open(path))
    assert sorted(json.dumps(r, sort_keys=True) for r in resumed) == sorted(json.dumps(r, sort_keys=True) for r in whole)


def test_curve_shards_merge_into_exactly_what_one_job_would_have_produced(tmp_path):
    from order_book.runner import IncompleteShards
    from papers.decomposition.detection_curve import merge_curve, run_curve_shard

    cfg = DetectionCurveConfig(thresholds=("medium",), participation=(0.2,), timing_jitter=(0.0,), size_jitter=(0.0,),
                               passive_share=(0.0,), seeds=(1000, 1001), observers=("flow_medium",), horizon=100, burn_in=300, after=60)
    whole = run_detection_curve(cfg, verbose=False)
    run_curve_shard(cfg, 2, tmp_path, verbose=False)
    with pytest.raises(IncompleteShards) as missing:
        merge_curve(cfg, tmp_path)
    assert missing.value.missing == [1]
    run_curve_shard(cfg, 1, tmp_path, verbose=False)
    merged = merge_curve(cfg, tmp_path)
    assert sorted(json.dumps(r, sort_keys=True) for r in merged) == sorted(json.dumps(r, sort_keys=True) for r in whole)
