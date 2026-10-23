"""E2 / E3 definitions and runner: pairing, determinism, summaries, serialisation."""

import json

import numpy as np

from order_book.experiments import (
    ADVERSARIES, E2Config, E3Config, EpisodeSpec, format_e2, format_e3, impact_paths, run_episode, run_specs,
    summarise_e2, summarise_e3,
)

QUICK = dict(horizon=200, decision_interval=5, burn_in=500, after=50)


def test_episode_spec_sides_and_counts():
    assert EpisodeSpec(1000, "twap_market", 0.1, "A0").resolved_side == "bid"
    assert EpisodeSpec(1001, "twap_market", 0.1, "A0").resolved_side == "ask"
    e3 = E3Config(strategies=("a", "b"), sizes=(0.1, 0.2), adversaries=("A0", "A4"), seeds=(1, 2, 3))
    assert len(e3.specs()) == 2 * 3 + 2 * 2 * 2 * 2 * 3
    e2 = E2Config(strategies=("a",), sizes=(0.1,), seeds=(1, 2))
    assert len(e2.specs()) == 2 and all(s.observers == e2.observers for s in e2.specs())
    assert set(ADVERSARIES) == {"A0", "A1-lenient", "A1-medium", "A1-strict", "A4"}


def test_episode_is_deterministic_and_serialisable():
    spec = EpisodeSpec(1000, "twap_passive", 0.1, "A1-medium", observers=("runs", "cusum"), **QUICK)
    a, b = run_episode(spec), run_episode(spec)
    assert a.shortfall_bps == b.shortfall_bps and a.executed == b.executed == a.quantity
    assert set(a.signals) == {"flow", "runs", "cusum"} and len(a.signals["runs"]) == spec.window
    assert a.executor_window[0] == 0 and 0 < a.executor_window[1] <= spec.window
    text = json.dumps(a.to_json())
    assert "base_config" not in json.loads(text)["spec"]


def test_visible_and_invisible_are_identical_under_a0_and_paired_under_a1():
    vis = run_episode(EpisodeSpec(1002, "twap_market", 0.2, "A0", "visible", **QUICK))
    inv = run_episode(EpisodeSpec(1002, "twap_market", 0.2, "A0", "invisible", **QUICK))
    assert vis.shortfall_bps == inv.shortfall_bps and vis.quantity == inv.quantity   # nobody reads detectors
    none = run_episode(EpisodeSpec(1002, "none", 0.0, "A0", "none", record_path=True, **QUICK))
    assert none.executed == 0 and none.shortfall_bps is None and len(none.mid_path) == none.spec["horizon"] + none.spec["after"]
    vis1 = run_episode(EpisodeSpec(1002, "twap_market", 0.2, "A1-medium", "visible", **QUICK))
    inv1 = run_episode(EpisodeSpec(1002, "twap_market", 0.2, "A1-medium", "invisible", **QUICK))
    assert vis1.quantity == inv1.quantity and vis1.arrival_mid == inv1.arrival_mid   # same burn-in, same task


def test_e3_runner_summary_and_gate_structure(tmp_path):
    cfg = E3Config(strategies=("twap_market", "immediate"), sizes=(0.1,), adversaries=("A0", "A1-medium"),
                   seeds=(1000, 1001), record_paths=True, **QUICK)
    with open(tmp_path / "episodes.jsonl", "w") as sink:
        results = run_specs(cfg.specs(), verbose=False, sink=sink)
    assert len(results) == len(cfg.specs()) == sum(1 for _ in open(tmp_path / "episodes.jsonl"))
    summary = summarise_e3(results)
    rows = summary["rows"]
    assert set(rows) == {"twap_market|0.1|A0", "twap_market|0.1|A1-medium", "immediate|0.1|A0", "immediate|0.1|A1-medium"}
    row = rows["twap_market|0.1|A1-medium"]
    assert row["episodes"] == 2 and set(row["leakage"]) >= {"mean", "low", "high", "p_greater", "p_holm", "n"}
    assert rows["twap_market|0.1|A0"]["leakage"]["mean"] == 0.0   # A0: visible == invisible exactly
    assert set(summary["gate"]["tests"]) == {"twap_market|0.1|A1-medium"} and isinstance(summary["gate"]["passed"], bool)
    assert "Phase 2 gate" in format_e3(summary)
    paths = impact_paths(results)
    assert len(paths) == 8 and all(  # 2 strategies x 2 adversaries x 2 seeds
        len(v) == cfg.horizon + cfg.after for v in paths.values())
    json.dumps(summary)


def test_e2_summary_has_every_observer_with_valid_scores():
    cfg = E2Config(strategies=("twap_market",), sizes=(0.1,), seeds=(1000, 1001), **QUICK)
    results = run_specs(cfg.specs(), verbose=False)
    summary = summarise_e2(results, cfg.threshold)
    row = summary["twap_market|0.1"]
    assert set(row) == {"flow", "flow_lenient", "flow_medium", "flow_strict", "runs", "cusum", "volume_clock", "regularity"}
    for m in row.values():
        auc, lo, hi = m["auc_abs"]
        assert 0.0 <= lo <= auc <= hi <= 1.0 and 0 <= m["delay_median"] <= cfg.horizon
        assert m["episodes"] == 2
        assert np.isnan(m["false_alarm_quiet"]) or 0.0 <= m["false_alarm_quiet"] <= 1.0
    assert "AUC|s|" in format_e2(summary)
