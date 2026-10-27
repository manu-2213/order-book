"""E2 / E3 definitions and runner: pairing, determinism, summaries, serialisation."""

import json

import numpy as np

from order_book.experiments import (
    ADVERSARIES, DEFAULT_REGIMES, E2Config, E3Config, E7Config, EpisodeSpec, RegimeSpec, format_e2, format_e3,
    format_e7, impact_paths, run_episode, run_specs, summarise_e2, summarise_e3, summarise_e7,
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


def test_e2_summary_handles_episodes_without_institutional_activity_without_warning():
    """When no institution is active outside the executor window, per-detector scores are undefined;
    the summary must return nan cleanly, never raise ``RuntimeWarning: Mean of empty slice``."""
    import warnings

    spec = EpisodeSpec(1000, "twap_market", 0.1, "A1-medium", observers=("runs",), **QUICK)
    r = run_episode(spec)
    r.institutional_windows = []                 # force the "no cover" case
    r.executor_window = (0, len(r.signals["runs"]))   # executor active for the whole window
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # any RuntimeWarning becomes a failure
        summary = summarise_e2([r])
    m = summary["twap_market|0.1"]["runs"]
    assert np.isnan(m["auc_institutional"]) and np.isnan(m["false_alarm_institutional"])
    assert np.isnan(m["false_alarm_quiet"])      # no negative (non-executor) steps either
    assert 0 <= m["delay_median"] <= spec.window
    json.dumps(summary)


def test_e7_grid_spans_regimes_and_leakage_grows_as_cover_falls():
    """A thin, no-cover regime makes the executor a large share of flow; the flow detector it faces must
    fire more and the leakage channel must be larger than in the deep-cover default."""
    regimes = (RegimeSpec("deep_cover", 1.0, 4), RegimeSpec("thin_no_cover", 0.4, 0, mm_size_scale=0.5))
    cfg = E7Config(regimes=regimes, strategies=("twap_market",), sizes=(0.4,), adversaries=("A1-medium",),
                   seeds=(1000, 1001, 1002, 1003), **QUICK)
    specs = cfg.specs()
    # 2 regimes x [ (none: 1 adv x 4 seeds) + (1 strat x 1 size x 1 adv x 2 vis x 4 seeds) ]
    assert len(specs) == 2 * (4 + 8)
    assert {s.regime for s in specs} == {"deep_cover", "thin_no_cover"}
    assert all(s.base_config is not None for s in specs)
    results = run_specs(specs, verbose=False)
    summary = summarise_e7(results)
    rows = summary["rows"]
    assert set(rows) == {"deep_cover|twap_market|0.4|A1-medium", "thin_no_cover|twap_market|0.4|A1-medium"}
    deep = rows["deep_cover|twap_market|0.4|A1-medium"]
    thin = rows["thin_no_cover|twap_market|0.4|A1-medium"]
    assert deep["episodes"] == thin["episodes"] == 4
    assert thin["volume_per_step"] < deep["volume_per_step"]      # thinner market
    assert thin["quantity"] < deep["quantity"]                   # same Q/V, smaller absolute order
    assert set(deep["leakage"]) >= {"mean", "low", "high", "p_greater", "n"}
    # the executor moves the detector it faces more when there is no institutional cover to hide in
    assert thin["flow_firing_channel"] >= deep["flow_firing_channel"]
    assert "regime" in format_e7(summary) and "thin_no_cover" in format_e7(summary)
    json.dumps(summary)


def test_default_regimes_build_valid_configs():
    from order_book.presets import adversary
    assert DEFAULT_REGIMES[0].name == "deep_cover" and DEFAULT_REGIMES[-1].institutions == 0
    for rs in DEFAULT_REGIMES:
        cfg = adversary("A4", base=rs.base())
        assert cfg.population.institutions == rs.institutions
        assert len(cfg.population.market_makers) == rs.makers


def test_strength_sweep_builds_prebuilt_configs_and_wires_response():
    from order_book.experiments import DETECTION_STRENGTHS, E7StrengthConfig, RegimeSpec, strength_config
    weak, strong = DETECTION_STRENGTHS[0], DETECTION_STRENGTHS[2]
    regime = RegimeSpec("low_cover", 1.0, 1)
    cfg_weak = strength_config(regime, weak, "A1")
    cfg_strong = strength_config(regime, strong, "A1")
    # strong: all 4 makers detect with a bigger response; weak: 2 makers, default response
    assert [m.detector for m in cfg_weak.population.market_makers] == ["flow", "flow", None, None]
    assert [m.detector for m in cfg_strong.population.market_makers] == ["flow"] * 4
    assert cfg_strong.population.market_makers[0].detect_skew == 8.0
    assert cfg_strong.flow_detector.threshold < cfg_weak.flow_detector.threshold  # lenient < strict
    assert cfg_weak.population.institutions == 1  # regime preserved
    specs = E7StrengthConfig(strengths=(weak, strong), strategies=("twap_market",), sizes=(0.4,),
                             seeds=(1000, 1001), **QUICK).specs()
    assert len(specs) == 2 * 1 * 1 * 2 * 2          # strengths x strat x size x {vis,invis} x seeds
    assert all(s.prebuilt_config and s.base_config is not None for s in specs)
    assert {s.regime for s in specs} == {"low_cover/weak", "low_cover/strong"}


def test_strength_sweep_runs_and_summarises():
    from order_book.experiments import DETECTION_STRENGTHS, E7StrengthConfig
    cfg = E7StrengthConfig(strengths=(DETECTION_STRENGTHS[1], DETECTION_STRENGTHS[3]), strategies=("twap_market",),
                           sizes=(0.4,), seeds=(1000, 1001, 1002, 1003), **QUICK)
    results = run_specs(cfg.specs(), verbose=False)
    summary = summarise_e7(results)
    assert set(summary["rows"]) == {"low_cover/default|twap_market|0.4|A1", "low_cover/max|twap_market|0.4|A1"}
    for r in summary["rows"].values():
        assert r["episodes"] == 4 and "mean" in r["leakage"]
    assert "low_cover/max" in format_e7(summary)
    json.dumps(summary)

