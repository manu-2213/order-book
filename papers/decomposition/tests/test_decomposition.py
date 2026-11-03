"""Shapley values, the decomposition's accounting, the arm tests, the protocol and the replication summaries."""

import json
from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from order_book.counterfactual import (
    ALL_CHANNELS, ARMS, CHARTISTS_BLIND, FULL, FUNDAMENTALISTS_BLIND, GHOST, MAKERS_BLIND, PREDATOR_CONTROL, PairedResult,
)
from papers.decomposition.decomposition import (
    DecompositionConfig, ProtocolConfig, abides_specs, bucci_specs, compare_executors, format_concavity,
    format_controls, format_decomposition, format_executors, format_protocol, run_paired_specs, shapley, shi_cartlidge_specs, summarise_abides,
    summarise_bucci, summarise_cell, summarise_concavity, summarise_decomposition, summarise_protocol,
    summarise_shi_cartlidge,
)
from order_book.experiments import CAPITAL_REGIMES, RegimeSpec
from order_book.presets import regime
from order_book.scoring import bootstrap_ratio, crn_diagnostics, paired_difference

QUICK = dict(burn_in=400, after=60)


def test_the_papers_tables_build_from_the_committed_summaries():
    from papers.decomposition.scripts import make_paper as paper
    tails = json.loads((paper.RESULTS / "tails.json").read_text())
    assert "tab:runaways" in paper.runaways_table(tails)
    assert "tab:causes" in paper.causes_table(paper.load("runaway_causes")["summary"])
    assert "tab:twap" in paper.twap_table(paper.load("e8_deep_cover"), paper.load("e8_capital_deep"))
    assert "tab:horizon" in paper.horizon_table(paper.load("e8_horizons"), paper.load("e8_capital_horizons"), tails["horizons"], tails["horizons_cap"])


def test_the_capital_regimes_change_only_what_speculators_and_the_noise_pool_hold():
    plain, rich = RegimeSpec("deep_cover", 1.0, 4).base(), CAPITAL_REGIMES[0].base()
    assert (rich.endowment.speculator, rich.endowment.noise) == (10 * plain.endowment.speculator, 10 * plain.endowment.noise)
    assert replace(rich, endowment=plain.endowment) == plain
    assert RegimeSpec("deep_cover", 1.0, 4).base() == regime(1.0, 4), "the default regime must not move, or stored results stop reproducing"


def game(players, value) -> dict:
    from itertools import combinations
    return {frozenset(c): value(frozenset(c)) for k in range(len(players) + 1) for c in combinations(players, k)}


# ----- Shapley ------------------------------------------------------------------------------ #

def test_shapley_of_an_additive_game_is_each_players_own_contribution():
    weights = {"I": 1.0, "P": -2.0, "D": 0.5, "A": 3.0}
    phi = shapley(game("IPDA", lambda s: 7.0 + sum(weights[p] for p in s)))
    assert phi == pytest.approx(weights)


def test_shapley_splits_an_interaction_equally_and_gives_a_dummy_nothing():
    # D and A only matter together: predation needs detection. I is a dummy.
    phi = shapley(game("IPDA", lambda s: 4.0 * ("D" in s and "A" in s) + 1.0 * ("P" in s)))
    assert phi["D"] == pytest.approx(2.0) and phi["A"] == pytest.approx(2.0)
    assert phi["P"] == pytest.approx(1.0) and phi["I"] == pytest.approx(0.0)


def test_shapley_is_the_average_marginal_contribution_over_every_ordering():
    rng = np.random.default_rng(0)
    values = game("IPDA", lambda s: 0.0)
    values = {k: float(rng.normal()) for k in values}
    phi = shapley(values)
    orderings = list(permutations("IPDA"))
    for player in "IPDA":
        total = 0.0
        for order in orderings:
            before = frozenset(order[:order.index(player)])
            total += values[before | {player}] - values[before]
        assert phi[player] == pytest.approx(total / len(orderings))
    assert sum(phi.values()) == pytest.approx(values[frozenset("IPDA")] - values[frozenset()])  # efficiency


def test_shapley_works_per_seed_on_arrays():
    values = game("IPDA", lambda s: np.arange(3.0) * len(s))
    phi = shapley(values)
    assert all(v == pytest.approx(np.arange(3.0) * 1.0) for v in phi.values())


# ----- the accounting of a cell, on results with known values ------------------------------- #

def fake(seed, channels, is_bps, drift=0.0, strategy="twap_market", size=0.2, q=None, peak=1.0, residual=0.5, end=1.0,
         mean=1.0, horizon=2_000, regime="r", rng="counter", lift=0.0) -> PairedResult:
    return PairedResult(seed, strategy, size, horizon, channels.key, regime, rng, "bid", 100, 100, 0, 0, 1.0, is_bps, drift, 0.5,
                        10, 2_000, 5.0, q if q is not None else size, mean, end, peak, residual, lift, 0.0, 0.1, None)


def additive_results(seeds=range(1000, 1012), effects=None, base=2.0, drift=0.0, noise=0.0, **kw):
    effects = effects or {"I": 1.0, "P": -0.5, "D": 0.3, "A": 0.2}
    rng = np.random.default_rng(0)
    out = []
    for seed in seeds:
        shared = rng.normal(0, 3.0)                 # a seed's natural drift, shared by all its worlds
        for c in ALL_CHANNELS:
            value = shared + base + sum(effects[p] for p in c.members) + rng.normal(0, noise)
            out.append(fake(seed, c, value, drift=shared + drift, **kw))
    return out


def test_cell_recovers_the_known_decomposition_and_the_accounting_identity_holds():
    results = additive_results(noise=0.0)
    summary = summarise_decomposition(results)
    row = summary["rows"]["r|twap_market|0.2|2000"]
    assert row["episodes"] == 12 and row["efficiency_gap"] < 1e-9
    assert row["reaction"]["mean"] == pytest.approx(1.0)                         # 1 - 0.5 + 0.3 + 0.2
    assert row["mechanical"][0] == pytest.approx(2.0)
    assert row["reaction_share"][0] == pytest.approx(1.0 / 3.0)
    phi = {k: v["mean"] for k, v in row["shapley"].items()}
    assert phi == pytest.approx({"inventory": 1.0, "price": -0.5, "detection": 0.3, "predation": 0.2})
    # IS = drift + mechanical + reaction, exactly.
    assert row["is"]["arm5"][0] == pytest.approx(row["natural_drift"][0] + row["mechanical"][0] + row["reaction"]["mean"])
    assert sum(phi.values()) == pytest.approx(row["reaction"]["mean"])
    for name, value in phi.items():
        assert row["shapley_share"][name][0] == pytest.approx(value / 3.0)
    # An additive game has order-free increments: both ladders read the same numbers.
    assert {k: v["mean"] for k, v in row["ladder"].items()} == pytest.approx(phi)
    assert {k: v["mean"] for k, v in row["ladder_plan_order"].items()} == pytest.approx(phi)
    assert "shapley" in row and row["shapley"]["inventory"]["p_holm"] >= row["shapley"]["inventory"]["p_two_sided"]
    json.dumps(summary)
    assert "phi I" in format_decomposition(summary)


def test_ladder_increments_depend_on_order_and_shapley_does_not():
    """The reason for the factorial: with an interaction the two chains disagree; Shapley stays one number."""
    results = []
    for seed in range(1000, 1008):
        for c in ALL_CHANNELS:
            value = 2.0 + (1.0 if {"P", "I"} <= c.members else 0.0)     # inventory matters only on the true book
            results.append(fake(seed, c, value))
    row = summarise_decomposition(results)["rows"]["r|twap_market|0.2|2000"]
    assert row["ladder"]["inventory"]["mean"] == pytest.approx(1.0)
    assert row["ladder_plan_order"]["inventory"]["mean"] == pytest.approx(0.0)
    assert row["shapley"]["inventory"]["mean"] == pytest.approx(0.5) == row["shapley"]["price"]["mean"]


def test_ladder_mode_needs_only_the_five_arms():
    results = [fake(seed, c, float(k)) for seed in range(1000, 1006) for k, c in ARMS.items()]
    row = summarise_decomposition(results)["rows"]["r|twap_market|0.2|2000"]
    assert "shapley" not in row
    assert [row["ladder"][k]["mean"] for k in ("price", "inventory", "detection", "predation")] == pytest.approx([1.0] * 4)
    assert row["reaction"]["mean"] == pytest.approx(4.0)
    format_decomposition({"rows": {"r|twap_market|0.2|2000": row}})     # falls back to the ladder columns


def test_controls_are_reported_when_present():
    results = additive_results()
    for seed in range(1000, 1012):
        value = next(r for r in results if r.seed == seed and r.channels == ARMS[2].key).shortfall_bps
        results.append(fake(seed, CHARTISTS_BLIND, value - 0.7))
        results.append(fake(seed, MAKERS_BLIND, value - 0.2))
        results.append(fake(seed, FUNDAMENTALISTS_BLIND, value + 0.4))
        value = next(r for r in results if r.seed == seed and r.channels == ARMS[4].key).shortfall_bps
        results.append(fake(seed, PREDATOR_CONTROL, value + 0.1))
    summary = summarise_decomposition(results)
    row = summary["rows"]["r|twap_market|0.2|2000"]
    split = row["price_split"]
    assert split["chartists"]["mean"] == pytest.approx(0.7) and split["makers"]["mean"] == pytest.approx(0.2)
    assert split["fundamentalists"]["mean"] == pytest.approx(-0.4)              # leaning against the move saves cost
    assert "chartists" in format_controls(summary)
    assert row["predator_on_natural_flow"]["mean"] == pytest.approx(0.1)
    assert row["predator_sighted"]["mean"] == pytest.approx(0.1)                # 0.2 total less 0.1 on natural flow


def test_a_seed_missing_from_any_arm_is_dropped_from_every_contrast_of_the_cell():
    results = additive_results(seeds=range(1000, 1006))
    results = [r for r in results if not (r.seed == 1003 and r.channels == "I-D-")]
    assert summarise_decomposition(results)["rows"]["r|twap_market|0.2|2000"]["episodes"] == 5
    assert summarise_cell({}) is None


# ----- concavity ---------------------------------------------------------------------------- #

def concave_results(n_seeds=16, reversion_full=0.6, reversion_ghost=0.2):
    rng = np.random.default_rng(1)
    out = []
    for seed in range(1000, 1000 + n_seeds):
        for size in (0.02, 0.05, 0.1, 0.2, 0.4):
            for k, c in ARMS.items():
                delta = 1.0 if k == 1 else 0.5           # ghost linear, full market square-root
                level = 20.0 * size ** delta * np.exp(rng.normal(0, 0.05))
                revert = reversion_ghost if k == 1 else reversion_full
                out.append(fake(seed, c, level, size=size, q=size, peak=level, mean=level, end=level, residual=level * (1 - revert)))
    return out


def test_concavity_recovers_exponents_by_arm_and_the_arm_1_vs_5_tests():
    summary = summarise_concavity(concave_results(), n_boot=200)
    cell = summary["r|twap_market|2000"]
    assert cell["arms"]["arm1"]["delta_impact_peak"] == pytest.approx(1.0, abs=0.05)
    assert cell["arms"]["arm5"]["delta_impact_peak"] == pytest.approx(0.5, abs=0.05)
    assert cell["arms"]["arm5"]["delta_cost"] == pytest.approx(0.5, abs=0.05)
    assert cell["arms"]["arm1"]["reversion"] == pytest.approx(0.2) and cell["arms"]["arm5"]["reversion"] == pytest.approx(0.6)
    test = cell["tests"]["delta_impact_peak_arm1_minus_arm5"]
    assert test["estimate"] == pytest.approx(0.5, abs=0.07) and test["low"] > 0
    reversion = cell["tests"]["reversion_arm5_minus_arm1"]
    assert reversion["estimate"] == pytest.approx(0.4) and reversion["low"] > 0
    assert cell["tests"]["reversion_arm5_minus_arm3"]["estimate"] == pytest.approx(0.0, abs=1e-9)   # same in the fake
    json.dumps(summary)
    assert "arm1" in format_concavity(summary)


def test_concavity_needs_more_than_one_size():
    results = [fake(seed, c, 1.0) for seed in range(1000, 1004) for c in ARMS.values()]
    assert summarise_concavity(results, n_boot=20) == {}


# ----- executors ---------------------------------------------------------------------------- #

def test_compare_executors_names_the_route_and_the_offset():
    results = []
    for seed in range(1000, 1010):
        drift = float(seed % 3)
        results += [fake(seed, GHOST, drift + 2.0, drift), fake(seed, FULL, drift + 2.0 + 1.0, drift)]    # the reference
        results += [fake(seed, GHOST, drift + 1.0, drift, strategy="paced"),                               # mech -1, react +0.5
                    fake(seed, FULL, drift + 1.0 + 1.5, drift, strategy="paced")]
        results += [fake(seed, GHOST, drift + 2.1, drift, strategy="stealthy"),                            # react -0.8
                    fake(seed, FULL, drift + 2.1 + 0.2, drift, strategy="stealthy")]
    comparison = compare_executors(results)
    paced = comparison["r|paced|0.2|2000"]
    assert paced["d_mechanical"]["mean"] == pytest.approx(-1.0) and paced["d_reaction"]["mean"] == pytest.approx(0.5)
    assert paced["cheaper"] and paced["route"] == "mechanical" and paced["offset"]
    stealthy = comparison["r|stealthy|0.2|2000"]
    assert stealthy["d_reaction"]["mean"] == pytest.approx(-0.8) and stealthy["route"] == "reaction" and stealthy["cheaper"]
    assert "r|twap_market|0.2|2000" not in comparison
    assert "cheaper via mechanical" in format_executors(comparison)
    json.dumps(comparison)


# ----- statistics ---------------------------------------------------------------------------- #

def test_paired_difference_has_a_two_sided_p_value():
    down = paired_difference(np.full(10, -1.0) + np.linspace(-0.1, 0.1, 10), np.zeros(10))
    assert down["p_two_sided"] < 0.01 and down["p_greater"] > 0.99
    none = paired_difference(np.zeros(5), np.zeros(5))
    assert none["p_two_sided"] == 1.0


def test_bootstrap_ratio_resamples_seeds_jointly():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    estimate, low, high = bootstrap_ratio(x, 2 * x)
    assert estimate == pytest.approx(0.5) and low == pytest.approx(0.5) and high == pytest.approx(0.5)
    assert np.isnan(bootstrap_ratio([], [])[0])
    assert np.isnan(bootstrap_ratio([1.0, 1.0], [0.0, 0.0])[0])


def test_crn_diagnostics_measures_variance_reduction_and_sharmas_condition():
    rng = np.random.default_rng(0)
    common = rng.normal(0, 3.0, 200)
    high = crn_diagnostics(common + rng.normal(0, 0.3, 200), common + rng.normal(0, 0.3, 200))
    assert high["variance_reduction"] > 20 and high["vr_low"] > 10 and high["sharma_holds"] and high["correlation"] > 0.9
    independent = crn_diagnostics(rng.normal(0, 3.0, 200), rng.normal(0, 3.0, 200))
    assert independent["variance_reduction"] == pytest.approx(1.0, abs=0.3) and not independent["sharma_holds"]
    anti = crn_diagnostics(common, -common + rng.normal(0, 0.3, 200))
    assert anti["variance_reduction"] < 1 and anti["correlation"] < 0 and not anti["sharma_holds"]
    # The two ways of stating the condition agree: VR > 1 exactly when the sample covariance is positive.
    assert (high["var_paired"] < high["var_independent"]) and (anti["var_paired"] > anti["var_independent"])
    assert np.isnan(crn_diagnostics([1.0], [2.0])["variance_reduction"])


def test_protocol_summary_compares_pairing_schemes():
    results = []
    rng = np.random.default_rng(0)
    for mode, noise in (("counter", 0.05), ("stateful", 1.0)):
        for seed in range(1000, 1020):
            shared = rng.normal(0, 3.0)
            for k, c in ARMS.items():
                results.append(fake(seed, c, shared + k + rng.normal(0, noise), rng=mode))
    summary = summarise_protocol(results)
    pooled = summary["pooled"]["total (5-1)"]
    assert pooled["counter"]["median_variance_reduction"] > pooled["stateful"]["median_variance_reduction"] > 1
    cell = summary["cells"]["twap_market|0.2|2000"]["detection (4-3)"]
    assert set(cell) == {"counter", "stateful"} and cell["counter"]["sharma_holds"]
    assert "Sharma" in format_protocol(summary)
    json.dumps(summary)


# ----- the replication summaries ------------------------------------------------------------- #

def test_shi_cartlidge_summary_tests_both_claims():
    results = []
    rng = np.random.default_rng(0)
    for seed in range(1000, 1016):
        shared = rng.normal(0, 3.0)
        for rate, reaction in ((0.01, 0.0), (0.05, 0.25), (0.1, 1.0), (0.2, 4.0)):      # reaction ~ rate^2
            name = f"pov@{rate:g}"
            results.append(fake(seed, GHOST, shared + 1.0, shared, strategy=name, size=rate))
            results.append(fake(seed, FULL, shared + 1.0 + reaction + rng.normal(0, 0.01), shared, strategy=name, size=rate))
    summary = summarise_shi_cartlidge(results)
    assert summary["zero_reaction_at_lowest_rate"] and summary["superlinear"]
    assert summary["growth_exponent_above"] == pytest.approx(2.0, abs=0.1)
    assert summary["growth_exponent_without_largest"] == pytest.approx(2.0, abs=0.1)
    assert summary["rows"]["pov@0.1"]["reaction"]["mean"] == pytest.approx(1.0, abs=0.05)
    assert not summarise_shi_cartlidge([]).get("superlinear")


def test_bucci_summary_finds_a_crossover():
    results = []
    for d, crossover in ((0, True), (4, False)):
        for seed in range(1000, 1006):
            for size in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
                # Linear below 0.1, square-root above; a pure square root has no crossover.
                level = (size / 0.1 if size <= 0.1 else (size / 0.1) ** 0.5) if crossover else (size / 0.1) ** 0.5
                results.append(fake(seed, ARMS[4], level, size=size, q=size, mean=level, regime=f"detecting={d}"))
    summary = summarise_bucci(results)
    assert summary["detecting=0"]["crossover"] and summary["detecting=0"]["delta_low"] == pytest.approx(1.0, abs=0.01)
    assert summary["detecting=4"]["delta_low"] == pytest.approx(summary["detecting=4"]["delta_high"], abs=0.01)
    assert not summary["detecting=4"]["crossover"]


def test_abides_summary_compares_lift_by_predictability():
    results = []
    for seed in range(1000, 1012):
        for strategy, lift in (("twap_market", 1.0), ("twap_random", 0.25)):
            results.append(fake(seed, ARMS[3], 1.0, strategy=strategy, lift=0.5))
            results.append(fake(seed, ARMS[5], 1.0, strategy=strategy, lift=0.5 + lift))
    summary = summarise_abides(results)
    assert summary["twap_market|0.2"]["lift_from_detection_and_predation"]["mean"] == pytest.approx(1.0)
    assert summary["twap_random|0.2"]["predictable_minus_this"]["mean"] == pytest.approx(0.75)
    assert "predictable_minus_this" not in summary["twap_market|0.2"]


def test_replication_specs_cover_their_grids():
    assert len(shi_cartlidge_specs(rates=(0.01, 0.1), seeds=(1, 2), **QUICK)) == 2 * 2 * 2
    specs = bucci_specs(detecting=(0, 4), sizes=(0.1, 0.2), seeds=(1,), **QUICK)
    assert len(specs) == 2 * 2 and {s.detecting for s in specs} == {0, 4} and all(s.channels == ARMS[4] for s in specs)
    assert specs[0].config().population.market_makers[0].detector is None
    assert specs[-1].config().population.market_makers[3].detector == "flow"
    assert len(abides_specs(strategies=("a", "b"), sizes=(0.2,), seeds=(1,), **QUICK)) == 2 * 2


# ----- end to end on the real simulator ------------------------------------------------------- #

def test_decomposition_config_counts_and_orders_seeds_outermost():
    cfg = DecompositionConfig(strategies=("a", "b"), sizes=(0.1, 0.2), horizons=(100, 200), seeds=(1, 2, 3), controls=False)
    specs = cfg.specs()
    assert len(specs) == 3 * 2 * 2 * 2 * 16
    assert [s.seed for s in specs[:64]] == [1] * 64                             # burn-in snapshots get reused
    assert len(cfg.channel_sets()) == 16
    assert len(replace(cfg, controls=True).channel_sets()) == 20
    assert len(replace(cfg, mode="ladder", controls=False).channel_sets()) == 5
    assert all(s.rng == "counter" for s in specs) and specs[0].regime == "deep_cover"
    assert specs[0].config().population.anticipators == 1                       # dormant in every arm without A


def test_end_to_end_on_the_simulator_the_accounting_is_exact_per_seed(tmp_path):
    cfg = DecompositionConfig(strategies=("twap_market",), sizes=(0.4,), horizons=(150,), seeds=(1000, 1001), **QUICK)
    with open(tmp_path / "episodes.jsonl", "w") as sink:
        results = run_paired_specs(cfg.specs(), verbose=False, sink=sink)
    assert len(results) == 20 * 2 == sum(1 for _ in open(tmp_path / "episodes.jsonl"))
    summary = summarise_decomposition(results)
    row = summary["rows"]["deep_cover|twap_market|0.4|150"]
    assert row["efficiency_gap"] < 1e-9
    assert row["is"]["arm5"][0] == pytest.approx(row["natural_drift"][0] + row["mechanical"][0] + row["reaction"]["mean"])
    assert row["completion"] > 0 and row["realised_q_over_v"] > 0
    json.dumps(summary)


def test_parallel_and_serial_runs_are_identical():
    cfg = DecompositionConfig(strategies=("twap_random",), sizes=(0.3,), horizons=(120,), seeds=(1000, 1001), mode="ladder",
                              controls=False, **QUICK)
    serial = run_paired_specs(cfg.specs(), verbose=False)
    parallel = run_paired_specs(cfg.specs(), workers=2, verbose=False)
    assert [r.shortfall_bps for r in serial] == [r.shortfall_bps for r in parallel]
    assert [r.impact_peak for r in serial] == [r.impact_peak for r in parallel]


def test_protocol_config_runs_both_schemes_on_the_same_seeds():
    cfg = ProtocolConfig(strategies=("twap_market",), sizes=(0.2,), seeds=(1, 2, 3))
    specs = cfg.specs()
    assert len(specs) == 2 * 3 * 5 and {s.rng for s in specs} == {"stateful", "counter"}


def test_a_grid_killed_part_way_resumes_without_repeating_runs(tmp_path):
    from papers.decomposition.decomposition import run_grid

    cfg = DecompositionConfig(strategies=("twap_market",), sizes=(0.3,), horizons=(120,), seeds=(1000, 1001), mode="ladder",
                              controls=False, **QUICK)
    specs = cfg.specs()
    path = tmp_path / "episodes.jsonl"
    whole = run_grid(specs, tmp_path / "whole.jsonl", verbose=False)
    run_grid(specs[:4], path, verbose=False)                      # the job dies here...
    with open(path, "a") as f:
        f.write('{"seed": 1000, "strat')                          # ...half way through a record
    resumed = run_grid(specs, path, verbose=False, resume=True)
    assert sum(1 for _ in open(path)) == len(specs) == len(resumed)
    assert sorted(r.key for r in resumed) == sorted(s.key for s in specs)
    by_key = {r.key: r.shortfall_bps for r in whole}
    assert all(by_key[r.key] == r.shortfall_bps for r in resumed)  # identical to an uninterrupted run
    assert summarise_decomposition(resumed)["rows"] == summarise_decomposition(whole)["rows"]


def test_shards_merge_into_exactly_what_one_job_would_have_produced(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)          # the retry message names the Slurm job when there is one
    from order_book.runner import IncompleteShards
    from papers.decomposition.decomposition import merge_grid, run_grid, run_shard

    cfg = DecompositionConfig(strategies=("twap_random",), sizes=(0.3,), horizons=(120,), seeds=(1000, 1001, 1002), mode="ladder",
                              controls=False, **QUICK)
    specs = cfg.specs()
    whole = run_grid(specs, tmp_path / "whole.jsonl", verbose=False)
    out = tmp_path / "sharded"
    run_shard(specs, cfg.seeds, 3, out, verbose=False)
    run_shard(specs, cfg.seeds, 1, out, verbose=False)
    with pytest.raises(IncompleteShards) as missing:                 # task 2 never ran
        merge_grid(specs, cfg.seeds, out)
    assert missing.value.missing == [2] and f"ARRAY=2 bash hpc/submit.sh {tmp_path.name} sharded" in str(missing.value)
    assert not (out / "episodes.jsonl").exists()                     # nothing half-merged is left behind
    run_shard(specs, cfg.seeds, 2, out, verbose=False)
    first = run_shard(specs, cfg.seeds, 1, out, verbose=False)       # resubmitting a finished task repeats nothing
    assert len(first) == len([s for s in specs if s.seed == 1000])
    merged = merge_grid(specs, cfg.seeds, out)
    assert [r.key for r in merged] == [s.key for s in specs]
    assert [r.shortfall_bps for r in merged] == [r.shortfall_bps for r in whole]
    assert summarise_decomposition(merged)["rows"] == summarise_decomposition(whole)["rows"]
    assert sum(1 for _ in open(out / "episodes.jsonl")) == len(specs)


def test_the_retry_message_names_the_array_job_from_inside_its_merge_job(tmp_path, monkeypatch):
    from order_book.runner import IncompleteShards

    out = tmp_path / "decomposition" / "any_folder"
    monkeypatch.setenv("SLURM_JOB_NAME", "e8_deep_cover_merge")
    assert "ARRAY=3,7 bash hpc/submit.sh decomposition e8_deep_cover\n" in str(IncompleteShards([3, 7], out)) + "\n"
    monkeypatch.delenv("SLURM_JOB_NAME")
    assert "ARRAY=3 bash hpc/submit.sh decomposition any_folder\n" in str(IncompleteShards([3], out)) + "\n"
