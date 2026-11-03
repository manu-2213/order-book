"""Decomposing a metaorder's cost into what it consumed and what the market did about it.

Each run of ``order_book.counterfactual`` prices one executor in one world; this
module turns the grid of worlds into the claim. For one seed, with v(S) the
implementation shortfall when the reaction channels S exist,

    natural drift   D         what the same fills would have cost at Arm 0's mid
    mechanical      v({}) - D  liquidity consumed in the ghost book, net of drift
    reaction        v(all) - v({})
    share           reaction / (v(all) - D)

so that IS = drift + mechanical + reaction exactly. The reaction is split two
ways: along the plan's ladder (``ARMS``), and by Shapley values over the 16
cells of the factorial, because ladder increments depend on the order channels
join. Shapley values add up to the reaction exactly, per seed:

    phi_c = sum over S not containing c of |S|! (3 - |S|)! / 4! x (v(S + c) - v(S))

Inference is over seeds: paired-difference bootstrap CIs, two-sided p-values
(a channel may reduce cost) and Holm correction across all cells of a grid.
Also here: the per-arm impact exponent and reversion (does reaction produce the
concavity?), the comparison of executors by where they pay, the protocol's
variance-reduction report and the three replication hooks.
"""

import json
import time
from dataclasses import dataclass
from math import factorial
from pathlib import Path

import numpy as np

from order_book.counterfactual import (
    ALL_CHANNELS, ARMS, CHANNEL_NAMES, CHARTISTS_BLIND, FULL, FUNDAMENTALISTS_BLIND, GHOST, MAKERS_BLIND, PLAN_LABEL_ORDER,
    PREDATOR_CONTROL, Channels, PairedResult, PairedSpec, run_paired,
)
from order_book.execution import Calibration, all_baselines
from order_book.experiments import RegimeSpec, SIZES, TEST_SEEDS
from order_book.runner import IncompleteShards, iter_jobs, load_results, load_shards, missing_shards, seed_of_shard, shard_dir
from order_book.scoring import bootstrap_ci, bootstrap_ratio, crn_diagnostics, holm, paired_difference
from order_book.validation import fit_power_law

CHANNEL_ORDER = "IPDA"
LADDER_STEPS = {2: "price", 3: "inventory", 4: "detection", 5: "predation"}
PLAN_LABEL_STEPS = {2: "inventory", 3: "price", 4: "detection", 5: "predation"}
CONTROLS = (CHARTISTS_BLIND, MAKERS_BLIND, FUNDAMENTALISTS_BLIND, PREDATOR_CONTROL)
PRICE_GROUPS = {"chartists": CHARTISTS_BLIND, "makers": MAKERS_BLIND, "fundamentalists": FUNDAMENTALISTS_BLIND}
DEEP_COVER = RegimeSpec("deep_cover", 1.0, 4)


def shapley(values: dict, players: str = CHANNEL_ORDER) -> dict:
    """Shapley value of each player from ``values`` keyed by frozenset of players; numbers or arrays per seed."""
    n = len(players)
    out = {}
    for player in players:
        others = [p for p in players if p != player]
        total = 0.0
        for mask in range(1 << (n - 1)):
            coalition = frozenset(p for i, p in enumerate(others) if mask >> i & 1)
            weight = factorial(len(coalition)) * factorial(n - 1 - len(coalition)) / factorial(n)
            total = total + weight * (values[coalition | {player}] - values[coalition])
        out[player] = total
    return out


# ----- the grid ---------------------------------------------------------------------------- #



@dataclass(frozen=True)
class DecompositionConfig:
    """E8: the channel decomposition over executors, sizes and horizons.

    ``mode``: ``"shapley"`` runs all 16 cells per seed, ``"ladder"`` only the five nested arms. ``controls``
    adds the four cells outside the factorial (chartists, makers or fundamentalists blind to the price move;
    blind anticipator). Seeds are the outer loop so each worker reuses its burn-in snapshot. Learned executors
    are written ``"rl@<path>"``.
    """

    name: str = "E8"
    strategies: tuple[str, ...] = tuple(all_baselines())
    sizes: tuple[float, ...] = (0.02,) + SIZES + (0.40,)
    horizons: tuple[int, ...] = (2_000,)
    seeds: tuple[int, ...] = tuple(TEST_SEEDS)
    regime: RegimeSpec = DEEP_COVER
    threshold: str = "medium"
    mode: str = "shapley"
    controls: bool = True
    decision_interval: int = 5
    burn_in: int = 2_000
    after: int = 500
    rng: str = "counter"

    def channel_sets(self) -> tuple[Channels, ...]:
        cells = ALL_CHANNELS if self.mode == "shapley" else tuple(ARMS.values())
        return cells + (CONTROLS if self.controls else ())

    def specs(self) -> list[PairedSpec]:
        base = self.regime.base()
        common = dict(decision_interval=self.decision_interval, burn_in=self.burn_in, after=self.after,
                      threshold=self.threshold, base_config=base, regime=self.regime.name, rng=self.rng)
        return [PairedSpec(seed, strategy, size, channels, horizon=horizon, **common)
                for seed in self.seeds for strategy in self.strategies for size in self.sizes
                for horizon in self.horizons for channels in self.channel_sets()]


def _run(args):
    spec, calibration = args
    return run_paired(spec, calibration)


def run_paired_specs(specs: list[PairedSpec], calibration: Calibration | None = None, workers: int = 1,
                     verbose: bool = True, label: str = "E8", sink=None) -> list[PairedResult]:
    """Run paired episodes (in parallel processes when ``workers > 1``), one progress line each, JSONL to ``sink``."""
    results = []
    t0 = time.perf_counter()
    jobs = [(s, calibration) for s in specs]
    chunksize = max(1, min(18, len(jobs) // (workers * 4) or 1))
    for i, r in enumerate(iter_jobs(_run, jobs, workers, chunksize), 1):
        results.append(r)
        if sink is not None:
            sink.write(json.dumps(r.to_json()) + "\n")
            sink.flush()
        if verbose:
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (len(specs) - i)
            is_txt = "   n/a" if r.shortfall_bps is None else f"{r.shortfall_bps:+7.2f}"
            print(f"[{label} {i:5d}/{len(specs)}] seed={r.seed} {r.strategy:<22s} Q/V={r.size_fraction:.3f} T={r.horizon} "
                  f"{r.channels:<6s} IS={is_txt} drift={r.drift_bps:+7.2f} real Q/V={r.q_over_v:.3f} "
                  f"(elapsed {elapsed:.0f}s, eta {eta:.0f}s)", flush=True)
    return results


def run_grid(specs: list[PairedSpec], path: Path, calibration: Calibration | None = None, workers: int = 1,
             verbose: bool = True, label: str = "E8", resume: bool = False) -> list[PairedResult]:
    """``run_paired_specs`` logging to ``path``. With ``resume``, runs already in the log are kept, not repeated."""
    done = load_results(path, PairedResult.from_json) if resume else []
    finished = {r.key for r in done}
    todo = [s for s in specs if s.key not in finished]
    if resume and verbose:
        print(f"[{label}] resuming: {len(done)} runs already in {path}, {len(todo)} to go", flush=True)
    with open(path, "a" if resume else "w") as sink:
        return done + run_paired_specs(todo, calibration, workers, verbose, label, sink)


def run_shard(specs: list[PairedSpec], seeds, shard: int, out: Path, calibration: Calibration | None = None, workers: int = 1,
              verbose: bool = True, label: str = "E8", log: str = "episodes.jsonl") -> list[PairedResult]:
    """One seed of a grid, in its own folder, always resuming so that a task killed at its time limit can be resubmitted."""
    seed = seed_of_shard(seeds, shard)
    directory = shard_dir(out, shard)
    directory.mkdir(parents=True, exist_ok=True)
    return run_grid([s for s in specs if s.seed == seed], directory / log, calibration, workers, verbose, f"{label} {shard}", True)


def merge_grid(specs: list[PairedSpec], seeds, out: Path, log: str = "episodes.jsonl") -> list[PairedResult]:
    """Every run of a sharded grid in spec order, also written to ``out/<log>``. Raises ``IncompleteShards`` if any is missing."""
    by_key = {r.key: r for r in load_shards(out, PairedResult.from_json, log)}
    missing = missing_shards({s.key: s.seed for s in specs}, set(by_key), seeds)
    if missing:
        raise IncompleteShards(missing, out)
    results = [by_key[s.key] for s in specs]
    with open(Path(out) / log, "w") as sink:
        for r in results:
            sink.write(json.dumps(r.to_json()) + "\n")
    return results


# ----- one cell ---------------------------------------------------------------------------- #

def _cells(results: list[PairedResult]) -> dict:
    cells: dict = {}
    for r in results:
        cells.setdefault((r.regime, r.strategy, r.size_fraction, r.horizon), {}).setdefault(r.channels, {})[r.seed] = r
    return cells


def _aligned(cell: dict, *keys: str) -> list[int]:
    """Seeds on which every named arm completed with a shortfall, so that all contrasts are paired."""
    seeds = None
    for key in keys:
        have = {s for s, r in cell.get(key, {}).items() if r.shortfall_bps is not None}
        seeds = have if seeds is None else seeds & have
    return sorted(seeds or ())


def _values(cell: dict, key: str, seeds: list[int], field: str = "shortfall_bps") -> np.ndarray:
    return np.array([getattr(cell[key][s], field) for s in seeds], float)


def _ladder(cell: dict, arms: dict, names: dict, seeds: list[int]) -> dict:
    return {names[k]: paired_difference(_values(cell, arms[k].key, seeds), _values(cell, arms[k - 1].key, seeds))
            for k in names}


def summarise_cell(cell: dict) -> dict | None:
    """All statistics of one (regime, strategy, size, horizon) cell, from its runs per channel key and seed."""
    ghost, full = GHOST.key, FULL.key
    if ghost not in cell or full not in cell:
        return None
    ladder_keys = [ARMS[k].key for k in ARMS]
    if not all(k in cell for k in ladder_keys):
        return None
    factorial_keys = [c.key for c in ALL_CHANNELS]
    has_factorial = all(k in cell for k in factorial_keys)
    seeds = _aligned(cell, *(factorial_keys if has_factorial else ladder_keys))
    if not seeds:
        return None
    is_full, is_ghost = _values(cell, full, seeds), _values(cell, ghost, seeds)
    drift = _values(cell, ghost, seeds, "drift_bps")
    row: dict = {
        "episodes": len(seeds),
        "quantity": float(np.mean([cell[full][s].quantity for s in seeds])),
        "realised_q_over_v": float(np.mean(_values(cell, full, seeds, "q_over_v"))),
        "completion": float(np.mean(_values(cell, full, seeds, "completion"))),
        "forced_share": float(np.mean([cell[full][s].forced / max(cell[full][s].quantity, 1) for s in seeds])),
        "aggressive_share": float(np.mean(_values(cell, full, seeds, "aggressive_share"))),
        "is": {f"arm{k}": bootstrap_ci(_values(cell, ARMS[k].key, seeds)) for k in ARMS},
        "natural_drift": bootstrap_ci(drift),
        "mechanical": bootstrap_ci(is_ghost - drift),
        "reaction": paired_difference(is_full, is_ghost),
        "reaction_share": bootstrap_ratio(is_full - is_ghost, is_full - drift),
        "ladder": _ladder(cell, ARMS, LADDER_STEPS, seeds),
    }
    row["by_arm"] = {"flow_firing": {f"arm{k}": float(np.mean(_values(cell, ARMS[k].key, seeds, "flow_firing"))) for k in ARMS},
                     "quote_lift": {f"arm{k}": float(np.mean(_values(cell, ARMS[k].key, seeds, "quote_lift"))) for k in ARMS}}
    if has_factorial:
        row["ladder_plan_order"] = _ladder(cell, PLAN_LABEL_ORDER, PLAN_LABEL_STEPS, seeds)
        values = {c.members: _values(cell, c.key, seeds) for c in ALL_CHANNELS}
        phi = shapley(values)
        reaction = is_full - is_ghost
        row["shapley"] = {CHANNEL_NAMES[c]: paired_difference(phi[c], np.zeros(len(seeds))) for c in CHANNEL_ORDER}
        row["shapley_share"] = {CHANNEL_NAMES[c]: bootstrap_ratio(phi[c], is_full - drift) for c in CHANNEL_ORDER}
        row["efficiency_gap"] = float(np.abs(sum(phi.values()) - reaction).max())
    price_split = {}
    for group, control in PRICE_GROUPS.items():
        if control.key in cell:
            keep = _aligned(cell, control.key, ARMS[2].key)
            price_split[group] = paired_difference(_values(cell, ARMS[2].key, keep), _values(cell, control.key, keep))
    if price_split:
        row["price_split"] = price_split
    if PREDATOR_CONTROL.key in cell:
        keep = _aligned(cell, PREDATOR_CONTROL.key, ARMS[4].key, ARMS[5].key)
        row["predator_on_natural_flow"] = paired_difference(_values(cell, PREDATOR_CONTROL.key, keep),
                                                            _values(cell, ARMS[4].key, keep))
        row["predator_sighted"] = paired_difference(_values(cell, ARMS[5].key, keep),
                                                    _values(cell, PREDATOR_CONTROL.key, keep))
    return row


def summarise_decomposition(results: list[PairedResult]) -> dict:
    """Every cell's statistics, with Holm-adjusted p-values across the grid for each family of tests."""
    rows = {}
    for (regime, strategy, size, horizon), cell in sorted(_cells(results).items()):
        row = summarise_cell(cell)
        if row is not None:
            rows[f"{regime}|{strategy}|{size}|{horizon}"] = row
    tests = [(row["reaction"], "p_holm") for row in rows.values()]
    for row in rows.values():
        tests += [(test, "p_holm") for test in row.get("shapley", {}).values()]
    pvals = [test["p_two_sided"] for test, _ in tests]
    for (test, name), p in zip(tests, holm(pvals) if pvals else []):
        test[name] = p
    return {"rows": rows}


def format_controls(summary: dict) -> str:
    """What the controls say: who is in the price arm's increment, and the anticipator with and without sight."""
    lines = [f"{'regime':<11}{'strategy':<24}{'Q/V':>5}{'T':>6}{'price':>7} | {'chartists':>22}{'makers':>22}{'fundamentalists':>22} | "
             f"{'predator natural':>17}{'sighted':>9}"]
    for key, r in summary["rows"].items():
        regime, strategy, size, horizon = key.split("|")
        split = r.get("price_split")
        if not split:
            continue
        cells = "".join(f"{split[g]['mean']:>7.2f} [{split[g]['low']:+.2f},{split[g]['high']:+.2f}]"[:22].rjust(22) for g in PRICE_GROUPS if g in split)
        natural, sighted = r.get("predator_on_natural_flow"), r.get("predator_sighted")
        tail = f"{natural['mean']:>17.2f}{sighted['mean']:>9.2f}" if natural else ""
        lines.append(f"{regime:<11}{strategy[:23]:<24}{float(size):>5.2f}{int(horizon):>6}{r['ladder']['price']['mean']:>7.2f} | {cells} | {tail}")
    return "\n".join(lines)


def format_decomposition(summary: dict) -> str:
    """One row per cell; the last four columns are ladder increments for a cell without the factorial."""
    header = (f"{'regime':<11}{'strategy':<24}{'Q/V':>5}{'real':>6}{'T':>6}{'n':>4}{'IS':>7}{'drift':>7}{'mech':>6}{'react':>7}"
              f"{'[95% CI]':>16}{'pHolm':>6}{'share':>7} | {'phi I':>6}{'phi P':>6}{'phi D':>6}{'phi A':>6}")
    lines = [header]
    for key, r in summary["rows"].items():
        regime, strategy, size, horizon = key.split("|")
        react = r["reaction"]
        phi = r.get("shapley")
        phi_txt = (" ".join(f"{phi[CHANNEL_NAMES[c]]['mean']:>5.2f}" for c in CHANNEL_ORDER) if phi
                   else " ".join(f"{r['ladder'][CHANNEL_NAMES[c]]['mean']:>5.2f}" for c in CHANNEL_ORDER))
        lines.append(f"{regime:<11}{strategy[:23]:<24}{float(size):>5.2f}{r['realised_q_over_v']:>6.3f}{int(horizon):>6}{r['episodes']:>4}"
                     f"{r['is']['arm5'][0]:>7.2f}{r['natural_drift'][0]:>7.2f}{r['mechanical'][0]:>6.2f}{react['mean']:>7.2f}"
                     f"  [{react['low']:+.2f},{react['high']:+.2f}]{react.get('p_holm', float('nan')):>6.2f}"
                     f"{r['reaction_share'][0]:>7.2f} | {phi_txt}")
    return "\n".join(lines)


# ----- output 2: does reaction produce the concavity? -------------------------------------- #

IMPACT_MEASURES = ("impact_mean", "impact_peak", "impact_end", "cost")


def _arm_points(results: list[PairedResult], key: str) -> list:
    return [r for r in results if r.channels == key and r.shortfall_bps is not None]


def _measure(r: PairedResult, measure: str) -> float:
    return r.shortfall_bps - r.drift_bps if measure == "cost" else getattr(r, measure)


def _fit(rows: list, measure: str, bins: int) -> float:
    return fit_power_law([r.q_over_v for r in rows], [_measure(r, measure) for r in rows], bins=bins)["delta"]


def _reversion(rows: list) -> float:
    """Share of the peak impact gone ``after`` steps past completion, pooled (Gomes & Waelbroeck's reversion)."""
    peak = sum(r.impact_peak for r in rows)
    return 1 - sum(r.impact_residual for r in rows) / peak if peak > 0 else float("nan")


def summarise_concavity(results: list[PairedResult], n_boot: int = 500, seed: int = 0) -> dict:
    """Impact exponent and reversion per arm for each (regime, strategy, horizon), and the Arm 1 vs Arm 5 tests.

    The exponent delta of impact ~ (Q/V)^delta is fitted on realised Q/V as in the validation run. CIs resample
    seeds, keeping each seed's runs across sizes and arms together, so the differences are paired. The plan's test:
    delta(Arm 1) - delta(Arm 5) > 0 means the reaction channels, not book consumption, produce the concavity;
    reversion(Arm 5) - reversion(Arm 1) > 0 means reaction drives reversion. Nobody in the ghost knows the executor
    was there, so its impact reverts completely by construction and that second test cannot come out positive;
    reversion(Arm 5) - reversion(Arm 3) asks the question among arms that have a permanent component, whether the
    detection and predation reactions add reversion to the price and inventory response.
    """
    groups: dict = {}
    for r in results:
        groups.setdefault((r.regime, r.strategy, r.horizon), []).append(r)
    rng = np.random.default_rng(seed)
    out = {}
    for (regime, strategy, horizon), rows in sorted(groups.items()):
        bins = len({r.size_fraction for r in rows})
        if bins < 2:
            continue
        seeds = sorted({r.seed for r in rows})
        by_seed_arm: dict = {}
        for r in rows:
            if r.shortfall_bps is not None:
                by_seed_arm.setdefault((r.channels, r.seed), []).append(r)
        arm_row = {}
        draws = rng.integers(0, len(seeds), (n_boot, len(seeds)))
        boots: dict = {}
        for k, channels in ARMS.items():
            pick = _arm_points(rows, channels.key)
            if not pick:
                continue
            arm_row[f"arm{k}"] = {"n": len(pick), **{f"delta_{m}": _fit(pick, m, bins) for m in IMPACT_MEASURES},
                                  "reversion": _reversion(pick),
                                  "mean_peak": float(np.mean([r.impact_peak for r in pick])),
                                  "mean_cost": float(np.mean([_measure(r, "cost") for r in pick]))}
            per_seed = [by_seed_arm.get((channels.key, s), []) for s in seeds]
            for measure in IMPACT_MEASURES:
                boots[(k, measure)] = np.array(
                    [_fit([r for i in d for r in per_seed[i]], measure, bins) for d in draws])
            boots[(k, "reversion")] = np.array([_reversion([r for i in d for r in per_seed[i]]) for d in draws])
        cell = {"arms": arm_row}
        if (1, IMPACT_MEASURES[0]) in boots and (5, IMPACT_MEASURES[0]) in boots:
            tests = {}
            for measure in IMPACT_MEASURES:
                d = boots[(1, measure)] - boots[(5, measure)]
                tests[f"delta_{measure}_arm1_minus_arm5"] = _interval(arm_row["arm1"][f"delta_{measure}"]
                                                                       - arm_row["arm5"][f"delta_{measure}"], d)
            d = boots[(5, "reversion")] - boots[(1, "reversion")]
            tests["reversion_arm5_minus_arm1"] = _interval(arm_row["arm5"]["reversion"] - arm_row["arm1"]["reversion"], d)
            if "arm3" in arm_row:
                d = boots[(5, "reversion")] - boots[(3, "reversion")]
                tests["reversion_arm5_minus_arm3"] = _interval(arm_row["arm5"]["reversion"] - arm_row["arm3"]["reversion"], d)
            cell["tests"] = tests
        out[f"{regime}|{strategy}|{horizon}"] = cell
    return out


def _interval(estimate: float, boots: np.ndarray) -> dict:
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return {"estimate": float(estimate), "low": float("nan"), "high": float("nan"), "n_boot": 0}
    return {"estimate": float(estimate), "low": float(np.quantile(boots, 0.025)),
            "high": float(np.quantile(boots, 0.975)), "n_boot": int(len(boots))}


def format_concavity(summary: dict) -> str:
    lines = [f"{'regime|strategy|T':<44}{'arm':>5}{'n':>5}{'d mean':>8}{'d peak':>8}{'d end':>8}{'d cost':>8}{'revert':>8}{'peak':>7}{'cost':>7}"]
    for key, cell in summary.items():
        for arm, a in cell["arms"].items():
            lines.append(f"{key:<44}{arm:>5}{a['n']:>5}{a['delta_impact_mean']:>8.2f}{a['delta_impact_peak']:>8.2f}"
                         f"{a['delta_impact_end']:>8.2f}{a['delta_cost']:>8.2f}{a['reversion']:>8.2f}{a['mean_peak']:>7.2f}{a['mean_cost']:>7.2f}")
        for name, t in cell.get("tests", {}).items():
            lines.append(f"    {name:<44} {t['estimate']:+.2f} [{t['low']:+.2f},{t['high']:+.2f}]")
    return "\n".join(lines)


# ----- output 4: where does an executor pay? ----------------------------------------------- #

def compare_executors(results: list[PairedResult], reference: str = "twap_market") -> dict:
    """Each executor against ``reference`` on the same seeds: change in IS, mechanical cost and reaction cost.

    ``route`` names the component that accounts for most of the IS change. An executor cheaper than the reference
    through ``mechanical`` while its reaction rises has bought the saving with more reaction, and vice versa.
    """
    cells = _cells(results)
    out: dict = {}
    for (regime, strategy, size, horizon), cell in sorted(cells.items()):
        base = cells.get((regime, reference, size, horizon))
        if strategy == reference or base is None:
            continue
        keys = (GHOST.key, FULL.key)
        seeds = sorted(set(_aligned(cell, *keys)) & set(_aligned(base, *keys)))
        if not seeds:
            continue

        def parts(c):
            full, ghost = _values(c, FULL.key, seeds), _values(c, GHOST.key, seeds)
            return full, ghost - _values(c, GHOST.key, seeds, "drift_bps"), full - ghost

        total, mechanical, reaction = parts(cell)
        total0, mechanical0, reaction0 = parts(base)
        d_total = paired_difference(total, total0)
        d_mechanical = paired_difference(mechanical, mechanical0)
        d_reaction = paired_difference(reaction, reaction0)
        # Drift differences between schedules are noise in this comparison, so the route looks at the two parts.
        route = "mechanical" if abs(d_mechanical["mean"]) >= abs(d_reaction["mean"]) else "reaction"
        out[f"{regime}|{strategy}|{size}|{horizon}"] = {
            "episodes": len(seeds), "d_is": d_total, "d_mechanical": d_mechanical, "d_reaction": d_reaction,
            "cheaper": d_mechanical["mean"] + d_reaction["mean"] < 0, "route": route,
            "offset": d_mechanical["mean"] * d_reaction["mean"] < 0,
        }
    return out


def format_executors(comparison: dict, reference: str = "twap_market") -> str:
    lines = [f"relative to {reference} (negative = cheaper)",
             f"{'regime':<11}{'strategy':<24}{'Q/V':>5}{'T':>6}{'d mech':>8}{'[95% CI]':>16}{'d react':>8}{'[95% CI]':>16}  route"]
    for key, c in comparison.items():
        regime, strategy, size, horizon = key.split("|")
        m, r = c["d_mechanical"], c["d_reaction"]
        direction = "cheaper" if c["cheaper"] else "dearer"
        offset = ", other part moves against it" if c["offset"] else ""
        lines.append(f"{regime:<11}{strategy[:23]:<24}{float(size):>5.2f}{int(horizon):>6}{m['mean']:>8.2f}"
                     f"  [{m['low']:+.2f},{m['high']:+.2f}]{r['mean']:>8.2f}  [{r['low']:+.2f},{r['high']:+.2f}]"
                     f"  {direction} via {c['route']}{offset}")
    return "\n".join(lines)


# ----- the paired-seed protocol ------------------------------------------------------------ #

CONTRASTS = {"total (5-1)": (5, 1), "price (2-1)": (2, 1), "inventory (3-2)": (3, 2),
             "detection (4-3)": (4, 3), "predation (5-4)": (5, 4)}


@dataclass(frozen=True)
class ProtocolConfig:
    """Runs the five arms under both pairing schemes on the same seeds, to measure what counter-based draws buy."""

    strategies: tuple[str, ...] = ("twap_market", "twap_passive", "twap_random")
    sizes: tuple[float, ...] = (0.10, 0.40)
    seeds: tuple[int, ...] = tuple(range(1000, 1040))
    modes: tuple[str, ...] = ("stateful", "counter")
    regime: RegimeSpec = DEEP_COVER
    horizon: int = 2_000
    burn_in: int = 2_000
    after: int = 500

    def specs(self) -> list[PairedSpec]:
        base = self.regime.base()
        return [PairedSpec(seed, strategy, size, channels, horizon=self.horizon, burn_in=self.burn_in, after=self.after,
                           base_config=base, regime=self.regime.name, rng=mode)
                for mode in self.modes for seed in self.seeds for strategy in self.strategies
                for size in self.sizes for channels in ARMS.values()]


def summarise_protocol(results: list[PairedResult]) -> dict:
    """Variance reduction and Sharma's condition per contrast and pairing scheme, per cell and pooled.

    The pooled figure is the median of the per-cell variance reductions; a cell whose arms pair perfectly (an infinite
    reduction) is left out of it, which can only understate the scheme that achieved it.
    """
    by_mode: dict = {}
    for r in results:
        by_mode.setdefault(r.rng, {}).setdefault((r.strategy, r.size_fraction, r.horizon), {}).setdefault(
            r.channels, {})[r.seed] = r
    cells: dict = {}
    pooled: dict = {}
    for mode, groups in by_mode.items():
        for (strategy, size, horizon), cell in sorted(groups.items()):
            seeds = _aligned(cell, *[ARMS[k].key for k in ARMS])
            for name, (k, j) in CONTRASTS.items():
                diag = crn_diagnostics(_values(cell, ARMS[k].key, seeds), _values(cell, ARMS[j].key, seeds))
                cells.setdefault(f"{strategy}|{size}|{horizon}", {}).setdefault(name, {})[mode] = diag
                pooled.setdefault(name, {}).setdefault(mode, []).append(diag["variance_reduction"])
    summary = {}
    for name, modes in pooled.items():
        row = {mode: {"median_variance_reduction": float(np.nanmedian(np.where(np.isinf(v), np.nan, v))) if len(v) else float("nan")}
               for mode, v in ((m, np.array(x, float)) for m, x in modes.items())}
        summary[name] = row
    return {"cells": cells, "pooled": summary}


def format_protocol(summary: dict) -> str:
    lines = [f"{'cell':<36}{'contrast':<18}{'scheme':<10}{'VR':>8}{'[95% CI]':>18}{'corr':>7}{'[95% CI]':>16}{'Sharma':>8}"]
    for cell, contrasts in summary["cells"].items():
        for name, modes in contrasts.items():
            for mode, d in modes.items():
                lines.append(f"{cell:<36}{name:<18}{mode:<10}{d['variance_reduction']:>8.1f}"
                             f"  [{d['vr_low']:>6.1f},{d['vr_high']:>7.1f}]{d['correlation']:>7.2f}"
                             f"  [{d['corr_low']:+.2f},{d['corr_high']:+.2f}]{'yes' if d['sharma_holds'] else 'no':>8}")
    lines.append("\npooled, median variance reduction over cells:")
    for name, modes in summary["pooled"].items():
        lines.append(f"  {name:<18}" + "  ".join(f"{m}: {v['median_variance_reduction']:.1f}" for m, v in modes.items()))
    return "\n".join(lines)


# ----- replication hooks ------------------------------------------------------------------- #

def _common(regime: RegimeSpec, horizon: int, burn_in: int, after: int, rng: str) -> dict:
    return dict(horizon=horizon, burn_in=burn_in, after=after, base_config=regime.base(), rng=rng)


def shi_cartlidge_specs(rates=(0.01, 0.02, 0.05, 0.10, 0.20, 0.40), seeds=tuple(TEST_SEEDS), regime: RegimeSpec = DEEP_COVER,
                        horizon: int = 2_000, burn_in: int = 2_000, after: int = 500, rng: str = "counter") -> list[PairedSpec]:
    """POV at fixed rates, ghost and full market: the reaction as a function of participation."""
    common = _common(regime, horizon, burn_in, after, rng)
    return [PairedSpec(seed, f"pov@{rate:g}", rate, channels, regime=f"sc|{regime.name}", **common)
            for seed in seeds for rate in rates for channels in (GHOST, FULL)]


def summarise_shi_cartlidge(results: list[PairedResult]) -> dict:
    """Shi & Cartlidge's two claims: no reaction at 1% POV, and superlinear growth of the reaction above it."""
    rows = {}
    for (_, strategy, size, _), cell in sorted(_cells(results).items(), key=lambda kv: kv[0][2]):
        seeds = _aligned(cell, GHOST.key, FULL.key)
        reaction = paired_difference(_values(cell, FULL.key, seeds), _values(cell, GHOST.key, seeds))
        rows[strategy] = {"rate": size, "realised_q_over_v": float(np.mean(_values(cell, FULL.key, seeds, "q_over_v"))),
                          "reaction": reaction, "episodes": len(seeds)}
    if not rows:
        return {"rows": rows}
    lowest = rows[min(rows, key=lambda k: rows[k]["rate"])]
    above = [r for r in rows.values() if r is not lowest]
    usable = sorted((r for r in above if r["reaction"]["mean"] > 0), key=lambda r: r["rate"])

    def growth(points):
        if len(points) < 2:
            return float("nan")
        return float(np.polyfit(np.log([r["rate"] for r in points]), np.log([r["reaction"]["mean"] for r in points]), 1)[0])

    delta = growth(usable)
    low = lowest["reaction"]
    # One extreme rate can carry the whole fit, so the exponent is also reported without the largest rate.
    return {"rows": rows, "zero_reaction_at_lowest_rate": bool(low["low"] <= 0 <= low["high"]),
            "growth_exponent_above": delta, "growth_exponent_without_largest": growth(usable[:-1]),
            "superlinear": bool(delta > 1) if delta == delta else False}


def bucci_specs(detecting=(0, 1, 2, 3, 4), sizes=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80), seeds=tuple(TEST_SEEDS),
                regime: RegimeSpec = DEEP_COVER, strategy: str = "twap_market", horizon: int = 2_000, burn_in: int = 2_000,
                after: int = 500, rng: str = "counter") -> list[PairedSpec]:
    """Impact over a wide size range in the A1 arm, as detecting makers replace plain ones."""
    common = _common(regime, horizon, burn_in, after, rng)
    return [PairedSpec(seed, strategy, size, ARMS[4], detecting=d, regime=f"detecting={d}", **common)
            for d in detecting for seed in seeds for size in sizes]


def summarise_bucci(results: list[PairedResult], measure: str = "impact_mean", n_boot: int = 500, seed: int = 0) -> dict:
    """Bucci et al.'s crossover: a local exponent at small Q/V near 1 falling toward 1/2 at large Q/V.

    Mean ``measure`` per nominal size, for each maker composition, gives one curve; the exponent is fitted
    separately on its lower and upper half. A crossover is the lower exponent significantly exceeding the
    upper one, with the interval from resampling seeds. The default measure is the mid's displacement, because cost
    includes the half spread, a constant that flattens the small sizes and would hide a linear regime.
    """
    groups: dict = {}
    for r in results:
        if r.shortfall_bps is not None:
            groups.setdefault(r.regime, {}).setdefault(r.size_fraction, {}).setdefault(r.seed, r)
    rng = np.random.default_rng(seed)
    out = {}
    for label, by_size in sorted(groups.items()):
        sizes = sorted(by_size)
        half = len(sizes) // 2
        seeds = sorted({s for runs in by_size.values() for s in runs})

        def curve(picked):
            x = np.array([np.mean([by_size[q][s].q_over_v for s in picked if s in by_size[q]] or [np.nan]) for q in sizes])
            y = np.array([np.mean([_measure(by_size[q][s], measure) for s in picked if s in by_size[q]] or [np.nan]) for q in sizes])
            return x, y

        def slopes(picked):
            x, y = curve(picked)
            fit = lambda idx: _log_slope(x[idx], y[idx])
            return fit(np.arange(half + 1)), fit(np.arange(half, len(sizes)))

        low, high = slopes(seeds)
        draws = rng.integers(0, len(seeds), (n_boot, len(seeds)))
        boots = np.array([slopes([seeds[i] for i in d]) for d in draws])
        interval = _interval(low - high, boots[:, 0] - boots[:, 1])
        x, y = curve(seeds)
        out[label] = {"q_over_v": x.tolist(), "impact": y.tolist(), "delta_low": low, "delta_high": high,
                      "low_minus_high": interval, "crossover": bool(interval["low"] > 1e-9)}
    return out


def _log_slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = (x > 0) & (y > 0)
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")


def abides_specs(strategies=("twap_market", "twap@size=0.5,timing=0.3", "twap_random", "mimic"), sizes=(0.20, 0.40),
                 seeds=tuple(TEST_SEEDS), regime: RegimeSpec = DEEP_COVER, horizon: int = 2_000, burn_in: int = 2_000,
                 after: int = 500, rng: str = "counter") -> list[PairedSpec]:
    """Predictable and randomised acquirers with and without detection and predation: Arms 3 and 5, the two the summary compares."""
    common = _common(regime, horizon, burn_in, after, rng)
    return [PairedSpec(seed, strategy, size, channels, regime=f"abides|{regime.name}", **common)
            for seed in seeds for strategy in strategies for size in sizes for channels in (ARMS[3], ARMS[5])]


def summarise_abides(results: list[PairedResult], predictable: str = "twap_market") -> dict:
    """ABIDES-MARL's quote-lifting: how far the opposite best moves against an acquirer, by how predictable it is.

    The lift caused by detection and predation is the mean quote lift in Arm 5 minus Arm 3; the test is whether it
    is larger for the ``predictable`` schedule than for each randomised one, paired by seed.
    """
    cells = _cells(results)
    out: dict = {}
    for (regime, strategy, size, horizon), cell in sorted(cells.items()):
        seeds = _aligned(cell, ARMS[3].key, ARMS[5].key)
        lift5, lift3 = _values(cell, ARMS[5].key, seeds, "quote_lift"), _values(cell, ARMS[3].key, seeds, "quote_lift")
        row = {"lift_arm3": float(lift3.mean()), "lift_arm5": float(lift5.mean()),
               "lift_from_detection_and_predation": paired_difference(lift5, lift3)}
        base = cells.get((regime, predictable, size, horizon))
        if base is not None and strategy != predictable:
            common = sorted(set(seeds) & set(_aligned(base, ARMS[3].key, ARMS[5].key)))
            mine = _values(cell, ARMS[5].key, common, "quote_lift") - _values(cell, ARMS[3].key, common, "quote_lift")
            theirs = _values(base, ARMS[5].key, common, "quote_lift") - _values(base, ARMS[3].key, common, "quote_lift")
            row["predictable_minus_this"] = paired_difference(theirs, mine)
        out[f"{strategy}|{size}"] = row
    return out
