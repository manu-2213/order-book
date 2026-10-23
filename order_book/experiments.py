"""Experiment definitions and runner for the measurement phase (plan sections 6, 9 and 10).

Everything an experiment needs is spelled out in frozen dataclasses:

    AdversarySpec   which market the executor trades in (A0, A1 at three thresholds, A4)
    OBSERVERS       detectors attached as observers (they see the market, nobody reacts to them)
    EpisodeSpec     one execution episode: seed, strategy, size, adversary, visibility, task
    E2Config        detectability of classic strategies against A1-A3 (hypothesis H1)
    E3Config        leakage cost by counterfactual: visible vs invisible executor on the same seed

Seeds: 0-999 are training seeds (calibration, detector training, tuning); test
seeds start at 1000. The executor's side is fixed by the seed's parity so
that every strategy and adversary level is paired on the same seeds, sides
and burn-in states. Results are JSON-serialisable records; the summaries
implement the paired-bootstrap protocol of section 10.
"""

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from order_book.config import FlowDetectorConfig, SimConfig
from order_book.detectors import FlowDetector, LearnedDetector, held_out_detectors
from order_book.execution import (
    Calibration, ExecutionTask, ExecutorAgent, all_baselines, expected_volume_from_tape, make_strategy,
)
from order_book.presets import FLOW_THRESHOLDS, adversary
from order_book.scoring import (
    activity_labels, bootstrap_ci, cvar, detection_delay, false_alarm_rate, holm, paired_difference, roc_auc,
)
from order_book.simulator import Simulator

TRAIN_SEEDS = range(0, 1000)
TEST_SEEDS = range(1000, 1020)
SIZES = (0.05, 0.10, 0.20)
EXECUTOR_ENDOWMENT = 50_000          # shares; cash = shares x mid, so any task size is funded


# ----- adversaries ------------------------------------------------------------------------ #

@dataclass(frozen=True)
class AdversarySpec:
    name: str
    level: str                      # A0 | A1 | A4
    threshold: str = "medium"       # flow-detector threshold the makers (and anticipator) react to

    def config(self, base: SimConfig | None = None) -> SimConfig:
        return adversary(self.level, self.threshold, base)


ADVERSARIES = {
    "A0": AdversarySpec("A0", "A0"),
    "A1-lenient": AdversarySpec("A1-lenient", "A1", "lenient"),
    "A1-medium": AdversarySpec("A1-medium", "A1", "medium"),
    "A1-strict": AdversarySpec("A1-strict", "A1", "strict"),
    "A4": AdversarySpec("A4", "A4", "medium"),
}

A1_OBSERVERS = tuple(f"flow_{k}" for k in FLOW_THRESHOLDS)
A2_OBSERVERS = ("runs", "cusum", "volume_clock", "regularity")
A3_OBSERVER = "learned"
OBSERVERS = A1_OBSERVERS + A2_OBSERVERS + (A3_OBSERVER,)


def make_observers(names, learned_weights: str | None):
    out = {}
    for name in names:
        if name.startswith("flow_"):
            out[name] = FlowDetector(FlowDetectorConfig(threshold=FLOW_THRESHOLDS[name[5:]]))
        elif name in A2_OBSERVERS:
            out[name] = held_out_detectors()[name]
        elif name == A3_OBSERVER:
            if learned_weights is None:
                continue  # A3 needs trained weights (scripts/train_detector.py); skipped if absent
            out[name] = LearnedDetector(learned_weights)
        else:
            raise KeyError(f"unknown observer {name!r}")
    return out


# ----- episodes --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    strategy: str                   # a key of ``all_baselines()``; ignored when visibility == "none"
    size_fraction: float
    adversary: str                  # a key of ``ADVERSARIES``
    visibility: str = "visible"     # visible | invisible (detectors ignore the executor) | none (no executor)
    side: str | None = None         # None: "bid" on even seeds, "ask" on odd seeds
    horizon: int = 2_000
    decision_interval: int = 5
    burn_in: int = 2_000
    after: int = 500                # steps recorded after the horizon (impact reversion)
    observers: tuple[str, ...] = ()
    record_path: bool = False
    base_config: SimConfig | None = None

    @property
    def resolved_side(self) -> str:
        return self.side or ("bid" if self.seed % 2 == 0 else "ask")

    @property
    def window(self) -> int:
        return self.horizon + self.after


@dataclass
class EpisodeResult:
    spec: dict
    side: str
    quantity: int
    executed: int
    forced: int
    incomplete: int
    completion: float
    shortfall_bps: float | None
    arrival_mid: float
    final_mid: float
    aggressive_share: float
    orders_sent: int
    end: int | None
    expected_volume: float
    executor_window: tuple[int, int]            # [start, end) in window steps
    institutional_windows: list                  # (start, end, side) in window steps
    signals: dict = field(default_factory=dict)  # observer name -> list of per-step signals
    mid_path: list | None = None
    seconds: float = 0.0

    def to_json(self) -> dict:
        d = asdict(self)
        d["spec"] = {k: v for k, v in self.spec.items() if k != "base_config"}
        return d


def run_episode(spec: EpisodeSpec, calibration: Calibration | None = None,
                learned_weights: str | None = None) -> EpisodeResult:
    t0 = time.perf_counter()
    cfg = ADVERSARIES[spec.adversary].config(spec.base_config)
    sim = Simulator(spec.seed, cfg, analysis=True)
    observers = make_observers(spec.observers, learned_weights)
    for name, det in observers.items():
        sim.detectors[f"obs:{name}"] = det
    executor = None
    if spec.visibility != "none":
        # The executor is added before the burn-in so its stream exists; it does nothing until ``start``.
        # Its expected volume is set after the burn-in from the public tape (identical across pairings).
        task = ExecutionTask(side=spec.resolved_side, size_fraction=spec.size_fraction, horizon=spec.horizon,
                             decision_interval=spec.decision_interval)
        executor = sim.add_external(ExecutorAgent, EXECUTOR_ENDOWMENT * cfg.book.mid, EXECUTOR_ENDOWMENT,
                                    task=task, strategy=make_strategy(spec.strategy, calibration),
                                    start=spec.burn_in, expected_volume=1.0)
        if spec.visibility == "invisible":
            sim.market.hide(executor.cid)
    sim.run(spec.burn_in)
    volume = expected_volume_from_tape(sim.market.tape, spec.burn_in)
    if executor is not None:
        executor.expected_volume = volume
        executor.quantity = max(1, round(spec.size_fraction * volume * spec.horizon))
    ps, window = sim.market.public, spec.window
    mids = np.empty(window)
    signals = {name: np.empty(window) for name in observers}
    signals["flow"] = np.empty(window)  # the detector the market makers actually react to
    for t in range(window):
        sim.step()
        mids[t] = ps.mid
        signals["flow"][t] = sim.detectors["flow"].signal
        for name, det in observers.items():
            signals[name][t] = det.signal
    sim.check_conservation()
    inst = [(r.start - spec.burn_in, (r.end if r.end is not None else sim.market.step) - spec.burn_in, r.side)
            for r in sim.metaorders.records if (r.end is None or r.end > spec.burn_in)]
    if executor is None:
        return EpisodeResult(asdict(spec), spec.resolved_side, 0, 0, 0, 0, 1.0, None, float(mids[0]), float(mids[-1]),
                             0.0, 0, None, volume, (0, spec.horizon), inst,
                             {k: v.tolist() for k, v in signals.items()},
                             mids.tolist() if spec.record_path else None, time.perf_counter() - t0)
    r = executor.result
    end_rel = (r.end - spec.burn_in) if r.end is not None else spec.horizon
    return EpisodeResult(
        asdict(spec), r.side, r.quantity, r.executed, r.forced, r.incomplete, r.completion, r.shortfall_bps,
        r.arrival_mid, float(mids[-1]), r.aggressive_executed / r.executed if r.executed else 0.0, r.orders_sent,
        end_rel, volume, (0, min(end_rel, window)), inst, {k: v.tolist() for k, v in signals.items()},
        mids.tolist() if spec.record_path else None, time.perf_counter() - t0,
    )


# ----- runner ----------------------------------------------------------------------------- #

def _run(args):
    spec, calibration, learned_weights = args
    return run_episode(spec, calibration, learned_weights)


def run_specs(specs: list[EpisodeSpec], calibration: Calibration | None = None, learned_weights: str | None = None,
              workers: int = 1, verbose: bool = True, label: str = "run", sink=None) -> list[EpisodeResult]:
    """Run episodes (in parallel processes when ``workers > 1``), printing one progress line per episode
    and appending each record to ``sink`` (a file-like, JSON lines) when given."""
    results = []
    t0 = time.perf_counter()
    jobs = [(s, calibration, learned_weights) for s in specs]
    if workers > 1:
        pool = ProcessPoolExecutor(workers)
        iterator = pool.map(_run, jobs)
    else:
        pool, iterator = None, map(_run, jobs)
    try:
        for i, r in enumerate(iterator, 1):
            results.append(r)
            if sink is not None:
                sink.write(json.dumps(r.to_json()) + "\n")
                sink.flush()
            if verbose:
                s = r.spec
                elapsed = time.perf_counter() - t0
                eta = elapsed / i * (len(specs) - i)
                is_txt = "   n/a  " if r.shortfall_bps is None else f"{r.shortfall_bps:+7.2f}"
                print(f"[{label} {i:4d}/{len(specs)}] seed={s['seed']} {s['strategy']:<22s} Q/V={s['size_fraction']:.2f} "
                      f"{s['adversary']:<10s} {s['visibility']:<9s} {r.side} IS={is_txt}bps exec={r.executed:5d} "
                      f"forced={r.forced:4d} ({r.seconds:.1f}s, elapsed {elapsed:.0f}s, eta {eta:.0f}s)", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return results


# ----- E2: detectability ------------------------------------------------------------------- #

@dataclass(frozen=True)
class E2Config:
    """How detectable are classic strategies, and how fast are they detected? (H1)

    Each strategy at each size runs in the A1-medium market (makers react to the flow detector)
    with every detector of levels A1-A3 attached as an observer. Scores: ROC AUC of |signal| at
    telling executor-active steps from all other steps of the window, ROC AUC of the signed signal
    against the executor's side, median and 90th-percentile detection delay at the operating
    threshold, false-alarm rate on steps where only institutions are active and on quiet steps, and,
    for comparison, each detector's AUC on the institutional metaorders of the same episodes.
    """

    name: str = "E2"
    strategies: tuple[str, ...] = ("twap_market", "twap_passive", "twap_random", "pov", "almgren_chriss_1e-06",
                                   "footprint", "mimic")
    sizes: tuple[float, ...] = SIZES
    adversary: str = "A1-medium"
    observers: tuple[str, ...] = OBSERVERS
    seeds: tuple[int, ...] = tuple(TEST_SEEDS)
    horizon: int = 2_000
    decision_interval: int = 5
    burn_in: int = 2_000
    after: int = 500
    threshold: float = 0.5          # operating threshold for delay and false alarms

    def specs(self) -> list[EpisodeSpec]:
        return [EpisodeSpec(seed, strategy, size, self.adversary, "visible", None, self.horizon,
                            self.decision_interval, self.burn_in, self.after, self.observers)
                for strategy in self.strategies for size in self.sizes for seed in self.seeds]


def summarise_e2(results: list[EpisodeResult], threshold: float = 0.5) -> dict:
    out: dict = {}
    detectors = sorted({k for r in results for k in r.signals})
    groups: dict = {}
    for r in results:
        groups.setdefault((r.spec["strategy"], r.spec["size_fraction"]), []).append(r)
    for (strategy, size), rs in sorted(groups.items()):
        row: dict = {}
        for det in detectors:
            aucs, signed, delays, fa_inst, fa_quiet, auc_inst = [], [], [], [], [], []
            for r in rs:
                sig = np.asarray(r.signals[det])
                n = len(sig)
                exec_mask = np.zeros(n, bool)
                exec_mask[r.executor_window[0]:r.executor_window[1]] = True
                inst = activity_labels(n, r.institutional_windows)
                aucs.append(roc_auc(np.abs(sig), exec_mask))
                signed.append(roc_auc(sig if r.side == "bid" else -sig, exec_mask))
                d = detection_delay(sig, r.executor_window[0], r.executor_window[1], r.side, threshold)
                delays.append(d if d is not None else r.executor_window[1] - r.executor_window[0])
                fa_inst.append(false_alarm_rate(sig, (~exec_mask) & (inst != 0), threshold))
                fa_quiet.append(false_alarm_rate(sig, (~exec_mask) & (inst == 0), threshold))
                auc_inst.append(roc_auc(np.abs(sig[~exec_mask]), inst[~exec_mask] != 0))
            row[det] = {
                "auc_abs": bootstrap_ci(aucs), "auc_signed": bootstrap_ci(signed),
                "delay_median": float(np.nanmedian(delays)), "delay_p90": float(np.nanpercentile(delays, 90)),
                "false_alarm_institutional": float(np.nanmean(fa_inst)),
                "false_alarm_quiet": float(np.nanmean(fa_quiet)),
                "auc_institutional": float(np.nanmean(auc_inst)), "episodes": len(rs),
            }
        out[f"{strategy}|{size}"] = row
    return out


def format_e2(summary: dict) -> str:
    lines = [f"{'strategy':<22s} {'Q/V':>5s} {'detector':<14s} {'AUC|s|':>7s} {'[95% CI]':>15s} {'AUC side':>8s} "
             f"{'delay med':>9s} {'p90':>6s} {'FA inst':>7s} {'FA quiet':>8s} {'AUC inst':>8s}"]
    for key, row in summary.items():
        strategy, size = key.split("|")
        for det, m in row.items():
            a, lo, hi = m["auc_abs"]
            lines.append(f"{strategy:<22s} {float(size):5.2f} {det:<14s} {a:7.3f} [{lo:6.3f},{hi:6.3f}] "
                         f"{m['auc_signed'][0]:8.3f} {m['delay_median']:9.0f} {m['delay_p90']:6.0f} "
                         f"{m['false_alarm_institutional']:7.3f} {m['false_alarm_quiet']:8.3f} {m['auc_institutional']:8.3f}")
    return "\n".join(lines)


# ----- E3: leakage ------------------------------------------------------------------------- #

@dataclass(frozen=True)
class E3Config:
    """How much does leakage cost? (H1)

    Every strategy at every size runs under A0, A1-medium and A4, each seed twice: with the executor
    visible and with it invisible to the detectors (``Market.hide``), plus one run per seed and
    adversary without any executor (the market's own path). Leakage cost = IS(visible) -
    IS(invisible), paired by seed, with a bootstrap CI and a one-sided p-value; IS(invisible) is the
    impact cost. Also reported: CVaR 95%, completion, forced quantity, results by side, and Holm-
    adjusted p-values over the strategy x size x adversary tests. Gate of Phase 2: TWAP (market)
    leakage significantly above zero at one size or more.
    """

    name: str = "E3"
    strategies: tuple[str, ...] = tuple(all_baselines())
    sizes: tuple[float, ...] = SIZES
    adversaries: tuple[str, ...] = ("A0", "A1-medium", "A4")
    visibilities: tuple[str, ...] = ("visible", "invisible")
    seeds: tuple[int, ...] = tuple(TEST_SEEDS)
    horizon: int = 2_000
    decision_interval: int = 5
    burn_in: int = 2_000
    after: int = 500
    record_paths: bool = False
    gate_strategy: str = "twap_market"

    def specs(self) -> list[EpisodeSpec]:
        specs = [EpisodeSpec(seed, "none", 0.0, adv, "none", None, self.horizon, self.decision_interval,
                             self.burn_in, self.after, (), self.record_paths)
                 for adv in self.adversaries for seed in self.seeds]
        specs += [EpisodeSpec(seed, strategy, size, adv, vis, None, self.horizon, self.decision_interval,
                              self.burn_in, self.after, (), self.record_paths)
                  for strategy in self.strategies for size in self.sizes for adv in self.adversaries
                  for vis in self.visibilities for seed in self.seeds]
        return specs


def summarise_e3(results: list[EpisodeResult], gate_strategy: str = "twap_market") -> dict:
    by_key: dict = {}
    for r in results:
        s = r.spec
        if s["visibility"] == "none":
            continue
        by_key.setdefault((s["strategy"], s["size_fraction"], s["adversary"]), {}).setdefault(
            s["visibility"], {})[s["seed"]] = r
    rows: dict = {}
    pvals, pkeys = [], []
    for (strategy, size, adv), vis in sorted(by_key.items()):
        visible = vis.get("visible", {})
        invisible = vis.get("invisible", {})
        seeds = sorted(set(visible) & set(invisible)) if invisible else sorted(visible)
        is_vis = [visible[s].shortfall_bps for s in seeds]
        row = {
            "episodes": len(seeds),
            "is_visible": bootstrap_ci(is_vis), "cvar95_visible": cvar(is_vis),
            "completion": float(np.mean([visible[s].completion for s in seeds])) if seeds else float("nan"),
            "forced_share": float(np.mean([visible[s].forced / max(visible[s].quantity, 1) for s in seeds])) if seeds else float("nan"),
            "by_side": {side: bootstrap_ci([visible[s].shortfall_bps for s in seeds if visible[s].side == side])
                        for side in ("bid", "ask")},
        }
        if invisible:
            is_inv = [invisible[s].shortfall_bps for s in seeds]
            row["is_invisible"] = bootstrap_ci(is_inv)
            row["leakage"] = paired_difference(is_vis, is_inv)
            pvals.append(row["leakage"]["p_greater"])
            pkeys.append((strategy, size, adv))
        rows[f"{strategy}|{size}|{adv}"] = row
    adjusted = holm(pvals) if pvals else []
    for key, p_adj in zip(pkeys, adjusted):
        rows[f"{key[0]}|{key[1]}|{key[2]}"]["leakage"]["p_holm"] = p_adj
    gate = [(k, v["leakage"]) for k, v in rows.items() if k.startswith(gate_strategy + "|") and "leakage" in v
            and not k.endswith("|A0")]
    passed = any(l["p_greater"] < 0.05 and l["mean"] > 0 for _, l in gate)
    return {"rows": rows, "gate": {"strategy": gate_strategy, "passed": passed,
                                   "tests": {k: l for k, l in gate}}}


def format_e3(summary: dict) -> str:
    lines = [f"{'strategy':<22s} {'Q/V':>5s} {'adversary':<10s} {'n':>3s} {'IS vis':>8s} {'IS invis':>8s} "
             f"{'leakage':>8s} {'[95% CI]':>17s} {'p>0':>6s} {'p Holm':>6s} {'CVaR95':>7s} {'compl':>6s} {'forced':>6s}"]
    for key, row in summary["rows"].items():
        strategy, size, adv = key.split("|")
        vis = row["is_visible"][0]
        inv = row.get("is_invisible", (float("nan"),))[0]
        leak = row.get("leakage")
        leak_txt = (f"{leak['mean']:8.2f} [{leak['low']:7.2f},{leak['high']:7.2f}] {leak['p_greater']:6.3f} "
                    f"{leak.get('p_holm', float('nan')):6.3f}") if leak else f"{'':8s} {'':17s} {'':6s} {'':6s}"
        lines.append(f"{strategy:<22s} {float(size):5.2f} {adv:<10s} {row['episodes']:3d} {vis:8.2f} {inv:8.2f} "
                     f"{leak_txt} {row['cvar95_visible']:7.2f} {row['completion']:6.3f} {row['forced_share']:6.3f}")
    g = summary["gate"]
    lines.append(f"\nPhase 2 gate ({g['strategy']} leakage > 0 at one size or more under A1/A4): "
                 f"{'PASS' if g['passed'] else 'FAIL'}")
    return "\n".join(lines)


def impact_paths(results: list[EpisodeResult]) -> dict:
    """Mid path with the executor minus the path without it, per (strategy, size, adversary, seed)."""
    none = {(r.spec["adversary"], r.spec["seed"]): np.asarray(r.mid_path) for r in results
            if r.spec["visibility"] == "none" and r.mid_path is not None}
    out = {}
    for r in results:
        s = r.spec
        if s["visibility"] == "visible" and r.mid_path is not None and (s["adversary"], s["seed"]) in none:
            sign = 1 if r.side == "bid" else -1
            out[f"{s['strategy']}|{s['size_fraction']}|{s['adversary']}|{s['seed']}"] = \
                (sign * (np.asarray(r.mid_path) - none[(s["adversary"], s["seed"])])).tolist()
    return out
