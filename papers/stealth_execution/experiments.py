"""The stealth-execution study: E2 detectability, E3 leakage, E7/E7b regimes and detector strength.

    E2Config          detectability of classic strategies against A1-A3 (hypothesis H1)
    E3Config          leakage cost by counterfactual: visible vs invisible executor on the same seed
    E7Config          where leakage appears as liquidity and institutional cover fall (H4)
    E7StrengthConfig  detector strength in a fixed liquid, low-cover regime

The harness these run on is ``order_book.experiments``; the summaries implement the paired-bootstrap
protocol of section 10 of the research plan.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np

from order_book.config import SimConfig
from order_book.execution import all_baselines
from order_book.experiments import (
    DEFAULT_REGIMES, OBSERVERS, SIZES, TEST_SEEDS, EpisodeResult, EpisodeSpec, RegimeSpec,
)
from order_book.scoring import (
    activity_labels, bootstrap_ci, cvar, detection_delay, false_alarm_rate, holm, paired_difference, roc_auc,
)


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


def _nan_aggregate(func, values, *args) -> float:
    """Apply ``func`` over the non-nan values; return nan (no warning) when there are none.

    Scores are legitimately undefined per episode (e.g. a detector's AUC on institutional metaorders
    when none are active outside the executor's window), so an all-nan group means "no data", not an
    error, and must not raise ``RuntimeWarning: Mean of empty slice``.
    """
    arr = np.asarray(values, float)
    arr = arr[~np.isnan(arr)]
    return float(func(arr, *args)) if arr.size else float("nan")


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
                "delay_median": _nan_aggregate(np.median, delays),
                "delay_p90": _nan_aggregate(np.percentile, delays, 90),
                "false_alarm_institutional": _nan_aggregate(np.mean, fa_inst),
                "false_alarm_quiet": _nan_aggregate(np.mean, fa_quiet),
                "auc_institutional": _nan_aggregate(np.mean, auc_inst), "episodes": len(rs),
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
    rng: str = "stateful"

    def specs(self) -> list[EpisodeSpec]:
        specs = [EpisodeSpec(seed, "none", 0.0, adv, "none", None, self.horizon, self.decision_interval,
                             self.burn_in, self.after, (), self.record_paths, rng=self.rng)
                 for adv in self.adversaries for seed in self.seeds]
        specs += [EpisodeSpec(seed, strategy, size, adv, vis, None, self.horizon, self.decision_interval,
                              self.burn_in, self.after, (), self.record_paths, rng=self.rng)
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


# ----- E7: where does leakage live? (plan section 9, hypothesis H4) ------------------------ #

# ----- E7 (regimes live in the library) ----------------------------------------------------- #

@dataclass(frozen=True)
class E7Config:
    """How much does cover help, and when does leakage appear? (H4)

    Runs the leakage counterfactual (visible vs invisible, plus no-executor) across a grid of market
    regimes and executor sizes, under A1-medium and A4. As liquidity and institutional cover fall and
    the order's share of flow rises, detection should bite and leakage should grow. Reports, per
    regime x strategy x size x adversary: the paired leakage with interval and p-value; the mean
    market volume per step and executor Q (so the order's share of liquidity is explicit); and how
    much the executor itself moves the flow detector (mean |signal| over its window, visible minus
    invisible), which is the detection channel the leakage flows through.
    """

    name: str = "E7"
    regimes: tuple[RegimeSpec, ...] = DEFAULT_REGIMES
    strategies: tuple[str, ...] = ("twap_market", "twap_passive")
    sizes: tuple[float, ...] = (0.2, 0.4)
    adversaries: tuple[str, ...] = ("A1-medium", "A4")
    visibilities: tuple[str, ...] = ("visible", "invisible")
    seeds: tuple[int, ...] = tuple(TEST_SEEDS)
    horizon: int = 2_000
    decision_interval: int = 5
    burn_in: int = 2_000
    after: int = 500
    rng: str = "stateful"

    def specs(self) -> list[EpisodeSpec]:
        specs = []
        for rs in self.regimes:
            base = rs.base()
            common = dict(horizon=self.horizon, decision_interval=self.decision_interval,
                          burn_in=self.burn_in, after=self.after, base_config=base, regime=rs.name, rng=self.rng)
            for adv in self.adversaries:
                for seed in self.seeds:
                    specs.append(EpisodeSpec(seed, "none", 0.0, adv, "none", **common))
            for strategy in self.strategies:
                for size in self.sizes:
                    for adv in self.adversaries:
                        for vis in self.visibilities:
                            for seed in self.seeds:
                                specs.append(EpisodeSpec(seed, strategy, size, adv, vis, **common))
        return specs


def _flow_firing(r: EpisodeResult) -> float:
    """Mean |flow detector signal| over the executor's active window (the detection the makers react to)."""
    sig = np.abs(np.asarray(r.signals.get("flow", [])))
    lo, hi = r.executor_window
    window = sig[lo:hi]
    return float(window.mean()) if window.size else float("nan")


def summarise_e7(results: list[EpisodeResult]) -> dict:
    by_key: dict = {}
    for r in results:
        s = r.spec
        if s["visibility"] == "none":
            continue
        by_key.setdefault((s["regime"], s["strategy"], s["size_fraction"], s["adversary"]), {}).setdefault(
            s["visibility"], {})[s["seed"]] = r
    rows: dict = {}
    for (reg, strategy, size, adv), vis in sorted(by_key.items()):
        visible, invisible = vis.get("visible", {}), vis.get("invisible", {})
        seeds = sorted(set(visible) & set(invisible))
        if not seeds:
            continue
        is_vis = [visible[s].shortfall_bps for s in seeds]
        is_inv = [invisible[s].shortfall_bps for s in seeds]
        firing_vis = [_flow_firing(visible[s]) for s in seeds]
        firing_inv = [_flow_firing(invisible[s]) for s in seeds]
        rows[f"{reg}|{strategy}|{size}|{adv}"] = {
            "episodes": len(seeds),
            "volume_per_step": float(np.mean([visible[s].expected_volume for s in seeds])),
            "quantity": float(np.mean([visible[s].quantity for s in seeds])),
            "is_visible": bootstrap_ci(is_vis), "is_invisible": bootstrap_ci(is_inv),
            "leakage": paired_difference(is_vis, is_inv),
            "cvar95_visible": cvar(is_vis), "completion": float(np.mean([visible[s].completion for s in seeds])),
            "flow_firing_visible": float(np.nanmean(firing_vis)),
            "flow_firing_channel": float(np.nanmean(np.array(firing_vis) - np.array(firing_inv))),
        }
    return {"rows": rows, "regimes": {rs.name: asdict(rs) for rs in DEFAULT_REGIMES}}


def format_e7(summary: dict) -> str:
    lines = [f"{'regime':<16}{'strategy':<14}{'Q/V':>5}{'adv':>11}{'V/step':>7}{'Q':>7}{'ISvis':>7}"
             f"{'leak':>7}{'[95% CI]':>16}{'p':>6}{'fire':>6}{'chan':>6}{'compl':>6}"]
    for key, r in summary["rows"].items():
        reg, strat, size, adv = key.split("|")
        leak = r["leakage"]
        lines.append(f"{reg:<16}{strat:<14}{float(size):>5.2f}{adv:>11}{r['volume_per_step']:>7.1f}"
                     f"{r['quantity']:>7.0f}{r['is_visible'][0]:>7.2f}{leak['mean']:>7.2f}"
                     f"  [{leak['low']:+.2f},{leak['high']:+.2f}]{leak['p_greater']:>6.2f}"
                     f"{r['flow_firing_visible']:>6.2f}{r['flow_firing_channel']:>6.2f}{r['completion']:>6.2f}")
    return "\n".join(lines)


# ----- E7b: detector-strength sweep in a fixed (liquid, low-cover) regime ------------------ #

@dataclass(frozen=True)
class StrengthSpec:
    """One detection-strength setting: the flow threshold, how many makers react, and how hard."""

    name: str
    threshold: str = "medium"       # flow-detector threshold the makers react to
    detecting: int = 2              # number of makers that react
    detect_skew: float = 4.0        # reservation shift per unit signal (ticks)
    detect_widen: float = 3.0       # extra half-spread on the pressured side per unit signal
    detect_size_cut: float = 0.7    # size cut on the pressured side per unit signal


DETECTION_STRENGTHS = (
    StrengthSpec("weak", "strict", 2, 4.0, 3.0, 0.7),       # fires rarely, moderate response
    StrengthSpec("default", "medium", 2, 4.0, 3.0, 0.7),    # the A1-medium response (E2/E3)
    StrengthSpec("strong", "lenient", 4, 8.0, 6.0, 0.9),    # all makers react, sooner and harder
    StrengthSpec("max", "lenient", 4, 12.0, 10.0, 1.0),     # pulls the pressured side entirely
)


def _with_response(cfg: SimConfig, skew: float, widen: float, size_cut: float) -> SimConfig:
    """Set the detection response on every market maker (before ``adversary`` wires who detects)."""
    makers = tuple(replace(m, detect_skew=skew, detect_widen=widen, detect_size_cut=size_cut)
                   for m in cfg.population.market_makers)
    return replace(cfg, population=replace(cfg.population, market_makers=makers))


def strength_config(regime_spec: RegimeSpec, strength: StrengthSpec, level: str = "A1") -> SimConfig:
    """Final config: the regime's market with the strength's maker response and detection wiring."""
    from order_book.presets import adversary
    base = _with_response(regime_spec.base(), strength.detect_skew, strength.detect_widen, strength.detect_size_cut)
    return adversary(level, strength.threshold, base=base, detecting=strength.detecting)


@dataclass(frozen=True)
class E7StrengthConfig:
    """Vary detector strength in a fixed liquid, low-cover regime to find where TWAP leakage becomes
    significant (~1-2 bps) without the market breaking down (the thin regimes just add variance).

    The regime stays liquid so impact variance is low and completion stays at 1; only the detector's
    threshold, the share of detecting makers and the response magnitude change. The setting with the
    largest significant leakage at full completion is the candidate Phase 3 RL environment.
    """

    name: str = "E7b"
    regime: RegimeSpec = RegimeSpec("low_cover", 1.0, 1)
    strengths: tuple[StrengthSpec, ...] = DETECTION_STRENGTHS
    strategies: tuple[str, ...] = ("twap_market", "twap_passive")
    sizes: tuple[float, ...] = (0.2, 0.4)
    level: str = "A1"               # A1 (detecting makers only) or A4 (plus an anticipator)
    seeds: tuple[int, ...] = tuple(TEST_SEEDS)
    horizon: int = 2_000
    decision_interval: int = 5
    burn_in: int = 2_000
    after: int = 500
    rng: str = "stateful"

    def specs(self) -> list[EpisodeSpec]:
        specs = []
        for strength in self.strengths:
            cfg = strength_config(self.regime, strength, self.level)
            label = f"{self.regime.name}/{strength.name}"
            common = dict(horizon=self.horizon, decision_interval=self.decision_interval, burn_in=self.burn_in,
                          after=self.after, base_config=cfg, regime=label, prebuilt_config=True, rng=self.rng)
            for strategy in self.strategies:
                for size in self.sizes:
                    for vis in ("visible", "invisible"):
                        for seed in self.seeds:
                            specs.append(EpisodeSpec(seed, strategy, size, self.level, vis, **common))
        return specs
