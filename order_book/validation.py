"""Validation of the background market against the targets of the research plan (section 4.2).

``run_validation`` simulates several seeds in analysis mode, recording the mid
path, spread, depth and the metaorder labels, and ``summarise`` turns them
into the stylised-fact and impact statistics. Targets are *reported*, never
enforced: ``check`` only compares measured values with the plan's starting
targets so that a run prints a pass/fail table.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from order_book.config import SimConfig
from order_book.simulator import Simulator


@dataclass
class SeedRun:
    seed: int
    steps: int
    mid: np.ndarray                 # per step
    spread: np.ndarray              # per step, nan when a side is empty
    one_sided: np.ndarray           # per step, True when a side is empty
    tape: list                      # (step, price, qty, aggressor, buyer, seller)
    metaorders: list                # MetaorderRecord
    budget_rejects: dict
    steps_per_second: float
    inst_cids: set = field(default_factory=set)


def run_seed(seed: int, steps: int, config: SimConfig | None = None, burn_in: int = 2_000) -> SeedRun:
    sim = Simulator(seed, config, analysis=True)
    sim.run(burn_in)
    offset = sim.market.step
    mid, spread = np.empty(steps), np.full(steps, np.nan)
    one_sided = np.zeros(steps, dtype=bool)
    ps = sim.market.public
    start = time.perf_counter()
    for t in range(steps):
        sim.step()
        mid[t] = ps.mid
        if ps.best_bid is not None and ps.best_ask is not None:
            spread[t] = ps.spread
        else:
            one_sided[t] = True
    elapsed = time.perf_counter() - start
    sim.check_conservation()
    tape = [(s - offset, p, q, a, b, sl) for s, p, q, a, b, sl in sim.market.tape if s >= offset]
    records = [r for r in sim.metaorders.records if r.end is not None and r.start >= offset]
    for r in records:  # express in post-burn-in steps so they index ``mid``
        r.start -= offset
        r.planned_end -= offset
        r.end -= offset
    inst = {cid for name, cids in sim.groups.items() if name.startswith("Inst") for cid in cids}
    return SeedRun(seed, steps, mid, spread, one_sided, tape, records, dict(sim.market.budget_rejects),
                   steps / elapsed, inst)


# ----- statistics ------------------------------------------------------------ #

def autocorrelation(x: np.ndarray, lags) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = (x * x).sum()
    if denom == 0:
        return np.zeros(len(lags))
    return np.array([(x[:-k] * x[k:]).sum() / denom if 0 < k < len(x) else 1.0 for k in lags])


def kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    m2 = (x * x).mean()
    return float((x ** 4).mean() / (m2 * m2)) if m2 > 0 else float("nan")


def returns(mid: np.ndarray, horizon: int) -> np.ndarray:
    return np.diff(mid[::horizon])


def trade_signs(tape) -> np.ndarray:
    return np.array([1 if t[3] == "buy" else -1 for t in tape], dtype=float)


REFERENCE_HORIZON = 2_000  # steps; the executor's horizon, so Q/V is on the same axis as the execution task


def impact_curve(run: SeedRun, after: int = 500, reference_horizon: int = REFERENCE_HORIZON) -> list[dict]:
    """Per completed metaorder: Q/V, impact at completion, peak impact and the impact ``after`` steps later.

    V is the run's mean market volume per step times ``reference_horizon``,
    so Q/V spans the size distribution (volume over the metaorder's own life
    would pin Q/V to about participation / (1 + participation)). Impacts are
    signed mid changes from the arrival mid in ticks, positive in the
    metaorder's direction.
    """
    volume_per_step = sum(t[2] for t in run.tape) / run.steps
    reference_volume = volume_per_step * reference_horizon
    out = []
    for r in run.metaorders:
        if r.executed < 1 or r.end <= r.start or r.end + 1 >= run.steps or reference_volume <= 0:
            continue
        sign = 1 if r.side == "bid" else -1
        path = sign * (run.mid[r.start:r.end + 1] - r.arrival_mid)
        later = min(run.steps - 1, r.end + after)
        out.append({"id": r.id, "seed": run.seed, "style": r.style, "size": r.executed,
                    "q_over_v": r.executed / reference_volume, "participation": r.participation,
                    "duration": r.end - r.start, "peak_ticks": float(path.max()), "end_ticks": float(path[-1]),
                    "residual_ticks": float(sign * (run.mid[later] - r.arrival_mid)),
                    "shortfall_bps": r.shortfall_bps})
    return out


def fit_power_law(x: np.ndarray, y: np.ndarray, bins: int = 6) -> dict:
    """Bin by x (log-spaced quantiles), average y per bin (negatives included), fit log y = a + delta log x
    over the bins with a positive mean."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = x > 0
    x, y = x[keep], y[keep]
    if len(x) < 4:
        return {"delta": float("nan"), "n": int(len(x)), "bins": []}
    edges = np.quantile(np.log(x), np.linspace(0, 1, min(bins, len(x) // 2) + 1))
    idx = np.clip(np.searchsorted(edges, np.log(x), side="right") - 1, 0, len(edges) - 2)
    bx, by, bn = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum():
            bx.append(np.exp(np.log(x[sel]).mean()))
            by.append(y[sel].mean())
            bn.append(int(sel.sum()))
    bx, by = np.array(bx), np.array(by)
    ok = by > 0
    if ok.sum() < 2:
        return {"delta": float("nan"), "n": int(len(x)), "bins": list(zip(bx, by, bn))}
    delta, intercept = np.polyfit(np.log(bx[ok]), np.log(by[ok]), 1)
    return {"delta": float(delta), "intercept": float(intercept), "n": int(len(x)),
            "bins": [(float(a), float(b), int(c)) for a, b, c in zip(bx, by, bn)]}


def summarise(runs: list[SeedRun], sample_every: int = 10, horizons=(1, 5, 20, 100)) -> dict:
    """Aggregate statistics across seeds; every entry lists per-seed values and a pooled summary."""
    s: dict = {"seeds": [r.seed for r in runs], "steps": runs[0].steps}
    s["steps_per_second"] = float(np.mean([r.steps_per_second for r in runs]))
    s["median_spread"] = [float(np.nanmedian(r.spread)) for r in runs]
    s["one_sided_share"] = [float(r.one_sided.mean()) for r in runs]
    s["return_autocorr"] = {}
    lags = (1, 2, 5, 10, 20, 50)
    for r in runs:
        ret = returns(r.mid, sample_every)
        s["return_autocorr"][r.seed] = dict(zip(lags, autocorrelation(ret, lags).round(4).tolist()))
    s["max_abs_return_autocorr_from_lag5"] = [
        float(max(abs(v) for k, v in s["return_autocorr"][r.seed].items() if k >= 5)) for r in runs]
    s["kurtosis"] = {h: [kurtosis(returns(r.mid, h)) for r in runs] for h in horizons}
    abs_lags = (1, 5, 20, 50, 100, 200)
    s["abs_return_autocorr"] = {}
    for r in runs:
        ret = np.abs(returns(r.mid, sample_every))
        s["abs_return_autocorr"][r.seed] = dict(zip(abs_lags, autocorrelation(ret, abs_lags).round(4).tolist()))
    sign_lags = (1, 10, 100, 1000)
    s["trade_sign_autocorr"] = {}
    for r in runs:
        signs = trade_signs(r.tape)
        s["trade_sign_autocorr"][r.seed] = dict(zip(sign_lags, autocorrelation(signs, sign_lags).round(4).tolist()))
    curve = [row for r in runs for row in impact_curve(r)]
    s["metaorders"] = len(curve)
    s["impact"] = fit_power_law([c["q_over_v"] for c in curve], [c["end_ticks"] for c in curve])
    s["impact_peak"] = fit_power_law([c["q_over_v"] for c in curve], [c["peak_ticks"] for c in curve])
    # Pooled ratios (sum over metaorders), so that near-zero peaks do not dominate.
    peak_sum = sum(c["peak_ticks"] for c in curve)
    s["impact_remaining_share"] = sum(c["residual_ticks"] for c in curve) / peak_sum if peak_sum > 0 else float("nan")
    s["impact_end_share"] = sum(c["end_ticks"] for c in curve) / peak_sum if peak_sum > 0 else float("nan")
    s["metaorder_shortfall_bps"] = {
        style: float(np.mean([c["shortfall_bps"] for c in curve if c["style"] == style]))
        for style in sorted({c["style"] for c in curve})}
    volume_rows = [(c["participation"], c["size"] / max(c["duration"], 1)) for c in curve]
    s["metaorder_lots_per_step"] = float(np.mean([v for _, v in volume_rows])) if volume_rows else float("nan")
    rejects = [r.budget_rejects for r in runs]
    total_orders = sum(len(r.tape) for r in runs)
    s["budget_rejects"] = {"bid": sum(x["bid"] for x in rejects), "ask": sum(x["ask"] for x in rejects),
                           "per_trade": (sum(x["bid"] + x["ask"] for x in rejects) / total_orders) if total_orders else 0}
    s["impact_rows"] = curve
    return s


def check(s: dict) -> list[tuple[str, str, str, bool]]:
    """(check, target, measured, passed) rows against the starting targets of the plan."""
    rows = []
    med = float(np.median(s["median_spread"]))
    rows.append(("Liquidity: median spread", "1-3 ticks", f"{med:.2f}", 1 <= med <= 3))
    one = float(np.max(s["one_sided_share"]))
    rows.append(("Liquidity: one side empty", "< 0.1% of steps", f"{one * 100:.3f}%", one < 0.001))
    ac = float(np.max(s["max_abs_return_autocorr_from_lag5"]))
    rows.append(("Return autocorrelation", "|ac| < 0.05 from lag 5", f"max {ac:.3f}", ac < 0.05))
    k_short = float(np.median(s["kurtosis"][1]))
    k_long = float(np.median(s["kurtosis"][max(s["kurtosis"])]))
    rows.append(("Fat tails", "kurtosis >> 3 at short horizons, falling", f"{k_short:.1f} -> {k_long:.1f}",
                 k_short > 4 and k_long < k_short))
    vc = [list(v.values()) for v in s["abs_return_autocorr"].values()]  # lags 1, 5, 20, 50, 100, 200
    vc_ok = all(row[0] > 0 and row[2] > 0 and row[0] >= row[4] for row in vc)
    rows.append(("Volatility clustering", "ac(|r|) > 0 at lags 1-20, decaying by 100",
                 f"lag1 {np.mean([r[0] for r in vc]):.3f}, lag20 {np.mean([r[2] for r in vc]):.3f}, "
                 f"lag100 {np.mean([r[4] for r in vc]):.3f}", vc_ok))
    ts = [list(v.values()) for v in s["trade_sign_autocorr"].values()]
    ts_ok = all(row[0] > 0 and row[1] > 0 and row[0] >= row[1] for row in ts)
    rows.append(("Order flow memory", "sign ac > 0, slowly decaying",
                 f"lag1 {np.mean([r[0] for r in ts]):.3f}, lag10 {np.mean([r[1] for r in ts]):.3f}, "
                 f"lag100 {np.mean([r[2] for r in ts]):.3f}", ts_ok))
    if "paired_impact" in s:  # injected metaorders differenced against the same seed without them
        p = s["paired_impact"]
        delta, delta_peak = p["delta_end"], p["delta_peak"]
        rows.append(("Square-root impact (paired)", "delta in [0.4, 0.7]",
                     f"delta {delta:.2f} at completion, {delta_peak:.2f} at peak (n={p['n']})",
                     0.4 <= delta <= 0.7 if not math.isnan(delta) else False))
        rem, at_end = p["remaining_share_of_peak"], p["end_share_of_peak"]
        rows.append(("Impact decay (paired)", "partial reversion after the end",
                     f"end {at_end * 100:.0f}% of peak; {rem * 100:.0f}% left 500 steps later",
                     0 < rem < at_end if not math.isnan(rem) else False))
    delta, delta_peak = s["impact"]["delta"], s["impact_peak"]["delta"]
    rows.append(("Square-root impact (natural)", "delta in [0.4, 0.7]; noisy, informational",
                 f"delta {delta:.2f} at completion, {delta_peak:.2f} at peak (n={s['impact']['n']})",
                 0.4 <= delta <= 0.7 if not math.isnan(delta) else False))
    rem, at_end = s["impact_remaining_share"], s["impact_end_share"]
    rows.append(("Impact decay (natural)", "partial reversion; informational",
                 f"end {at_end * 100:.0f}% of peak; {rem * 100:.0f}% left 500 steps later" if not math.isnan(rem) else "n/a",
                 0 < rem < at_end if not math.isnan(rem) else False))
    br = s["budget_rejects"]
    total = br["bid"] + br["ask"]
    balanced = total < 20 or min(br["bid"], br["ask"]) / total > 0.2  # a handful of rejects cannot be "unbalanced"
    rows.append(("Budgets", "< 1% of trades rejected, balanced",
                 f"bid {br['bid']} ask {br['ask']} ({br['per_trade'] * 100:.3f}% of trades)",
                 br["per_trade"] < 0.01 and balanced))
    sps = s["steps_per_second"]
    rows.append(("Speed", ">= 10,000 steps/s (analysis mode)", f"{sps:,.0f}", sps >= 10_000))
    return rows


def run_validation(seeds, steps: int, config: SimConfig | None = None, burn_in: int = 2_000,
                   paired_impact: bool = True, impact_fractions=(0.01, 0.02, 0.05, 0.1, 0.2)) -> dict:
    """Stylised facts over ``seeds`` x ``steps`` plus, by default, the paired impact measurement."""
    runs = [run_seed(int(seed), steps, config, burn_in) for seed in seeds]
    summary = summarise(runs)
    if paired_impact:
        from order_book.impact import measure_impact, summarise_impact  # local import: impact imports this module
        impact_runs = measure_impact(seeds, impact_fractions, config=config, burn_in=burn_in)
        summary["paired_impact"] = {k: v for k, v in summarise_impact(impact_runs).items()}
    return summary


def format_table(rows) -> str:
    width = max(len(r[0]) for r in rows)
    lines = [f"{'check':{width}}  {'target':38}  {'measured':42}  result"]
    for name, target, measured, ok in rows:
        lines.append(f"{name:{width}}  {target:38}  {measured:42}  {'PASS' if ok else 'FAIL'}")
    return "\n".join(lines)
