"""What the market looks like with no executor, and how configuration changes move it and a few test orders.

Spread, depth, volume and volatility at each horizon, then five orders with every channel on. These are tuning seeds, so
nothing is scored on seeds it was chosen on.

    python -m papers.decomposition.scripts.market_probe --workers 8
    python -m papers.decomposition.scripts.market_probe --quick

Outputs: outputs/decomposition/market_probe/episodes.jsonl.
"""

import argparse
import json
from dataclasses import replace

import numpy as np

from order_book.agents import OrderAnticipator
from order_book.counterfactual import FULL, PairedSpec, run_paired
from order_book.experiments import RegimeSpec
from order_book.presets import adversary
from order_book.provenance import write_manifest
from order_book.runner import iter_jobs
from order_book.simulator import Simulator
from papers.decomposition import OUT
from scripts._common import calibration_or_default

BASE = RegimeSpec("deep_cover", 1.0, 4).base()
VARIANTS = {"base": {}, "capital10": dict(capital=10), "deep5": dict(deep=5), "window100": dict(levels=100), "sigma1": dict(sigma=1.0), "sigma3": dict(sigma=3.0),
            "kappa0.25": dict(kappa=0.25), "offset10": dict(offset=10.0), "cap10_deep5": dict(capital=10, deep=5), "cap10_sig1": dict(capital=10, sigma=1.0),
            "cap10_win100": dict(capital=10, levels=100), "cap10_deep5_win100_sig1": dict(capital=10, deep=5, levels=100, sigma=1.0),
            "cap10_deep5_win100_sig3_off10": dict(capital=10, deep=5, levels=100, sigma=3.0, offset=10.0),
            "large_tick": dict(capital=10, half=3.0, improve=False)}  # the market the learned executor has a margin in
ORDERS = (("twap_market", 0.2, 2000), ("twap_market", 0.4, 2000), ("twap_market", 0.4, 8000), ("immediate", 0.4, 2000), ("pov", 0.4, 2000))
HORIZONS = (1, 10, 60, 600, 2000, 8000)
BURN_IN, WINDOW, SAMPLE_EVERY = 2_000, 12_000, 25


def market(capital=1, deep=1, levels=None, sigma=None, kappa=None, offset=None, half=None, improve=True):
    cfg = BASE
    if half is not None or not improve:
        makers = tuple(replace(m, min_half_spread=half if half is not None else m.min_half_spread) for m in cfg.population.market_makers)
        cfg = replace(cfg, population=replace(cfg.population, market_makers=makers), noise=replace(cfg.noise, improve=improve),
                      speculators=replace(cfg.speculators, improve=improve))
    if capital != 1:
        cfg = replace(cfg, endowment=replace(cfg.endowment, speculator=cfg.endowment.speculator * capital, noise=cfg.endowment.noise * capital))
    if deep != 1 or kappa is not None:
        def maker(m):
            if deep != 1:
                m = replace(m, size=(m.size[0] * deep, m.size[1] * deep), max_inventory=m.max_inventory * deep)
            return replace(m, kappa=m.kappa if kappa is None else kappa)
        cfg = replace(cfg, population=replace(cfg.population, market_makers=tuple(maker(m) for m in cfg.population.market_makers)))
        if deep != 1:
            cfg = replace(cfg, endowment=replace(cfg.endowment, maker=cfg.endowment.maker * deep))
    if levels:
        cfg = replace(cfg, book=replace(cfg.book, num_levels=levels))
    if sigma:
        cfg = replace(cfg, fundamental=replace(cfg.fundamental, sigma=sigma))
    if offset:
        cfg = replace(cfg, noise=replace(cfg.noise, mean_offset=offset))
    return cfg


def run_market(job: tuple) -> dict:
    variant, seed, window = job
    sim = Simulator(seed, replace(adversary("A4", "medium", market(**VARIANTS[variant]), None), rng="counter"))
    for agent in sim.agents:
        if isinstance(agent, OrderAnticipator):
            agent.active = False
    sim.run(BURN_IN)
    public, book = sim.market.public, sim.book
    mids, spreads, volume = np.empty(window), np.empty(window), 0
    best, within_5, total = [], [], []
    for t in range(window):
        sim.run(1)
        mids[t], spreads[t] = public.mid, public.spread
        volume += public.step_volume
        if t % SAMPLE_EVERY == 0:
            for side in ("bid", "ask"):
                levels = book.levels(side)
                if levels:
                    prices, sizes = np.array(levels).T
                    away = np.abs(prices - prices[0])
                    best.append(sizes[away < 1].sum()), within_5.append(sizes[away < 5].sum()), total.append(sizes.sum())
    record = {"kind": "market", "variant": variant, "seed": seed, "spread_mean": float(spreads.mean()), "spread_is_one_tick": float((spreads <= 1).mean()),
              "volume_per_step": volume / window, "depth_best": float(np.mean(best)), "depth_5": float(np.mean(within_5)), "depth_all": float(np.mean(total))}
    for h in HORIZONS:
        if h < window:
            record[f"sd_{h}"] = float(np.std(mids[h:] - mids[:-h]))
    return record


def run_order(job: tuple) -> dict:
    variant, seed, strategy, size, horizon = job
    spec = PairedSpec(seed, strategy, size, FULL, horizon=horizon, base_config=market(**VARIANTS[variant]), regime="probe")
    r = run_paired(spec, calibration_or_default())
    return {"kind": "order", "variant": variant, "seed": seed, "strategy": strategy, "size": size, "horizon": horizon, "quantity": r.quantity,
            "shortfall_bps": r.shortfall_bps, "peak": r.impact_peak, "q_over_v": r.q_over_v, "completion": r.completion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    variants, seeds, orders, window = list(VARIANTS), range(0, 20), ORDERS, WINDOW
    if args.quick:
        variants, seeds, orders, window = ["base", "capital10"], range(0, 2), (("twap_market", 0.2, 300),), 700
    out = OUT / ("market_probe_quick" if args.quick else "market_probe")  # a smoke test never overwrites real results
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, {"variants": {v: VARIANTS[v] for v in variants}, "seeds": list(seeds), "orders": orders, "window": window})
    with open(out / "episodes.jsonl", "w") as sink:
        for record in iter_jobs(run_market, [(v, s, window) for v in variants for s in seeds], args.workers, 2):
            sink.write(json.dumps(record) + "\n")
        for record in iter_jobs(run_order, [(v, s, *o) for v in variants for o in orders for s in seeds], args.workers, 4):
            sink.write(json.dumps(record) + "\n")
    print(f"saved {out / 'episodes.jsonl'}")


if __name__ == "__main__":
    main()
