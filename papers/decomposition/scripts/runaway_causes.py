"""What drives the price runaways: switch each candidate off after the arrival, in both worlds, on the same seeds.

Every run has all channels on and the same market up to the arrival, so a change in cost comes from the intervention.

    baseline        the market as it is
    no_inv_limit    makers have no inventory limit
    no_vol_pull     makers neither pull on a volatility spike nor shrink their quotes before one
    no_switch       speculators keep the strategy they held at the arrival
    spec_capacity   speculators have effectively unlimited capital and no position cap; this also mutes switching,
                    because fitness is a return on wealth
    noise_budget    the noise pool holds 100 times as much, in the same proportions
    all_three       no_switch, spec_capacity and noise_budget together

The interventions are descriptive: they do not go through the ghost arm, so they give costs and no shares.

    python -m papers.decomposition.scripts.runaway_causes --workers 8
    python -m papers.decomposition.scripts.runaway_causes --quick

Outputs: outputs/decomposition/runaway_causes/episodes.jsonl and summary.json.
"""

import argparse
import json
from dataclasses import dataclass, replace

import numpy as np

from order_book.agents import MarketMaker, NoiseFlow
from order_book.counterfactual import FULL, PairedRun, PairedSpec, burned_in
from order_book.execution import ExecutionTask, make_strategy
from order_book.experiments import DEFAULT_REGIMES
from order_book.provenance import write_manifest
from order_book.runner import iter_jobs
from papers.decomposition import OUT
from scripts._common import calibration_or_default, write_json

VARIANTS = ("baseline", "no_inv_limit", "no_vol_pull", "no_switch", "spec_capacity", "noise_budget", "all_three")
INTERVENTIONS = {"no_inv_limit": ("limit",), "no_vol_pull": ("pull",), "no_switch": ("switch",), "spec_capacity": ("spec",),
                 "noise_budget": ("noise",), "all_three": ("switch", "spec", "noise")}
RUNAWAY_TICKS = 500         # peak impact above this is a runaway: 5% of a mid of 10,000 ticks
EXTRA_SPECULATOR_SHARES = 100_000
NOISE_SCALE = 100


@dataclass(frozen=True)
class RunawayConfig:
    cells: tuple = (("immediate", 0.2, 2000), ("immediate", 0.4, 2000), ("limit_market", 0.4, 2000), ("pov", 0.4, 2000),
                    ("twap_market", 0.4, 2000), ("twap_market", 0.2, 8000), ("twap_market", 0.4, 8000), ("pov", 0.4, 8000))
    seeds: tuple = tuple(range(1000, 1020))     # where the runaways were found; this describes them and selects nothing
    regime: str = "deep_cover"
    burn_in: int = 2_000
    after: int = 500

    def jobs(self) -> list[tuple]:
        return [(seed, *cell, variant, self) for cell in self.cells for variant in VARIANTS for seed in self.seeds]


def intervene(sim, variant: str) -> None:
    """Applied after the burn-in, so the market up to the arrival is the same in every variant."""
    what = INTERVENTIONS.get(variant, ())
    for agent in sim.agents:
        if isinstance(agent, MarketMaker):
            if "limit" in what:
                agent.cfg = replace(agent.cfg, max_inventory=10**9)
            if "pull" in what:
                agent.cfg = replace(agent.cfg, pull_ratio=float("inf"), fragility=0.0)
        if isinstance(agent, NoiseFlow) and "noise" in what:
            agent.account.cash *= NOISE_SCALE
            agent.account.shares *= NOISE_SCALE
    speculators = sim.speculators
    if "switch" in what:
        speculators._switch = lambda: None
    if "spec" in what:
        speculators.max_position = 10**9
        speculators.target_shares += EXTRA_SPECULATOR_SHARES
        for account in speculators.accounts:
            account.shares += EXTRA_SPECULATOR_SHARES
            account.cash += int(EXTRA_SPECULATOR_SHARES * sim.market.public.mid)


def run_one(job: tuple) -> dict:
    seed, strategy, size, horizon, variant, cfg = job
    regime = next(r for r in DEFAULT_REGIMES if r.name == cfg.regime)
    spec = PairedSpec(seed, strategy, size, FULL, horizon=horizon, burn_in=cfg.burn_in, after=cfg.after, base_config=regime.base(), regime=regime.name)
    blob, volume = burned_in(seed, spec.config(), spec.burn_in)
    run = PairedRun(blob, FULL, None)
    for sim in (run.reference, run.treated):
        intervene(sim, variant)
    task = ExecutionTask(side=spec.resolved_side, size_fraction=size, horizon=horizon, decision_interval=spec.decision_interval)
    executor = run.add_executor(task, make_strategy(strategy, calibration_or_default()), volume)
    public, reference = run.market.public, run.reference.market.public
    sign = 1 if task.side == "bid" else -1
    peak = end = 0.0
    for _ in range(spec.window):
        run.step()
        end = sign * (public.mid - reference.mid)
        peak = max(peak, end)
    if not executor.done:
        executor._finish()
    r = executor.result
    return {"seed": seed, "strategy": strategy, "size": size, "horizon": horizon, "variant": variant, "quantity": r.quantity,
            "shortfall_bps": r.shortfall_bps, "completion": r.completion, "peak": peak, "end": end}


def summarise(records: list[dict]) -> dict:
    cells: dict = {}
    for r in records:
        cells.setdefault(f"{r['strategy']}|{r['size']}|{r['horizon']}", {}).setdefault(r["variant"], []).append(r)
    summary = {}
    for key, variants in cells.items():
        summary[key] = {}
        for variant, runs in variants.items():
            cost = np.array([r["shortfall_bps"] for r in runs], float)
            peak = np.array([r["peak"] for r in runs])
            summary[key][variant] = {"mean": float(np.nanmean(cost)), "median": float(np.nanmedian(cost)), "max": float(np.nanmax(cost)),
                                     "runaways": int((peak > RUNAWAY_TICKS).sum()), "largest_peak": float(peak.max()), "episodes": len(runs)}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = RunawayConfig()
    if args.quick:
        cfg = replace(cfg, cells=(("twap_market", 0.2, 300),), seeds=(1000, 1001), burn_in=500, after=100)
    out = OUT / ("runaway_causes_quick" if args.quick else "runaway_causes")  # a smoke test never overwrites real results
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, cfg)
    jobs = cfg.jobs()
    print(f"[runaway] {len(jobs)} runs: {len(cfg.cells)} cells x {len(VARIANTS)} variants x {len(cfg.seeds)} seeds")
    records = []
    with open(out / "episodes.jsonl", "w") as sink:
        for record in iter_jobs(run_one, jobs, args.workers, chunksize=4):
            records.append(record)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
    summary = summarise(records)
    write_json({"summary": summary}, out / "summary.json")
    for key, variants in summary.items():
        print(key)
        for variant, s in variants.items():
            print(f"  {variant:<15} mean {s['mean']:8.1f}  median {s['median']:7.1f}  runaways {s['runaways']:2d}/{s['episodes']}  largest peak {s['largest_peak']:7.0f}")


if __name__ == "__main__":
    main()
