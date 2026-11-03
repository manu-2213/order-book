"""How much any executor can gain over TWAP in the market the policies were trained in, and how much stealth is worth.

Two questions a learned executor's result raises: is the agent handicapped, or is there nothing to win? The first part runs
every baseline, dialled TWAPs, momentum-timed TWAPs and a post-and-hold passive TWAP against ``twap_market`` on the unseen
seeds, paired, in the full market: the margin available to tuned heuristics. The second part asks what detection and
predation would cost TWAP under a more sensitive adversary than this market's: a lenient detector threshold, makers and the
anticipator reacting three times as strongly, and a larger order.

    python -m papers.decomposition.scripts.rl_ceiling --workers 16
    python -m papers.decomposition.scripts.rl_ceiling --regime low_cover_large_tick --workers 16
    python -m papers.decomposition.scripts.rl_ceiling --quick

Outputs: outputs/decomposition/rl_ceiling/{episodes.jsonl,summary.json}, or rl_ceiling_<market>/ for another regime.
"""

import argparse
import json
import math
from dataclasses import asdict, replace

import numpy as np

from order_book.counterfactual import ARMS, FULL, PairedSpec, run_paired
from order_book.execution import make_strategy
from order_book.execution.task import Decision, ExecState, Strategy
from order_book.provenance import write_manifest
from order_book.runner import iter_jobs
from order_book.scoring import bootstrap_ci, holm, paired_difference
from order_book.training import FIRST_TEST_SEED, PairedExecutionEnv
from papers.decomposition import OUT
from papers.decomposition.scripts.train_rl import REGIMES
from scripts._common import write_json

DEFAULT_MARKET = "low_cover_capital10"
SIZES = (0.2, 0.4)
UNSEEN = tuple(range(FIRST_TEST_SEED, FIRST_TEST_SEED + 80))   # the seeds rl_eval scores the policies on
DECOMPOSED = tuple(range(1020, 1040))                           # the seeds e8_rl decomposes them on


class SignalTwap(Strategy):
    """TWAP whose pace follows the fast momentum in the executor's direction: ``k > 0`` trades more while the price moves
    against it (momentum), ``k < 0`` while it moves its way (reversion)."""

    def __init__(self, k: float, lookback: int = 0, scale: float = 0.5):
        self.k, self.lookback, self.scale = k, lookback, scale
        self.name = f"signal_twap_k{k:g}_l{lookback}"

    def decide(self, s: ExecState, rng) -> Decision:
        m = s.sign * s.public.momentum[self.lookback]
        mult = min(2.5, max(0.0, 1 + self.k * math.tanh(m / self.scale)))
        return Decision(cancel=True, market_qty=min(s.remaining, math.ceil(s.remaining / s.decisions_left * mult)))


class PostAndHold(Strategy):
    """Posts a slice at own best (or ``behind`` ticks behind it) and leaves it resting, so queue priority is kept; market
    orders only catch up the schedule, and the last two decisions cross what is left."""

    def __init__(self, behind: int = 0):
        self.behind = behind
        self.name = f"post_and_hold_{behind}"

    def decide(self, s: ExecState, rng) -> Decision:
        if s.decisions_left <= 2:
            return Decision(cancel=True, market_qty=s.remaining)
        catch_up = max(0, math.floor(s.quantity * s.elapsed / s.horizon) - s.executed - s.own_resting)
        slice_qty = math.ceil(max(0, s.remaining - s.own_resting - catch_up) / s.decisions_left)
        return Decision(cancel=False, market_qty=catch_up, limit_qty=slice_qty, limit_price=None if self.behind == 0 else s.behind(self.behind))


HEURISTICS = ("twap_market", "twap_passive", "twap_random", "pov@0.2", "pov@0.3", "pov@0.5", "almgren_chriss_1e-05", "almgren_chriss_1e-06",
              "almgren_chriss_1e-07", "obizhaeva_wang", "limit_market", "footprint", "mimic", "twap@passive=0.25", "twap@passive=0.5",
              "twap@passive=0.75", "twap@passive=1", "twap@timing=0.3", "twap@size=0.5", "twap@size=0.5,timing=0.3,passive=0.5",
              "signal@+0.5", "signal@-0.5", "signal@+1", "signal@-1", "signal@+0.5,1", "signal@-0.5,1", "hold@0", "hold@1")


def strategy(name: str) -> Strategy:
    if name.startswith("signal@"):
        k, _, lag = name[7:].partition(",")
        return SignalTwap(float(k), int(lag) if lag else 0)
    if name.startswith("hold@"):
        return PostAndHold(int(name[5:]))
    return make_strategy(name)


def stronger(cfg, k: float):
    """The makers' and the anticipator's response to detection scaled by ``k``."""
    pop = cfg.population
    makers = tuple(replace(m, detect_skew=m.detect_skew * k, detect_widen=m.detect_widen * k, detect_size_cut=min(1.0, m.detect_size_cut * k))
                   for m in pop.market_makers)
    return replace(cfg, population=replace(pop, market_makers=makers), anticipator=replace(cfg.anticipator, rate=cfg.anticipator.rate * k))


def adversaries(base) -> dict:
    return {"medium": ("medium", base, 0.4), "lenient": ("lenient", base, 0.4), "lenient_x3": ("lenient", stronger(base, 3), 0.4),
            "lenient_x3_0.6": ("lenient", stronger(base, 3), 0.6)}


_ENVS: dict = {}  # per worker, by (horizon, burn_in): workers are spawned, so settings travel in the job


def _heuristic(job):
    name, seed, size, horizon, burn_in, market = job
    env = _ENVS.get((horizon, burn_in, market))
    if env is None:
        env = _ENVS[(horizon, burn_in, market)] = PairedExecutionEnv(configs=((market, REGIMES[market].base()),), horizon=horizon, burn_in=burn_in)
    _, r = env.rollout(strategy(name), seed, size)
    return {"part": "ceiling", "strategy": name, **asdict(r)}


def _adversary(job):
    name, seed, arm_key, horizon, burn_in, market = job
    threshold, cfg, size = adversaries(REGIMES[market].base())[name]
    arm = next(a for a in (FULL, ARMS[3], ARMS[1]) if a.key == arm_key)
    r = run_paired(PairedSpec(seed, "twap_market", size, arm, horizon=horizon, burn_in=burn_in, after=50, threshold=threshold,
                              base_config=cfg, regime=name))
    return {"part": "adversary", "variant": name, "seed": seed, "size": size, "arm": arm_key, "cost_is": r.shortfall_bps, "drift": r.drift_bps,
            "firing": r.flow_firing}


def summarise(rows: list[dict]) -> dict:
    ceiling = [r for r in rows if r["part"] == "ceiling"]
    index = {(r["strategy"], r["size"], r["seed"]): r for r in ceiling}
    seeds = sorted({r["seed"] for r in ceiling})
    sizes = sorted({r["size"] for r in ceiling})
    twap = lambda size, key: [index[("twap_market", size, s)][key] for s in seeds]
    pooled = lambda name, key: np.mean([[index[(name, z, s)][key] for s in seeds] for z in sizes], axis=0)
    out = {}
    for name in dict.fromkeys(r["strategy"] for r in ceiling):
        if name == "twap_market":
            continue
        top = [index[(name, sizes[-1], s)] for s in seeds]
        out[name] = {"by_size": {str(z): paired_difference([index[(name, z, s)]["cost_is"] for s in seeds], twap(z, "cost_is")) for z in sizes},
                     "pooled": paired_difference(pooled(name, "cost_is"), pooled("twap_market", "cost_is")),
                     "net": paired_difference(pooled(name, "cost_net"), pooled("twap_market", "cost_net")),
                     "passive": float(np.mean([1 - r["aggressive_share"] for r in top])), "forced": float(np.mean([r["forced"] / r["quantity"] for r in top])),
                     "exposure": float(np.mean([r["exposure"] for r in top]))}
    for row, p in zip(out.values(), holm([row["pooled"]["p_two_sided"] for row in out.values()])):
        row["pooled"]["p_holm"] = p
    adversary = [r for r in rows if r["part"] == "adversary"]
    aidx = {(r["variant"], r["seed"], r["arm"]): r for r in adversary}
    probe = {}
    for name in dict.fromkeys(r["variant"] for r in adversary):
        aseeds = sorted({r["seed"] for r in adversary if r["variant"] == name})
        full, ip, ghost = ({arm: np.array([aidx[(name, s, arm)]["cost_is"] for s in aseeds]) for arm in (FULL.key, ARMS[3].key, ARMS[1].key)}[k] for k in (FULL.key, ARMS[3].key, ARMS[1].key))
        drift = np.array([aidx[(name, s, FULL.key)]["drift"] for s in aseeds])
        probe[name] = {"size": aidx[(name, aseeds[0], FULL.key)]["size"], "is": bootstrap_ci(full), "median_is": float(np.median(full)),
                       "detection_predation": bootstrap_ci(full - ip), "reaction": bootstrap_ci(full - ghost), "mechanical": bootstrap_ci(ghost - drift),
                       "firing": float(np.mean([aidx[(name, s, FULL.key)]["firing"] for s in aseeds])), "n": len(aseeds)}
    twap_cost = {str(z): float(np.mean(twap(z, "cost_is"))) for z in sizes}
    return {"market": ceiling[0]["regime"], "seeds": [seeds[0], seeds[-1]], "twap_is": twap_cost, "ceiling": out, "adversary": probe}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default=DEFAULT_MARKET, choices=sorted(REGIMES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    market = args.regime
    names, sizes, seeds, decomposed, horizon, burn_in = HEURISTICS, SIZES, UNSEEN, DECOMPOSED, 2_000, 2_000
    if args.quick:
        names, sizes, seeds, decomposed, horizon, burn_in = ("twap_market", "twap_passive", "signal@+0.5", "hold@0"), (0.4,), UNSEEN[:2], DECOMPOSED[:2], 200, 400
    name = "rl_ceiling" if market == DEFAULT_MARKET else f"rl_ceiling_{market.removeprefix('low_cover_')}"
    out = OUT / (f"{name}_quick" if args.quick else name)  # a smoke test never overwrites real results
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, {"market": market, "heuristics": names, "sizes": sizes, "seeds": [seeds[0], seeds[-1]], "adversary_seeds": [decomposed[0], decomposed[-1]],
                         "horizon": horizon, "burn_in": burn_in})
    jobs = [(n, s, z, horizon, burn_in, market) for n in names for z in sizes for s in seeds]
    probes = [(n, s, arm.key, horizon, burn_in, market) for n in adversaries(REGIMES[market].base()) for s in decomposed for arm in (FULL, ARMS[3], ARMS[1])]
    rows = []
    with open(out / "episodes.jsonl", "w") as sink:
        for part, fn, work in (("ceiling", _heuristic, jobs), ("adversary", _adversary, probes)):
            for record in iter_jobs(fn, work, args.workers, 4):
                rows.append(record)
                sink.write(json.dumps(record) + "\n")
            print(f"[rl_ceiling] {part}: {len(work)} runs", flush=True)
    summary = summarise(rows)
    write_json(summary, out / "summary.json")
    for name, row in sorted(summary["ceiling"].items(), key=lambda kv: kv[1]["pooled"]["mean"]):
        p = row["pooled"]
        print(f"{name:36} IS - TWAP pooled {p['mean']:+.2f} [{p['low']:+.2f},{p['high']:+.2f}] Holm {p['p_holm']:.2f}  passive {row['passive']:.2f} forced {row['forced']:.3f}")
    for name, row in summary["adversary"].items():
        print(f"{name:16} Q/V {row['size']} IS {row['is'][0]:.2f}  detection+predation {row['detection_predation'][0]:+.2f} [{row['detection_predation'][1]:+.2f},{row['detection_predation'][2]:+.2f}]"
              f"  reaction {row['reaction'][0]:+.2f}  firing {row['firing']:.3f}")


if __name__ == "__main__":
    main()
