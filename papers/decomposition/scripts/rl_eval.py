"""The trained policies against TWAP on seeds they were selected on and on seeds nobody has seen, in the paired worlds.

Each policy runs greedily on every size and market it was trained for. The validation group reproduces the selection;
the unseen group is the estimate to quote. Differences are paired by episode, in basis points of the arrival mid: the
shortfall and the cost net of natural drift, with the executor's exposure to the makers' detector alongside. The analysis
works per seed: Holm across the cells, each policy pooled over sizes in one market, GRPO against PPO, and whether a policy's
drift gain follows what TWAP paid in drift that seed (front-loading into a trend) or holds at any drift (timing).

    python -m papers.decomposition.scripts.rl_eval --workers 16
    python -m papers.decomposition.scripts.rl_eval --sweep-set large_tick --workers 16
    python -m papers.decomposition.scripts.rl_eval --policies outputs/decomposition/rl/ppo/best_policy.pt --workers 8

Outputs: outputs/decomposition/rl_eval/{summary.json,analysis.json,episodes.jsonl}, or rl_eval_<set>/ for another sweep set.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from order_book.execution import ACTIONS, PolicyNet
from order_book.provenance import write_manifest
from order_book.scoring import bootstrap_ci, holm, paired_difference
from order_book.training import FIRST_TEST_SEED, PairedExecutionEnv, Rollouts, VALIDATION_SEEDS, compare, cpu_state_dict
from papers.decomposition import OUT
from papers.decomposition.scripts.train_rl import REGIMES, SWEEPS, SWEEP_MARKET
from scripts._common import write_json


PAIRS = (("grpo", "ppo"), ("grpo_detect0.03", "ppo_detect0.03"), ("grpo_mixed", "ppo_mixed"), ("grpo_seed1", "ppo_seed1"),
         ("lt_grpo", "lt_ppo"), ("lt_grpo_seed1", "lt_ppo_seed1"))


def policy_runs(root: Path, sweep: tuple) -> list[Path]:
    """The sweep's training runs under ``root``, by their best checkpoint; leftovers from other sweeps are ignored. Refuses
    unfinished runs: a chain that ran out of hours would otherwise be scored as if it had converged."""
    runs = [root / row["name"] for row in sweep if (root / row["name"] / "history.json").exists()]
    unfinished = [run.name for run in runs if json.loads((run / "history.json").read_text())["status"] != "done"]
    if unfinished:
        raise SystemExit(f"training not finished for {unfinished}: add links (CHAIN=2 bash hpc/submit.sh decomposition train_rl), then submit_all.sh rl_results")
    return [run / "best_policy.pt" for run in runs]


def episode_row(name: str, record, transitions=()) -> dict:
    return {"name": name, **asdict(record), "hold": float(np.mean([ACTIONS[t.action][0] == 0 for t in transitions])) if transitions else 0.0}


def evaluate(policy: Path, groups: dict, workers: int, quick: bool, twap: dict, episodes: list) -> list[dict]:
    """One policy on every (market, size, seed) of ``groups``, appended to ``episodes``. ``twap`` caches TWAP's episodes across
    policies by episode."""
    history = json.loads((policy.parent / "history.json").read_text())
    cfg = history["config"]
    horizon = 200 if quick else cfg["horizon"]
    env = PairedExecutionEnv(configs=tuple((name, REGIMES[name].base()) for name in cfg["regimes"]), sizes=tuple(cfg["sizes"]), horizon=horizon,
                             burn_in=400 if quick else cfg["burn_in"])
    weights = cpu_state_dict(PolicyNet.load(policy))
    rollouts = Rollouts(env, workers)
    rows = []
    try:
        for label, seeds in groups.items():
            jobs = [(seed, size, ci) for ci in range(len(env.configs)) for size in env.sizes for seed in seeds]
            key = lambda seed, size, ci: (env.configs[ci][0], size, seed, horizon, env.burn_in)
            missing = [job for job in jobs if key(*job) not in twap]
            for job, (_, r) in zip(missing, rollouts.run([("twap_market", True, *job, 0) for job in missing])):
                twap[key(*job)] = r
            results = rollouts.run([(weights, True, *job, 0) for job in jobs])
            learned = [r for _, r in results]
            episodes += [episode_row(policy.parent.name, r, ts) for ts, r in results]
            for ci, (regime, _) in enumerate(env.configs):
                for size in env.sizes:
                    a = [r for r in learned if r.regime == regime and r.size == size]
                    b = [twap[(regime, size, r.seed, horizon, env.burn_in)] for r in a]
                    c = compare(a, b)
                    rows.append({"policy": str(policy), "name": policy.parent.name, "method": cfg["method"], "reward": cfg["reward"], "regime": regime,
                                 "size": size, "seeds": label, "n": len(a), **c})
                    print(f"{policy.parent.name:<16} {regime:<22} {size:.2f} {label:<10} n={len(a):<3} IS policy-TWAP {c['gap_is']:+.2f} [{c['gap_is_low']:+.2f},{c['gap_is_high']:+.2f}]  "
                          f"net {c['gap_net']:+.2f} [{c['gap_net_low']:+.2f},{c['gap_net_high']:+.2f}]  exposure {c['exposure']:.3f} vs {c['twap_exposure']:.3f}", flush=True)
    finally:
        rollouts.close()
    return rows


def analyse(episodes: list[dict], reference: str | None = None) -> dict:
    """The unseen seeds, per seed, in the one market every policy of the set is scored in (by default the first policy's).
    Inference resamples seeds; every family of tests is Holm-adjusted."""
    REFERENCE = reference or next(e["regime"] for e in episodes if e["name"] != "twap_market")
    index = {(e["name"], e["regime"], e["size"], e["seed"]): e for e in episodes}
    seeds = sorted({e["seed"] for e in episodes if e["seed"] >= FIRST_TEST_SEED})
    names = [n for n in dict.fromkeys(e["name"] for e in episodes) if n != "twap_market"]
    sizes = sorted({e["size"] for e in episodes})
    values = lambda name, regime, size, key: [index[(name, regime, size, s)][key] for s in seeds]
    cells = [(n, r, z) for n in names for r in sorted({e["regime"] for e in episodes}) for z in sizes if (n, r, z, seeds[0]) in index]
    gaps = [paired_difference(values(n, r, z, "cost_is"), values("twap_market", r, z, "cost_is")) for n, r, z in cells]
    pooled = lambda name, key: np.mean([values(name, REFERENCE, z, key) for z in sizes], axis=0)
    twap = {key: pooled("twap_market", key) for key in ("cost_is", "cost_net", "drift", "exposure")}
    policies = {}
    for name in names:
        shortfall = pooled(name, "cost_is") - twap["cost_is"]
        drift = pooled(name, "drift") - twap["drift"]
        fit = lambda i: np.polyfit(twap["drift"][i.astype(int)], drift[i.astype(int)], 1)
        everything = np.arange(len(seeds))
        policies[name] = {"is": paired_difference(pooled(name, "cost_is"), twap["cost_is"]), "median_is": float(np.median(shortfall)),
                          "net": paired_difference(pooled(name, "cost_net"), twap["cost_net"]), "drift": paired_difference(pooled(name, "drift"), twap["drift"]),
                          "drift_slope": float(fit(everything)[0]), "drift_intercept": bootstrap_ci(everything, lambda i: fit(i)[1]),
                          "exposure": paired_difference(pooled(name, "exposure"), twap["exposure"]), "mean_exposure": float(pooled(name, "exposure").mean()),
                          "forced": float(np.mean([e["forced"] / e["quantity"] for e in episodes if e["name"] == name and e["regime"] == REFERENCE and e["seed"] in seeds])),
                          "hold": float(pooled(name, "hold").mean()), "passive": float(1 - pooled(name, "aggressive_share").mean())}
    for row, p in zip(policies.values(), holm([row["is"]["p_two_sided"] for row in policies.values()])):
        row["is"]["p_holm"] = p
    pairs = [(a, b) for a, b in PAIRS if a in policies and b in policies]
    out = {"seeds": [seeds[0], seeds[-1]], "market": REFERENCE, "twap_exposure": float(twap["exposure"].mean()), "policies": policies,
           "cells": [{"name": n, "regime": r, "size": z, **g, "p_holm": p} for (n, r, z), g, p in zip(cells, gaps, holm([g["p_two_sided"] for g in gaps]))],
           "grpo_minus_ppo": {a: {key: paired_difference(pooled(a, key), pooled(b, key)) for key in ("cost_is", "cost_net", "drift")} for a, b in pairs}}
    if pairs:
        family = lambda members: np.mean([pooled(n, "cost_is") for n in members], axis=0)
        grpo, ppo = family([a for a, _ in pairs]), family([b for _, b in pairs])
        out["families"] = {"grpo_minus_ppo": paired_difference(grpo, ppo), "grpo_minus_twap": paired_difference(grpo, twap["cost_is"]), "ppo_minus_twap": paired_difference(ppo, twap["cost_is"])}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=None, help="best_policy.pt files; default: the sweep set's runs under outputs/decomposition/rl/")
    parser.add_argument("--sweep-set", default="base", choices=sorted(SWEEPS))
    parser.add_argument("--unseen-seeds", type=int, default=80, help=f"seeds from {FIRST_TEST_SEED}")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    root = OUT / ("rl_quick" if args.quick else "rl")
    policies = [Path(p) for p in args.policies] if args.policies else policy_runs(root, SWEEPS[args.sweep_set])
    if not policies:
        raise SystemExit(f"no trained policies under {root}")
    groups = {"validation": VALIDATION_SEEDS, "unseen": tuple(range(FIRST_TEST_SEED, FIRST_TEST_SEED + args.unseen_seeds))}
    if args.quick:
        groups = {"validation": VALIDATION_SEEDS[:2], "unseen": (FIRST_TEST_SEED, FIRST_TEST_SEED + 1)}
    name = "rl_eval" if args.sweep_set == "base" else f"rl_eval_{args.sweep_set}"
    out = OUT / (f"{name}_quick" if args.quick else name)  # a smoke test never overwrites real results
    write_manifest(out, vars(args))
    rows, twap, episodes = [], {}, []
    for policy in policies:
        rows += evaluate(policy, groups, args.workers, args.quick, twap, episodes)
    episodes += [episode_row("twap_market", r) for r in twap.values()]
    with open(out / "episodes.jsonl", "w") as sink:
        for episode in episodes:
            sink.write(json.dumps(episode) + "\n")
    write_json(rows, out / "summary.json")
    write_json(analyse(episodes, None if args.policies else SWEEP_MARKET[args.sweep_set]), out / "analysis.json")


if __name__ == "__main__":
    main()
