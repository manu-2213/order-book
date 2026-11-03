"""The trained policies against TWAP on seeds they were selected on and on seeds nobody has seen, in the paired worlds.

Each policy runs greedily on every size and market it was trained for. The validation group reproduces the selection;
the unseen group is the estimate to quote. Differences are paired by episode, in basis points of the arrival mid: the
shortfall and the cost net of natural drift, with the executor's exposure to the makers' detector alongside.

    python -m papers.decomposition.scripts.rl_eval --workers 16
    python -m papers.decomposition.scripts.rl_eval --policies outputs/decomposition/rl/ppo/best_policy.pt --workers 8

Outputs: outputs/decomposition/rl_eval/summary.json.
"""

import argparse
import json
from pathlib import Path

from order_book.execution import PolicyNet
from order_book.provenance import write_manifest
from order_book.training import FIRST_TEST_SEED, PairedExecutionEnv, Rollouts, VALIDATION_SEEDS, compare, cpu_state_dict
from papers.decomposition import OUT
from papers.decomposition.scripts.train_rl import REGIMES, SWEEP
from scripts._common import write_json


def policy_runs(root: Path) -> list[Path]:
    """The sweep's training runs under ``root``, by their best checkpoint; leftovers from other sweeps are ignored. Refuses
    unfinished runs: a chain that ran out of hours would otherwise be scored as if it had converged."""
    runs = [root / row["name"] for row in SWEEP if (root / row["name"] / "history.json").exists()]
    unfinished = [run.name for run in runs if json.loads((run / "history.json").read_text())["status"] != "done"]
    if unfinished:
        raise SystemExit(f"training not finished for {unfinished}: add links (CHAIN=2 bash hpc/submit.sh decomposition train_rl), then submit_all.sh rl_results")
    return [run / "best_policy.pt" for run in runs]


def evaluate(policy: Path, groups: dict, workers: int, quick: bool, twap: dict) -> list[dict]:
    """One policy on every (market, size, seed) of ``groups``. ``twap`` caches TWAP's episodes across policies by episode."""
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
            learned = [r for _, r in rollouts.run([(weights, True, *job, 0) for job in jobs])]
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=None, help="best_policy.pt files; default: the sweep's runs under outputs/decomposition/rl/")
    parser.add_argument("--unseen-seeds", type=int, default=80, help=f"seeds from {FIRST_TEST_SEED}")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    root = OUT / ("rl_quick" if args.quick else "rl")
    policies = [Path(p) for p in args.policies] if args.policies else policy_runs(root)
    if not policies:
        raise SystemExit(f"no trained policies under {root}")
    groups = {"validation": VALIDATION_SEEDS, "unseen": tuple(range(FIRST_TEST_SEED, FIRST_TEST_SEED + args.unseen_seeds))}
    if args.quick:
        groups = {"validation": VALIDATION_SEEDS[:2], "unseen": (FIRST_TEST_SEED, FIRST_TEST_SEED + 1)}
    out = OUT / ("rl_eval_quick" if args.quick else "rl_eval")  # a smoke test never overwrites real results
    write_manifest(out, vars(args))
    rows, twap = [], {}
    for policy in policies:
        rows += evaluate(policy, groups, args.workers, args.quick, twap)
    write_json(rows, out / "summary.json")


if __name__ == "__main__":
    main()
