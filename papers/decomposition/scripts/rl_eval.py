"""The trained policies against TWAP on seeds they were selected on and on seeds nobody has seen, in the paired worlds.

Each policy runs greedily on every size and market it was trained for. The validation group reproduces the selection;
the unseen group is the estimate to quote. Differences are paired by episode, in basis points of the arrival mid: the
shortfall and the cost net of natural drift, with the executor's exposure to the makers' detector alongside.

    python -m papers.decomposition.scripts.rl_eval --workers 16
    python -m papers.decomposition.scripts.rl_eval --policies outputs/decomposition/rl/net/best_policy.pt --workers 8

Outputs: outputs/decomposition/rl_eval/summary.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from order_book.execution import PolicyNet
from order_book.provenance import write_manifest
from order_book.scoring import paired_difference
from order_book.training import FIRST_TEST_SEED, PairedExecutionEnv, Rollouts, VALIDATION_SEEDS, cpu_state_dict
from papers.decomposition import OUT
from papers.decomposition.scripts.train_rl import REGIMES
from scripts._common import write_json


def policy_runs(root: Path) -> list[Path]:
    """Every finished training run under ``root``, by its best checkpoint."""
    return sorted(p for p in root.glob("*/best_policy.pt") if (p.parent / "history.json").exists())


def evaluate(policy: Path, groups: dict, workers: int, quick: bool) -> list[dict]:
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
            twap = [r for _, r in rollouts.run([("twap_market", True, *job) for job in jobs])]
            learned = [r for _, r in rollouts.run([(weights, True, *job) for job in jobs])]
            for ci, (regime, _) in enumerate(env.configs):
                for size in env.sizes:
                    a = [r for r in learned if r.regime == regime and r.size == size]
                    b = [r for r in twap if r.regime == regime and r.size == size]
                    rows.append({"policy": str(policy), "name": policy.parent.name, "regime": regime, "size": size, "seeds": label, "n": len(a),
                                 "reward": cfg["reward"], "twap_is": float(np.mean([r.cost_is for r in b])), "ppo_is": float(np.mean([r.cost_is for r in a])),
                                 "twap_net": float(np.mean([r.cost_net for r in b])), "ppo_net": float(np.mean([r.cost_net for r in a])),
                                 "d_is": paired_difference([r.cost_is for r in a], [r.cost_is for r in b]),
                                 "d_net": paired_difference([r.cost_net for r in a], [r.cost_net for r in b]),
                                 "exposure": float(np.mean([r.exposure for r in a])), "twap_exposure": float(np.mean([r.exposure for r in b])),
                                 "completion": float(np.mean([r.completion for r in a])), "forced_share": float(np.mean([r.forced / r.quantity for r in a])),
                                 "passive_share": float(np.mean([1 - r.aggressive_share for r in a]))})
                    d = rows[-1]["d_net"]
                    print(f"{policy.parent.name:<16} {regime:<22} {size:.2f} {label:<10} n={len(a):<3} net PPO-TWAP {d['mean']:+.2f} [{d['low']:+.2f},{d['high']:+.2f}]  "
                          f"IS {rows[-1]['d_is']['mean']:+.2f}  exposure {rows[-1]['exposure']:.3f} vs {rows[-1]['twap_exposure']:.3f}", flush=True)
    finally:
        rollouts.close()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=None, help="best_policy.pt files; default: every run under outputs/decomposition/rl/")
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
    rows = []
    for policy in policies:
        rows += evaluate(policy, groups, args.workers, args.quick)
    write_json(rows, out / "summary.json")


if __name__ == "__main__":
    main()
