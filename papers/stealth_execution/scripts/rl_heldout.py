"""Re-evaluate trained policies against TWAP on the seeds they were selected on and on seeds they have never seen.

    python -m papers.stealth_execution.scripts.rl_heldout --workers 8
    python -m papers.stealth_execution.scripts.rl_heldout --policy outputs/stealth_execution/rl/policy.pt --size 0.2 --workers 8

``train_rl`` and ``sweep_rl`` score greedy checkpoints on seeds 1000..1000+n-1 (n = 40 in the sweep) and pick the best
checkpoint and hyperparameters from those scores, so the gap they report is selected on its own test set. This runs the
same ``ExecutionEnv`` rollout, in the same stateful market, on those seeds and on the next ones. The first row reproduces
the stored numbers; the second is the unbiased estimate. Output: outputs/stealth_execution/rl_heldout/summary.json.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from order_book.execution import ExecutionEnv, make_strategy
from order_book.provenance import write_manifest
from order_book.scoring import paired_difference
from papers.stealth_execution import OUT
from scripts._common import write_json

SWEEP_BEST = (("outputs/stealth_execution/rl/sweep/lr1e-3_ent0.003_ep4_sz0.20/policy.pt", 0.20),
              ("outputs/stealth_execution/rl/sweep/lr1e-3_ent0.01_ep10_sz0.40/best_policy.pt", 0.40))


def _rollout(job):
    import torch
    torch.set_num_threads(1)
    name, size, seed = job
    _, info = ExecutionEnv(size_fraction=size).rollout(make_strategy(name), seed)
    return info.shortfall_bps, info.completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=None, help="weights to evaluate; default: the two best of the sweep")
    parser.add_argument("--size", type=float, default=0.20, help="Q/V for --policy")
    parser.add_argument("--selected-seeds", type=int, default=40, help="how many seeds from 1000 the selection used")
    parser.add_argument("--unseen-seeds", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    policies = ((args.policy, args.size),) if args.policy else SWEEP_BEST
    first_unseen = 1000 + args.selected_seeds
    groups = {"selected": range(1000, first_unseen), "unseen": range(first_unseen, first_unseen + args.unseen_seeds)}
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for policy, size in policies:
            for label, seeds in groups.items():
                twap = list(pool.map(_rollout, [("twap_market", size, s) for s in seeds]))
                learned = list(pool.map(_rollout, [(f"rl@{policy}", size, s) for s in seeds]))
                gap = paired_difference([r[0] for r in learned], [r[0] for r in twap])
                rows.append({"policy": policy, "size": size, "seeds": label, "n": len(seeds),
                             "twap": float(np.mean([r[0] for r in twap])), "ppo": float(np.mean([r[0] for r in learned])),
                             "ppo_minus_twap": gap, "completion": float(np.mean([r[1] for r in learned]))})
                print(f"Q/V={size:.2f} {label:<9} n={len(seeds):<3} TWAP {rows[-1]['twap']:+.2f}  PPO {rows[-1]['ppo']:+.2f}  "
                      f"PPO-TWAP {gap['mean']:+.2f} [{gap['low']:+.2f},{gap['high']:+.2f}] p={gap['p_two_sided']:.3f}  "
                      f"completion {rows[-1]['completion']:.3f}", flush=True)
    write_manifest(OUT / "rl_heldout", vars(args))
    write_json(rows, OUT / "rl_heldout" / "summary.json")


if __name__ == "__main__":
    main()
