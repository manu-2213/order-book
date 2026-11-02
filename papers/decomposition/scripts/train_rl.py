"""Train the learned executor in the paired worlds, where the decomposition will score it.

The reward is the cost of each fill against the reference world's mid at that step (cost net of natural drift), or the
plain shortfall with ``--reward arrival``; ``--detection-penalty`` charges time spent under the makers' detector and
``--impact-penalty`` weights the displacement of the price. Episodes draw a seed below 800 and a size and market from
the lists given; the checkpoint is chosen on seeds 800-839 against TWAP on the same episodes; seeds from 1000 up are
never used here.

    python -m papers.decomposition.scripts.train_rl --name net --workers 8
    python -m papers.decomposition.scripts.train_rl --sweep 3 --workers 16 --resume --hours 0.9    # one row of SWEEP, as the cluster runs it
    python -m papers.decomposition.scripts.train_rl --quick

A run stopped by ``--hours`` leaves outputs/decomposition/rl/<name>/checkpoint.pt and resumes with ``--resume``; the chained
cluster jobs do exactly this. Outputs: policy.pt (final), best_policy.pt (lowest validation gap), history.json, manifest.json.
"""

import argparse
import json
import sys
from dataclasses import replace

from order_book.execution.ppo import PPOConfig
from order_book.experiments import CAPITAL_REGIMES
from order_book.provenance import write_manifest
from order_book.training import PairedExecutionEnv, PairedTrainer, RewardSpec, TRAIN_SEEDS, TrainingConfig, config_summary
from papers.decomposition import OUT
from scripts._common import write_json

# The cluster's sweep: one row per array task. Everything not named here is the default below.
SWEEP = (
    dict(name="net"),                                                   # the main agent: cost net of drift
    dict(name="net_detect0.01", detection_penalty=0.01),                # and increasingly, staying under the makers' detector
    dict(name="net_detect0.03", detection_penalty=0.03),
    dict(name="net_detect0.1", detection_penalty=0.1),
    dict(name="is", reward="arrival"),                                  # the ablation: the plain shortfall, drift and all
    dict(name="net_impact1", impact_penalty=1.0),                       # the price's displacement weighted twice
    dict(name="net_mixed", regimes=("low_cover_capital10", "deep_cover_capital10")),  # trained across both markets
    dict(name="net_seed1", seed=1),                                     # another initialisation and episode draw
)
REGIMES = {r.name: r for r in CAPITAL_REGIMES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="net", help="run name under outputs/decomposition/rl/")
    parser.add_argument("--sweep", type=int, default=None, help="take the settings of this row of SWEEP (1-based, a Slurm array task id)")
    parser.add_argument("--regimes", nargs="+", default=("low_cover_capital10",), choices=sorted(REGIMES))
    parser.add_argument("--sizes", type=float, nargs="+", default=(0.1, 0.2, 0.4))
    parser.add_argument("--reward", default="arm0", choices=("arm0", "arrival"))
    parser.add_argument("--detection-penalty", type=float, default=0.0)
    parser.add_argument("--impact-penalty", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=2_000)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--episodes", type=int, default=64, help="episodes rolled out per iteration")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hours", type=float, default=None, help="stop with a checkpoint before this much wall time")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def settings(args) -> argparse.Namespace:
    if args.sweep is not None:
        row = dict(SWEEP[args.sweep - 1])
        args.name = row.pop("name")
        for key, value in row.items():
            setattr(args, key, value)
    return args


def main(argv=None):
    args = settings(build_parser().parse_args(argv))
    if args.quick:
        args.iterations, args.episodes, args.epochs, args.horizon, args.eval_every = min(args.iterations, 3), 4, 1, 200, 2
    env = PairedExecutionEnv(configs=tuple((name, REGIMES[name].base()) for name in args.regimes), sizes=tuple(args.sizes),
                             reward=RewardSpec(args.reward, args.detection_penalty, args.impact_penalty), horizon=args.horizon,
                             burn_in=400 if args.quick else 2_000)
    ppo = PPOConfig(iterations=args.iterations, episodes_per_update=args.episodes, epochs=args.epochs, minibatch=args.minibatch, lr=args.lr,
                    entropy_coef=args.entropy, seed=args.seed, train_seeds=TRAIN_SEEDS)
    training = TrainingConfig(ppo=ppo, eval_every=args.eval_every)
    if args.quick:
        training = replace(training, validation_seeds=(800, 801))
    out = OUT / ("rl_quick" if args.quick else "rl") / args.name  # a smoke test never overwrites real results
    out.mkdir(parents=True, exist_ok=True)
    summary = config_summary(env, training)
    trainer = PairedTrainer(env, training, workers=args.workers, out=out)
    resumed = args.resume and trainer.resume()
    if resumed and trainer.iteration >= ppo.iterations:
        print(f"[{args.name}] already done at iteration {trainer.iteration}")
        return
    write_manifest(out, summary, resume=resumed)
    print(f"[{args.name}] regimes={args.regimes} sizes={args.sizes} reward={env.reward} | iterations={ppo.iterations} episodes/it={ppo.episodes_per_update} "
          f"workers={args.workers}" + (f" | resumed at {trainer.iteration}" if resumed else ""), flush=True)
    try:
        status = trainer.train(hours=args.hours)
    finally:
        trainer.close()
    write_json({"name": args.name, "status": status, "iteration": trainer.iteration, "config": summary, "best": trainer.best,
                "evals": trainer.evals, "history": trainer.history}, out / "history.json")
    print(f"[{args.name}] {status} at iteration {trainer.iteration}" + (f"; best validation gap {trainer.best['gap_net']:+.2f} bps at {trainer.best['iteration']}"
                                                                        if trainer.best else ""), flush=True)


if __name__ == "__main__":
    main()
