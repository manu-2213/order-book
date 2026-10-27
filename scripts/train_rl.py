"""Phase 3: train the RL executor with PPO in the agent-based book (plan section 5).

The policy is trained in the E7b "default strength, low-cover" market -- a
liquid book with a single institution for cover and two detecting makers, the
regime where TWAP leaks measurably while the book stays stable. Training is
verbose: one line per iteration with the mean implementation shortfall,
completion, return, policy entropy, the loss terms, elapsed time and ETA, so
progress is visible across epochs. Every ``--eval-every`` iterations the
current policy is rolled out greedily on held-out test seeds and compared with
TWAP. Pass ``--wandb`` to stream every iteration and evaluation to Weights &
Biases.

    python -m scripts.train_rl --iterations 300 --episodes 32 --wandb
    python -m scripts.train_rl --quick                         # smoke test, ~1 min
    python -m scripts.train_rl --size 0.40 --institutions 1    # a different task/regime

Output: outputs/rl/policy.pt (the trained weights), outputs/rl/best_policy.pt
(the lowest-shortfall greedy checkpoint), and outputs/rl/history.json (the
per-iteration training curve). Evaluate the trained policy against the
baselines through the usual harness with strategy name "rl@outputs/rl/policy.pt",
e.g. in E3/E7.
"""

import argparse
import time

import numpy as np

from order_book.execution import (
    ExecutionEnv, PPOConfig, PPOTrainer, PolicyStrategy, make_strategy, phase3_config,
)
from scripts import OUTPUTS
from scripts._common import write_json


def evaluate(env: ExecutionEnv, strategy, seeds) -> dict:
    """Greedy roll-outs on held-out seeds: mean implementation shortfall, completion and forced share."""
    short, compl, forced = [], [], []
    for seed in seeds:
        _, info = env.rollout(strategy, int(seed))
        if info.shortfall_bps is not None:
            short.append(info.shortfall_bps)
        compl.append(info.completion)
        forced.append(info.forced / info.executed if info.executed else 0.0)
    return {
        "shortfall": float(np.mean(short)) if short else float("nan"),
        "completion": float(np.mean(compl)) if compl else float("nan"),
        "forced_share": float(np.mean(forced)) if forced else float("nan"),
        "n": len(seeds),
    }


def _init_wandb(args, run_cfg: dict, twap_eval: dict):
    if not args.wandb and args.wandb_mode == "disabled":
        return None
    import wandb
    mode = args.wandb_mode if args.wandb_mode != "disabled" else ("online" if args.wandb else "disabled")
    run_name = args.wandb_name or args.out.replace("/", "_")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        config={**run_cfg, "baseline_twap_shortfall_bps": twap_eval["shortfall"]},
        mode=mode,
        reinit="finish_previous",
    )
    return run


def run_training(args) -> dict:
    if args.quick:
        args.iterations, args.episodes, args.epochs = 5, 4, 2
        args.horizon, args.burn_in, args.eval_every, args.eval_seeds = 400, 600, 5, 4

    tag = args.wandb_name or args.out
    env = ExecutionEnv(config=phase3_config(institutions=args.institutions), size_fraction=args.size,
                       horizon=args.horizon, burn_in=args.burn_in)
    cfg = PPOConfig(iterations=args.iterations, episodes_per_update=args.episodes, epochs=args.epochs,
                    minibatch=args.minibatch, lr=args.lr, entropy_coef=args.entropy, seed=args.seed)
    run_cfg = {
        "size": args.size, "institutions": args.institutions, "horizon": args.horizon,
        "burn_in": args.burn_in, "iterations": cfg.iterations,
        "episodes_per_update": cfg.episodes_per_update, "epochs": cfg.epochs,
        "minibatch": cfg.minibatch, "lr": cfg.lr, "entropy_coef": cfg.entropy_coef, "seed": cfg.seed,
    }
    print(f"[{tag}] regime=low_cover(institutions={args.institutions}) size={args.size} horizon={args.horizon} "
          f"| PPO iterations={cfg.iterations} episodes/it={cfg.episodes_per_update} epochs={cfg.epochs} "
          f"lr={cfg.lr:g} ent={cfg.entropy_coef:g}", flush=True)

    eval_seeds = tuple(range(1000, 1000 + args.eval_seeds))
    twap = make_strategy("twap_market")
    twap_eval = evaluate(env, twap, eval_seeds)
    print(f"[{tag}] baseline TWAP on {twap_eval['n']} held-out seeds: "
          f"IS {twap_eval['shortfall']:+.2f}bps  compl {twap_eval['completion']:.3f}", flush=True)

    wb = _init_wandb(args, run_cfg, twap_eval)
    out = OUTPUTS / args.out
    out.mkdir(parents=True, exist_ok=True)

    trainer = PPOTrainer(env, cfg)
    evals: list[dict] = []
    best_eval: dict | None = None
    t0 = time.perf_counter()
    for it in range(1, cfg.iterations + 1):
        batch, stats = trainer.collect()
        losses = trainer.update(batch) if stats["transitions"] else {}
        row = {"iteration": it, **stats, **losses}
        trainer.history.append(row)
        elapsed = time.perf_counter() - t0
        eta = elapsed / it * (cfg.iterations - it)
        print(f"[{tag} {it:4d}/{cfg.iterations}] IS {stats['shortfall']:+6.2f}bps  "
              f"compl {stats['completion']:.3f}  ret {stats['return']:+6.2f}  "
              f"ent {losses.get('entropy', float('nan')):.3f}  "
              f"ploss {losses.get('policy_loss', float('nan')):+.4f}  "
              f"vloss {losses.get('value_loss', float('nan')):.3f}  "
              f"kl {losses.get('kl', float('nan')):+.4f}  "
              f"({elapsed:.0f}s, eta {eta:.0f}s)", flush=True)

        wb_payload = {
            "iteration": it,
            "train/shortfall_bps": stats["shortfall"],
            "train/completion": stats["completion"],
            "train/return": stats["return"],
            "train/transitions": stats["transitions"],
            "train/entropy": losses.get("entropy", float("nan")),
            "train/policy_loss": losses.get("policy_loss", float("nan")),
            "train/value_loss": losses.get("value_loss", float("nan")),
            "train/kl": losses.get("kl", float("nan")),
            "time/elapsed_s": elapsed,
            "time/eta_s": eta,
        }

        if it % args.eval_every == 0 or it == cfg.iterations:
            greedy = PolicyStrategy(trainer.net, deterministic=True)
            ev = evaluate(env, greedy, eval_seeds)
            ev_row = {"iteration": it, **ev, "gap_vs_twap_bps": ev["shortfall"] - twap_eval["shortfall"]}
            evals.append(ev_row)
            if best_eval is None or ev["shortfall"] < best_eval["shortfall"]:
                best_eval = ev_row
                trainer.net.save(out / "best_policy.pt")
            print(f"[{tag} eval {it:4d}] greedy IS {ev['shortfall']:+.2f}bps  "
                  f"compl {ev['completion']:.3f}  forced {ev['forced_share']:.3f}  "
                  f"vs TWAP {ev_row['gap_vs_twap_bps']:+.2f}bps "
                  f"(TWAP {twap_eval['shortfall']:+.2f}bps, best {best_eval['shortfall']:+.2f}bps@{best_eval['iteration']})",
                  flush=True)
            wb_payload.update({
                "eval/shortfall_bps": ev["shortfall"],
                "eval/completion": ev["completion"],
                "eval/forced_share": ev["forced_share"],
                "eval/gap_vs_twap_bps": ev_row["gap_vs_twap_bps"],
                "eval/twap_shortfall_bps": twap_eval["shortfall"],
                "eval/best_shortfall_bps": best_eval["shortfall"],
            })

        if wb is not None:
            wb.log(wb_payload, step=it)

    trainer.net.save(out / "policy.pt")
    print(f"saved {out / 'policy.pt'}", flush=True)
    result = {"config": run_cfg, "baseline_twap": twap_eval, "best_eval": best_eval,
              "evals": evals, "history": trainer.history}
    write_json(result, out / "history.json")
    if wb is not None:
        if best_eval is not None:
            wb.summary["best_eval_shortfall_bps"] = best_eval["shortfall"]
            wb.summary["best_eval_completion"] = best_eval["completion"]
            wb.summary["best_eval_iteration"] = best_eval["iteration"]
            wb.summary["best_gap_vs_twap_bps"] = best_eval["gap_vs_twap_bps"]
        wb.finish()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=32, help="episodes rolled out per iteration")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--size", type=float, default=0.20, help="order size as a fraction of expected volume")
    parser.add_argument("--institutions", type=int, default=1, help="institutional cover in the training market")
    parser.add_argument("--horizon", type=int, default=2_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--out", type=str, default="rl", help="output subdir under outputs/ (one per sweep run)")
    parser.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="stealth-execution")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--quick", action="store_true")
    return parser


def main():
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
