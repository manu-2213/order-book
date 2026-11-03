"""E8: decompose implementation shortfall into mechanical cost and four reaction channels.

    python -m papers.decomposition.scripts.decompose --workers 8                          # 12 baselines x 5 sizes x 20 worlds x 20 seeds
    python -m papers.decomposition.scripts.decompose --workers 8 --rl outputs/rl/policy.pt --regime low_cover   # + the PPO executor
    python -m papers.decomposition.scripts.decompose --strategies twap_market --sizes 0.05 0.2 0.4 --horizons 500 2000 8000 --workers 8
    python -m papers.decomposition.scripts.decompose --mode ladder --workers 8            # only the 5 nested arms, no Shapley
    python -m papers.decomposition.scripts.decompose --quick                              # smoke test, seconds

Every run steps Arm 0 and a treated world in lockstep on counter-based draws (order_book.counterfactual).
Prints the decomposition per (strategy, size, horizon), the per-arm impact exponents and reversion, and every
executor against the reference. Outputs: outputs/decomposition/e8/episodes.jsonl, outputs/decomposition/e8/summary.json.
"""

import argparse
from dataclasses import replace

from papers.decomposition.decomposition import (
    DecompositionConfig, compare_executors, format_concavity, format_controls, format_decomposition, format_executors,
    merge_grid, run_grid, run_shard, summarise_concavity, summarise_decomposition,
)
from order_book.experiments import CAPITAL_REGIMES, DEFAULT_REGIMES
from order_book.provenance import write_manifest
from order_book.runner import shard_dir
from papers.decomposition import OUT
from scripts._common import calibration_or_default, write_json

REGIMES = DEFAULT_REGIMES + CAPITAL_REGIMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="test seeds first-seed..first-seed+n-1")
    parser.add_argument("--first-seed", type=int, default=1000, help="1020 for seeds the design was not chosen on")
    parser.add_argument("--strategies", nargs="+", default=None, help="baseline names, pov@rate, twap@dials or rl@path")
    parser.add_argument("--rl", default=None, help="add the learned executor from these weights")
    parser.add_argument("--sizes", type=float, nargs="+", default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--regime", default="deep_cover", choices=[r.name for r in REGIMES])
    parser.add_argument("--threshold", default="medium", choices=("lenient", "medium", "strict"))
    parser.add_argument("--mode", default="shapley", choices=("shapley", "ladder"))
    parser.add_argument("--no-controls", action="store_true", help="skip the four cells outside the factorial")
    parser.add_argument("--rng", default="counter", choices=("counter", "stateful"))
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--after", type=int, default=500)
    parser.add_argument("--reference", default="twap_market", help="executor the others are compared against")
    parser.add_argument("--out", default="e8", help="output directory under outputs/")
    parser.add_argument("--resume", action="store_true", help="keep the runs already in episodes.jsonl and do the rest")
    parser.add_argument("--shard", type=int, default=None, help="run only this seed (1-based, a Slurm array task id), then stop")
    parser.add_argument("--merge", action="store_true", help="combine finished shards and write the summary")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    regime = next(r for r in REGIMES if r.name == args.regime)
    cfg = DecompositionConfig(seeds=tuple(range(args.first_seed, args.first_seed + args.seeds)), regime=regime, threshold=args.threshold,
                              mode=args.mode, controls=not args.no_controls, rng=args.rng, burn_in=args.burn_in,
                              after=args.after)
    if args.strategies:
        cfg = replace(cfg, strategies=tuple(args.strategies))
    if args.rl:
        cfg = replace(cfg, strategies=cfg.strategies + (f"rl@{args.rl}",))
    if args.sizes:
        cfg = replace(cfg, sizes=tuple(args.sizes))
    if args.horizons:
        cfg = replace(cfg, horizons=tuple(args.horizons))
    if args.quick:
        cfg = replace(cfg, strategies=("twap_market", "immediate"), sizes=(0.1, 0.4), horizons=(300,),
                      seeds=(1000, 1001), burn_in=500, after=100)
    out = OUT / (f"{args.out}_quick" if args.quick else args.out)  # a smoke test never overwrites real results
    out.mkdir(parents=True, exist_ok=True)
    specs = cfg.specs()
    print(f"[E8] {len(specs)} runs: {len(cfg.strategies)} strategies x {len(cfg.sizes)} sizes x {len(cfg.horizons)} horizons x "
          f"{len(cfg.channel_sets())} worlds x {len(cfg.seeds)} seeds\n     regime={cfg.regime.name} threshold={cfg.threshold} "
          f"rng={cfg.rng} burn_in={cfg.burn_in} after={cfg.after}")
    if args.shard is not None:
        write_manifest(shard_dir(out, args.shard), cfg, True)
        run_shard(specs, cfg.seeds, args.shard, out, calibration_or_default(), args.workers)
        return
    if args.merge:
        results = merge_grid(specs, cfg.seeds, out)
        write_manifest(out, cfg)
    else:
        write_manifest(out, cfg, args.resume)
        results = run_grid(specs, out / "episodes.jsonl", calibration_or_default(), args.workers, True, "E8", args.resume)
    summary = summarise_decomposition(results)
    print("\n" + format_decomposition(summary))
    if not args.no_controls:
        print("\n" + format_controls(summary))
    concavity = summarise_concavity(results)
    print("\n" + format_concavity(concavity))
    executors = compare_executors(results, args.reference)
    print("\n" + format_executors(executors, args.reference))
    write_json({"config": {k: v for k, v in cfg.__dict__.items()}, "summary": summary, "concavity": concavity,
                "executors": executors, "reference": args.reference}, out / "summary.json")


if __name__ == "__main__":
    main()
