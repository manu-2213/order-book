"""Replication hooks for the decomposition: three published claims re-run in this market.

    python -m papers.decomposition.scripts.replications --workers 8                  # all three
    python -m papers.decomposition.scripts.replications --only shi_cartlidge --workers 8
    python -m papers.decomposition.scripts.replications --quick                      # smoke test, seconds

    shi_cartlidge  no reaction at 1% POV, superlinear growth of the reaction above it
    bucci          linear-to-square-root crossover in impact as detecting makers replace plain ones
    abides         quote-lifting against a predictable acquirer, compared with randomised ones

Outputs: outputs/decomposition/replications/summary.json.
"""

import argparse
import json

from papers.decomposition.decomposition import (
    abides_specs, bucci_specs, merge_grid, run_grid, run_shard, shi_cartlidge_specs, summarise_abides, summarise_bucci,
    summarise_shi_cartlidge,
)
from order_book.experiments import DEFAULT_REGIMES
from order_book.provenance import write_manifest
from order_book.runner import shard_dir
from papers.decomposition import OUT
from scripts._common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=("shi_cartlidge", "bucci", "abides"),
                        choices=("shi_cartlidge", "bucci", "abides"))
    parser.add_argument("--seeds", type=int, default=20, help="test seeds 1000..1000+n-1")
    parser.add_argument("--regime", default="deep_cover", choices=[r.name for r in DEFAULT_REGIMES])
    parser.add_argument("--resume", action="store_true", help="keep the runs already logged and do the rest")
    parser.add_argument("--shard", type=int, default=None, help="run only this seed (1-based, a Slurm array task id), then stop")
    parser.add_argument("--merge", action="store_true", help="combine finished shards and write the summary")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = tuple(range(1000, 1000 + args.seeds))
    regime = next(r for r in DEFAULT_REGIMES if r.name == args.regime)
    timing = dict(horizon=2_000, burn_in=2_000, after=500)
    if args.quick:
        seeds, timing = (1000, 1001), dict(horizon=300, burn_in=500, after=100)
    directory = OUT / ("replications_quick" if args.quick else "replications")  # a smoke test never overwrites real results
    directory.mkdir(parents=True, exist_ok=True)
    config = {"only": list(args.only), "seeds": seeds, "regime": regime.name, **timing}
    write_manifest(shard_dir(directory, args.shard) if args.shard is not None else directory, config, args.shard is not None or args.resume)

    quick = args.quick
    grids = {}
    if "shi_cartlidge" in args.only:
        grids["shi_cartlidge"] = shi_cartlidge_specs(seeds=seeds, regime=regime, **timing, **(dict(rates=(0.01, 0.1, 0.4)) if quick else {}))
    if "bucci" in args.only:
        grids["bucci"] = bucci_specs(seeds=seeds, regime=regime, **timing, **(dict(detecting=(0, 4), sizes=(0.01, 0.1, 0.4, 0.8)) if quick else {}))
    if "abides" in args.only:
        grids["abides"] = abides_specs(seeds=seeds, regime=regime, **timing, **(dict(strategies=("twap_market", "twap_random"), sizes=(0.4,)) if quick else {}))
    if args.shard is not None:
        for hook, specs in grids.items():
            run_shard(specs, seeds, args.shard, directory, None, args.workers, True, hook, f"{hook}.jsonl")
        return

    def results_of(hook):
        if args.merge:
            return merge_grid(grids[hook], seeds, directory, f"{hook}.jsonl")
        return run_grid(grids[hook], directory / f"{hook}.jsonl", None, args.workers, False, hook, args.resume)

    out = {}
    if "shi_cartlidge" in args.only:
        results = results_of("shi_cartlidge")
        out["shi_cartlidge"] = s = summarise_shi_cartlidge(results)
        print("\nShi & Cartlidge: the reaction (Arm 5 - Arm 1) by POV rate")
        for name, row in s["rows"].items():
            r = row["reaction"]
            print(f"  {name:<12} realised Q/V {row['realised_q_over_v']:.3f}  reaction {r['mean']:+.2f} [{r['low']:+.2f},{r['high']:+.2f}] bps")
        print(f"  zero at the lowest rate: {s['zero_reaction_at_lowest_rate']}; growth exponent above it: "
              f"{s['growth_exponent_above']:.2f} ({s['growth_exponent_without_largest']:.2f} without the largest rate); "
              f"superlinear: {s['superlinear']}")
    if "bucci" in args.only:
        results = results_of("bucci")
        out["bucci"] = s = summarise_bucci(results)
        print("\nBucci et al.: local impact exponent at small and large Q/V, by maker composition")
        for label, row in s.items():
            print(f"  {label:<12} delta small {row['delta_low']:+.2f}  delta large {row['delta_high']:+.2f}  crossover: {row['crossover']}")
    if "abides" in args.only:
        results = results_of("abides")
        out["abides"] = s = summarise_abides(results)
        print("\nABIDES-MARL: opposite-best lift against the acquirer (ticks)")
        for key, row in s.items():
            d = row["lift_from_detection_and_predation"]
            extra = row.get("predictable_minus_this")
            extra_txt = f"  predictable minus this {extra['mean']:+.3f} [{extra['low']:+.3f},{extra['high']:+.3f}]" if extra else ""
            print(f"  {key:<34} Arm 3 {row['lift_arm3']:+.2f}  Arm 5 {row['lift_arm5']:+.2f}  from D+A {d['mean']:+.3f} [{d['low']:+.3f},{d['high']:+.3f}]{extra_txt}")
    path = directory / "summary.json"
    if path.exists() and not args.quick:  # a partial run with --only keeps the other hooks' results
        out = {**json.loads(path.read_text()), **out}
    write_json(out, path)


if __name__ == "__main__":
    main()
