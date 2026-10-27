"""E7: where does leakage live across market regimes? (plan section 9, hypothesis H4)

The E1-E3 gate showed that under full liquidity and institutional cover a paced schedule is
essentially undetectable. E7 pushes toward the regime where the problem bites: thinner markets and
less legitimate large-order flow, so the metaorder is a significant share of the available liquidity.
It runs the leakage counterfactual (visible vs invisible + no-executor) across a grid of regimes and
executor sizes under A1-medium and A4.

    python -m scripts.e7_regimes --workers 8                  # full grid
    python -m scripts.e7_regimes --quick                      # smoke test, seconds
    python -m scripts.e7_regimes --workers 8 --sizes 0.2 0.4 0.6 --seeds 40

Output: outputs/e7/episodes.jsonl, outputs/e7/summary.json, and a table of leakage, the order's share
of liquidity (V/step and Q), how much the executor moves the flow detector (fire, channel) and
completion, per regime x strategy x size x adversary.
"""

import argparse
from dataclasses import replace

from order_book.experiments import DEFAULT_REGIMES, E7Config, format_e7, run_specs, summarise_e7
from scripts import OUTPUTS
from scripts._common import calibration_or_default, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="test seeds 1000..1000+n-1")
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--sizes", type=float, nargs="+", default=None)
    parser.add_argument("--adversaries", nargs="+", default=None)
    parser.add_argument("--regimes", nargs="+", default=None, help="subset of regime names to run")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = E7Config(seeds=tuple(range(1000, 1000 + args.seeds)))
    if args.strategies:
        cfg = replace(cfg, strategies=tuple(args.strategies))
    if args.sizes:
        cfg = replace(cfg, sizes=tuple(args.sizes))
    if args.adversaries:
        cfg = replace(cfg, adversaries=tuple(args.adversaries))
    if args.regimes:
        chosen = {r.name: r for r in DEFAULT_REGIMES}
        cfg = replace(cfg, regimes=tuple(chosen[name] for name in args.regimes))
    if args.quick:
        cfg = replace(cfg, regimes=(DEFAULT_REGIMES[0], DEFAULT_REGIMES[-1]), strategies=("twap_market",),
                      sizes=(0.4,), adversaries=("A1-medium",), seeds=(1000, 1001),
                      horizon=400, burn_in=1_000, after=100)
    out = OUTPUTS / "e7"
    out.mkdir(parents=True, exist_ok=True)
    specs = cfg.specs()
    print(f"[E7] {len(specs)} episodes across {len(cfg.regimes)} regimes "
          f"({', '.join(r.name for r in cfg.regimes)})\n     strategies={cfg.strategies} sizes={cfg.sizes} "
          f"adversaries={cfg.adversaries} seeds={cfg.seeds[0]}..{cfg.seeds[-1]} T={cfg.horizon}")
    calibration = calibration_or_default()
    with open(out / "episodes.jsonl", "w") as sink:
        results = run_specs(specs, calibration, None, args.workers, True, "E7", sink)
    summary = summarise_e7(results)
    print("\n" + format_e7(summary))
    write_json({"config": {k: (v if k != "regimes" else [r.name for r in cfg.regimes])
                           for k, v in cfg.__dict__.items()}, "summary": summary}, out / "summary.json")


if __name__ == "__main__":
    main()
