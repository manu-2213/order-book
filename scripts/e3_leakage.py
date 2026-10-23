"""E3: how much does leakage cost? (plan section 9, hypothesis H1; Phase 2 gate)

    python -m scripts.e3_leakage --workers 8                 # full: 12 strategies x 3 sizes x 3 adversaries
                                                             #       x {visible, invisible} x 20 seeds + no-executor runs
    python -m scripts.e3_leakage --quick                     # smoke test, seconds
    python -m scripts.e3_leakage --strategies twap_market twap_passive --sizes 0.1 0.2 --seeds 40 --paths

Every strategy runs on the same seeds, sides (by seed parity) and burn-in states under A0, A1-medium
and A4, visible and invisible to the detectors; plus one run per seed and adversary without any
executor. Leakage cost = IS(visible) - IS(invisible) paired by seed, with bootstrap CI, one-sided
p-value and Holm correction; IS(invisible) is the impact cost. Each episode prints one progress line.
Outputs: outputs/e3/episodes.jsonl, outputs/e3/summary.json, outputs/e3/impact_paths.npz (with
--paths) and the table on stdout, ending with the Phase 2 gate verdict.
"""

import argparse
from dataclasses import replace

import numpy as np

from order_book.experiments import E3Config, format_e3, impact_paths, run_specs, summarise_e3
from scripts import OUTPUTS
from scripts._common import calibration_or_default, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="test seeds 1000..1000+n-1")
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--sizes", type=float, nargs="+", default=None)
    parser.add_argument("--adversaries", nargs="+", default=None)
    parser.add_argument("--paths", action="store_true", help="record mid paths (impact paths vs the no-executor run)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = E3Config(seeds=tuple(range(1000, 1000 + args.seeds)), record_paths=args.paths)
    if args.strategies:
        cfg = replace(cfg, strategies=tuple(args.strategies))
    if args.sizes:
        cfg = replace(cfg, sizes=tuple(args.sizes))
    if args.adversaries:
        cfg = replace(cfg, adversaries=tuple(args.adversaries))
    if args.quick:
        cfg = replace(cfg, strategies=("twap_market", "immediate"), sizes=(0.1,), adversaries=("A0", "A1-medium"),
                      seeds=(1000, 1001), horizon=400, burn_in=1_000, after=100, record_paths=True)
    out = OUTPUTS / "e3"
    out.mkdir(parents=True, exist_ok=True)
    specs = cfg.specs()
    print(f"[E3] {len(specs)} episodes: strategies={cfg.strategies}\n     sizes={cfg.sizes} adversaries={cfg.adversaries} "
          f"visibilities={cfg.visibilities} seeds={cfg.seeds[0]}..{cfg.seeds[-1]} T={cfg.horizon} N={cfg.decision_interval}")
    calibration = calibration_or_default()
    with open(out / "episodes.jsonl", "w") as sink:
        results = run_specs(specs, calibration, None, args.workers, True, "E3", sink)
    summary = summarise_e3(results, cfg.gate_strategy)
    print("\n" + format_e3(summary))
    write_json({"config": {k: v for k, v in cfg.__dict__.items()}, "summary": summary}, out / "summary.json")
    if cfg.record_paths:
        paths = impact_paths(results)
        np.savez_compressed(out / "impact_paths.npz", **{k: np.asarray(v) for k, v in paths.items()})
        print(f"saved {out / 'impact_paths.npz'} ({len(paths)} paths)")


if __name__ == "__main__":
    main()
