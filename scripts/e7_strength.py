"""E7b: detector-strength sweep in a fixed liquid, low-cover regime (plan section 9, H4 / detector strength).

E7 showed that cutting institutional cover in a still-liquid market produces clean, significant
leakage (~0.3 bps), while thinning liquidity only adds variance. This sweep holds the market at that
liquid low-cover regime and turns up the detector: threshold, share of detecting makers, and response
magnitude. It finds the setting where TWAP leakage is largest and significant while completion stays
at 1 -- the candidate Phase 3 RL environment.

    python -m scripts.e7_strength --workers 8                 # 4 strengths x 2 strategies x 2 sizes x 20 seeds
    python -m scripts.e7_strength --quick                     # smoke test, seconds
    python -m scripts.e7_strength --workers 8 --level A4 --seeds 40

Output: outputs/e7b/episodes.jsonl, outputs/e7b/summary.json, and a leakage table per
regime/strength x strategy x size (same columns as E7).
"""

import argparse
from dataclasses import replace

from order_book.experiments import DETECTION_STRENGTHS, E7StrengthConfig, format_e7, run_specs, summarise_e7
from scripts import OUTPUTS
from scripts._common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="test seeds 1000..1000+n-1")
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--sizes", type=float, nargs="+", default=None)
    parser.add_argument("--strengths", nargs="+", default=None, help="subset of strength names")
    parser.add_argument("--level", choices=("A1", "A4"), default="A1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = E7StrengthConfig(seeds=tuple(range(1000, 1000 + args.seeds)), level=args.level)
    if args.strategies:
        cfg = replace(cfg, strategies=tuple(args.strategies))
    if args.sizes:
        cfg = replace(cfg, sizes=tuple(args.sizes))
    if args.strengths:
        chosen = {s.name: s for s in DETECTION_STRENGTHS}
        cfg = replace(cfg, strengths=tuple(chosen[name] for name in args.strengths))
    if args.quick:
        cfg = replace(cfg, strengths=(DETECTION_STRENGTHS[1], DETECTION_STRENGTHS[2]), strategies=("twap_market",),
                      sizes=(0.4,), seeds=(1000, 1001), horizon=400, burn_in=1_000, after=100)
    out = OUTPUTS / "e7b"
    out.mkdir(parents=True, exist_ok=True)
    specs = cfg.specs()
    print(f"[E7b] {len(specs)} episodes: regime={cfg.regime.name} (liquidity={cfg.regime.liquidity}, "
          f"institutions={cfg.regime.institutions}) level={cfg.level}\n      strengths="
          f"{[s.name for s in cfg.strengths]} strategies={cfg.strategies} sizes={cfg.sizes} "
          f"seeds={cfg.seeds[0]}..{cfg.seeds[-1]}")
    with open(out / "episodes.jsonl", "w") as sink:
        results = run_specs(specs, None, None, args.workers, True, "E7b", sink)
    summary = summarise_e7(results)
    print("\n" + format_e7(summary))
    write_json({"config": {"regime": cfg.regime.name, "level": cfg.level,
                           "strengths": [s.name for s in cfg.strengths]}, "summary": summary}, out / "summary.json")


if __name__ == "__main__":
    main()
