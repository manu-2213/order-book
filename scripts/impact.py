"""Paired square-root-impact measurement with injected metaorders (plan section 4.2, E1).

    python -m scripts.impact                      # 3 seeds x 5 sizes, finishes in seconds
    python -m scripts.impact --seeds 20 --styles twap_market twap_passive pov
"""

import argparse
import json

import numpy as np

from order_book.config import EXECUTION_STYLES
from order_book.impact import measure_impact, summarise_impact
from scripts import OUTPUTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.01, 0.02, 0.05, 0.1, 0.2))
    parser.add_argument("--styles", nargs="+", choices=EXECUTION_STYLES, default=("twap_market",))
    parser.add_argument("--horizon", type=int, default=2_000, help="execution horizon of the injected order")
    parser.add_argument("--after", type=int, default=500, help="steps after completion to track reversion")
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--side", choices=("bid", "ask"), default="bid")
    args = parser.parse_args()
    runs = measure_impact(range(args.first_seed, args.first_seed + args.seeds), tuple(args.fractions),
                          tuple(args.styles), burn_in=args.burn_in, reference_horizon=args.horizon,
                          after=args.after, side=args.side)
    s = summarise_impact(runs, args.after)
    print(f"{s['n']} injected metaorders; delta (impact at completion) = {s['delta_end']:.2f}, "
          f"delta (peak) = {s['delta_peak']:.2f}; end = {s['end_share_of_peak'] * 100:.0f}% of peak, "
          f"{s['remaining_share_of_peak'] * 100:.0f}% of peak remains {args.after} steps after; "
          f"mean shortfall {s['mean_shortfall_bps']:.1f} bps")
    for k, v in s["per_fraction"].items():
        print(f"  Q/V {k:5.3f}: n={v['n']:3d}  impact at completion {v['mean_end']:6.2f} +- {v['sd_end']:.2f} ticks"
              f"  peak {v['mean_peak']:6.2f}")
    out = OUTPUTS / "impact.json"
    paths = {f"{r.seed}_{r.style}_{r.participation:.3f}": r.path.tolist() for r in runs}
    with open(out, "w") as f:
        json.dump({**{k: v for k, v in s.items()}, "paths": paths}, f, default=float)
    np.savez_compressed(OUTPUTS / "impact_paths.npz", **{k: np.asarray(v) for k, v in paths.items()})
    print(f"saved {out} and {OUTPUTS / 'impact_paths.npz'}")


if __name__ == "__main__":
    main()
