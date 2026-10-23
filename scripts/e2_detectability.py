"""E2: how detectable are classic strategies? (plan section 9, hypothesis H1)

    python -m scripts.e2_detectability                       # full: 7 strategies x 3 sizes x 20 seeds
    python -m scripts.e2_detectability --quick               # smoke test, seconds
    python -m scripts.e2_detectability --workers 8 --seeds 40 --strategies twap_market pov

Market: A1-medium (two of four makers react to the flow detector). Observers: the flow detector at
three thresholds (A1), runs / CUSUM / volume-clock / regularity (A2) and the learned classifier (A3,
if trained). Each episode prints one progress line. Outputs: outputs/e2/episodes.jsonl (every
episode with its per-step detector signals), outputs/e2/summary.json and the table on stdout:
ROC AUC of |signal| (executor-active vs other steps) with bootstrap CI, signed AUC against the
executor's side, detection delay (median, p90), false alarms on institutional-only and quiet steps,
and the same detector's AUC on the institutional metaorders of the episode.
"""

import argparse
from dataclasses import replace

from order_book.experiments import E2Config, format_e2, run_specs, summarise_e2
from scripts import OUTPUTS
from scripts._common import calibration_or_default, learned_weights_or_none, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="test seeds 1000..1000+n-1")
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--sizes", type=float, nargs="+", default=None)
    parser.add_argument("--adversary", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = E2Config(seeds=tuple(range(1000, 1000 + args.seeds)))
    if args.strategies:
        cfg = replace(cfg, strategies=tuple(args.strategies))
    if args.sizes:
        cfg = replace(cfg, sizes=tuple(args.sizes))
    if args.adversary:
        cfg = replace(cfg, adversary=args.adversary)
    if args.quick:
        cfg = replace(cfg, strategies=("twap_market", "twap_passive"), sizes=(0.1,), seeds=(1000, 1001),
                      horizon=400, burn_in=1_000, after=100)
    out = OUTPUTS / "e2"
    out.mkdir(parents=True, exist_ok=True)
    specs = cfg.specs()
    print(f"[E2] {len(specs)} episodes: strategies={cfg.strategies} sizes={cfg.sizes} adversary={cfg.adversary} "
          f"observers={cfg.observers} seeds={cfg.seeds[0]}..{cfg.seeds[-1]} T={cfg.horizon} N={cfg.decision_interval}")
    calibration, weights = calibration_or_default(), learned_weights_or_none()
    with open(out / "episodes.jsonl", "w") as sink:
        results = run_specs(specs, calibration, weights, args.workers, True, "E2", sink)
    summary = summarise_e2(results, cfg.threshold)
    print("\n" + format_e2(summary))
    write_json({"config": {k: v for k, v in cfg.__dict__.items()}, "summary": summary}, out / "summary.json")


if __name__ == "__main__":
    main()
