"""Fit the model-based baselines' parameters under A0 on training seeds (plan section 7).

    python -m scripts.calibrate                      # 10 training seeds, full horizon (~3 min)
    python -m scripts.calibrate --quick              # 2 seeds, short horizon (seconds)

Writes outputs/calibration.json, which scripts/e2_detectability.py and scripts/e3_leakage.py load
when present (otherwise the defaults in order_book.execution.strategies.Calibration are used).
"""

import argparse

from order_book.execution.calibration import calibrate, save
from scripts import OUTPUTS

CALIBRATION_PATH = OUTPUTS / "calibration.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="training seeds 0..seeds-1")
    parser.add_argument("--horizon", type=int, default=2_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--after", type=int, default=500)
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.05, 0.1, 0.2))
    parser.add_argument("--quick", action="store_true", help="2 seeds, horizon 500: smoke test only")
    parser.add_argument("--out", default=str(CALIBRATION_PATH))
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.horizon, args.burn_in, args.after = 2, 500, 1_000, 200
    cal = calibrate(range(args.seeds), tuple(args.fractions), args.horizon, args.burn_in, args.after)
    save(cal, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
