"""Validate the background market against the plan's section 4.2 targets.

Run from the repo root (defaults finish in well under a minute; the paper run
uses more seeds and steps):

    python -m scripts.validate                          # 5 seeds x 10k steps
    python -m scripts.validate --seeds 10 --steps 50000 --preset private
"""

import argparse
import json

from order_book.config import SimConfig
from order_book.presets import adversary, no_institutions, private_valuations
from order_book.validation import check, format_table, run_validation
from scripts import OUTPUTS

PRESETS = {"default": lambda: SimConfig(), "private": private_valuations, "A0": lambda: adversary("A0"),
           "A4": lambda: adversary("A4"), "no_institutions": no_institutions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="default")
    parser.add_argument("--no-impact", action="store_true", help="skip the paired injected-metaorder measurement")
    parser.add_argument("--out", default=None, help="JSON path (default outputs/validation_<preset>.json)")
    args = parser.parse_args()
    summary = run_validation(range(args.first_seed, args.first_seed + args.seeds), args.steps,
                             PRESETS[args.preset](), args.burn_in, paired_impact=not args.no_impact)
    rows = check(summary)
    print(format_table(rows))
    print(f"\n{summary['metaorders']} metaorders; mean shortfall by style (bps): "
          + ", ".join(f"{k} {v:.1f}" for k, v in summary["metaorder_shortfall_bps"].items()))
    if "paired_impact" in summary:
        print("paired impact by Q/V:", {k: round(v["mean_end"], 2) for k, v in summary["paired_impact"]["per_fraction"].items()})
    out = args.out or OUTPUTS / f"validation_{args.preset}.json"
    summary["checks"] = [{"check": a, "target": b, "measured": c, "passed": d} for a, b, c, d in rows]
    with open(out, "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
