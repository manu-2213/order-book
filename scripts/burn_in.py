"""Burn in several seeds and save the RL start-state pool.

Run from the repo root: python -m scripts.burn_in [--seeds 8] [--burn-in 5000]
"""

import argparse
import pickle
from pathlib import Path

from order_book.env import STATES_PATH, burn_in


def main():
    parser = argparse.ArgumentParser(description="Burn in seeds and save the state pool.")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--burn-in", type=int, default=5_000)
    parser.add_argument("--out", type=Path, default=STATES_PATH)
    args = parser.parse_args()
    states = burn_in(range(args.seeds), args.burn_in)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(states, f)
    print(f"saved {len(states)} states ({sum(map(len, states)) / 1e6:.1f} MB) to {args.out}")


if __name__ == "__main__":
    main()
