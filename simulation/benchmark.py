"""Steps per second of the background market, plus a cProfile summary.

Run from the repo root: python -m simulation.benchmark [--steps 20000]
"""

import argparse
import cProfile
import pstats
import time

from simulation.simulator import Simulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--profile-steps", type=int, default=5_000)
    parser.add_argument("--analysis", action="store_true", help="keep the full tape (default: training mode)")
    args = parser.parse_args()

    sim = Simulator(seed=0, analysis=args.analysis)
    start = time.perf_counter()
    sim.run(args.steps)
    elapsed = time.perf_counter() - start
    sim.check_conservation()
    print(f"{args.steps} steps in {elapsed:.2f}s -> {args.steps / elapsed:,.0f} steps/s   "
          f"(last ${sim.market.public.mid / 100:.2f}, {sim.book.trade_id - 1} trades, "
          f"{len(sim.book._orders)} resting)")

    profiler = cProfile.Profile()
    profiler.enable()
    Simulator(seed=1, analysis=args.analysis).run(args.profile_steps)
    profiler.disable()
    pstats.Stats(profiler).sort_stats("tottime").print_stats(20)


if __name__ == "__main__":
    main()
