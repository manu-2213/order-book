"""Independent seeded runs in parallel processes: python -m simulation.batch."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation.simulator import Simulator


def _mid_path(args):
    seed, steps, num_levels = args
    sim = Simulator(seed, num_levels, analysis=False)
    path = np.empty(steps)
    for t in range(steps):
        sim.step()
        path[t] = sim.market.public.mid
    return path


def simulate_paths(runs=100, steps=10_000, num_levels=20, seed=0, workers=None):
    """Mid-price paths with shape (runs, steps) for seeds seed..seed+runs-1."""
    if min(runs, steps, num_levels) < 1 or seed < 0:
        raise ValueError("runs, steps and num_levels must be positive; seed must be nonnegative.")
    seeds = np.arange(seed, seed + runs)
    jobs = [(int(s), steps, num_levels) for s in seeds]
    workers = min(runs, workers or os.cpu_count() or 1)
    if workers == 1:
        paths = [_mid_path(job) for job in jobs]
    else:
        with ProcessPoolExecutor(workers) as pool:
            paths = list(pool.map(_mid_path, jobs))
    return seeds, np.array(paths)


def plot_paths(paths, path, tick=0.01):
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(paths.shape[1])
    for prices in paths:
        ax.plot(x, prices * tick, lw=0.6, alpha=0.35)
    low, median, high = np.percentile(paths * tick, [5, 50, 95], axis=0)
    ax.fill_between(x, low, high, color="black", alpha=0.1, label="5th-95th percentile")
    ax.plot(x, median, color="black", lw=1.8, label="median")
    ax.set(title=f"{len(paths)} independent sampled mid-price paths", xlabel="step", ylabel="mid ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--levels", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    args = parser.parse_args()
    seeds, paths = simulate_paths(args.runs, args.steps, args.levels, args.seed, args.workers)
    out = Path(__file__).parent
    np.savez_compressed(out / "price_paths.npz", seeds=seeds, mid=paths, tick_size=0.01)
    plot_paths(paths, out / "price_paths.png")
    print(f"saved {out / 'price_paths.png'} and {out / 'price_paths.npz'}")
