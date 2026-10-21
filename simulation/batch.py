"""Independent seeded simulations: python -m simulation.batch."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation.run import simulate


def _price_path(settings):
    seed, steps, num_levels = settings
    _, hist, _, _ = simulate(steps=steps, seed=seed, num_levels=num_levels,
                              record_books=False)
    return hist["last"]


def simulate_paths(runs=100, steps=10_000, num_levels=20, seed=0, workers=None):
    """Return seeds and last-trade prices, with shape (runs, steps).

    Prices before the first trade remain NaN. Independent seeds vary both the
    initial agent endowments/activity and their subsequent actions.
    """
    if min(runs, steps, num_levels) < 1 or seed < 0:
        raise ValueError("runs, steps and num_levels must be positive; seed must be nonnegative.")
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive.")
    workers = min(runs, workers if workers is not None else min(4, os.cpu_count() or 1))
    seeds = np.arange(seed, seed + runs)
    settings = [(int(s), steps, num_levels) for s in seeds]
    paths = np.empty((runs, steps))
    if workers == 1:
        for i, setting in enumerate(settings):
            paths[i] = _price_path(setting)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, path in enumerate(pool.map(_price_path, settings)):
                paths[i] = path
                print(f"\rCompleted {i + 1}/{runs} simulations", end="", flush=True)
        print()
    return seeds, paths


def plot_paths(paths, path, tick=0.01):
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(paths.shape[1])
    for prices in paths:
        ax.plot(x, prices * tick, lw=0.6, alpha=0.35)
    observed = np.any(np.isfinite(paths), axis=0)
    if np.any(observed):
        low, median, high = np.nanpercentile(paths[:, observed] * tick, [5, 50, 95], axis=0)
        ax.fill_between(x[observed], low, high, color="black", alpha=0.1,
                        label="Empirical 5th-95th percentiles")
        ax.plot(x[observed], median, color="black", lw=1.8, label="Median")
        ax.legend()
    ax.set(title=f"{len(paths)} independent sampled price paths",
           xlabel="Timestep", ylabel="Last-trade price ($)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--levels", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="first seed; runs use consecutive seeds")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default: up to 4)")
    args = parser.parse_args()
    seeds, paths = simulate_paths(args.runs, args.steps, args.levels, args.seed, args.workers)
    output = Path(__file__).parent
    np.savez_compressed(output / "price_paths.npz", seeds=seeds, prices=paths,
                        num_levels=args.levels, tick_size=0.01)
    fig = plot_paths(paths, output / "price_paths.png")
    plt.close(fig)
    print(f"Saved {output / 'price_paths.png'} and {output / 'price_paths.npz'}")


if __name__ == "__main__":
    main()
