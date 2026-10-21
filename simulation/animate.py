"""Render the simulation as a video: price/inventory on the left, live order book on the right.

Run from the repo root: python -m simulation.animate [--stride 10] [--fps 30]
"""

import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402

from simulation.run import simulate  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def animate(hist, names, path=Path(__file__).with_name("simulation.mp4"),
            stride=10, fps=30, window=1000, half_depth=25, tick=0.01):
    """One frame per ``stride`` steps; price panels show the trailing ``window`` steps,
    the book panel shows ``half_depth`` ticks either side of the current mid."""
    n = len(hist["last"])
    x = np.arange(n)
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=(2, 1))
    ax_price, ax_inv = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
    ax_book = fig.add_subplot(grid[:, 1])

    band = [ax_price.fill_between([], [], [])]
    (last_line,) = ax_price.plot([], [], color="tab:orange", lw=0.9, label="last trade")
    (mid_line,) = ax_price.plot([], [], color="tab:blue", lw=0.7, label="mid")
    ax_price.set(ylabel="price ($)", title="Price")
    ax_price.legend(loc="upper left")
    inv_lines = [ax_inv.plot([], [], lw=1.6 if nm.startswith("MM") else 0.9, label=nm)[0] for nm in names]
    shares = hist["shares"] / hist["shares"][0] * 100  # percent of initial group holdings
    ax_inv.set(xlabel="step", ylabel="% of initial shares", title="Inventory by group",
               ylim=(0, np.nanmax(shares) * 1.3))
    ax_inv.legend(fontsize=7, ncol=5, loc="upper left")

    # Book bars live on a fixed grid of tick offsets from mid; only widths change.
    offsets = np.arange(-half_depth, half_depth + 1)
    bid_bars = ax_book.barh(offsets, np.zeros_like(offsets), height=0.85, color="tab:green", label="bids")
    ask_bars = ax_book.barh(offsets, np.zeros_like(offsets), height=0.85, color="tab:red", label="asks")
    max_vol = max((v for bids, asks in hist["books"] for _, v, _ in bids + asks), default=1)
    ax_book.set(xlim=(0, max_vol * 1.05), ylim=(-half_depth - 1, half_depth + 1), xlabel="resting lots")
    ax_book.legend(loc="lower right")
    title = fig.suptitle("")

    def update(frame):
        t = min(frame * stride, n - 1)
        lo = max(0, t - window)
        sl = slice(lo, t + 1)
        band[0].remove()
        band[0] = ax_price.fill_between(x[sl], hist["bid"][sl] * tick, hist["ask"][sl] * tick,
                                        color="tab:blue", alpha=0.2, lw=0)
        last_line.set_data(x[sl], hist["last"][sl] * tick)
        mid_line.set_data(x[sl], hist["mid"][sl] * tick)
        prices = np.concatenate([hist["bid"][sl], hist["ask"][sl], hist["last"][sl]]) * tick
        lo_p, hi_p = np.nanmin(prices), np.nanmax(prices)
        ax_price.set(xlim=(lo, max(lo + window, t)), ylim=(lo_p - 0.02, hi_p + 0.02))
        for line, j in zip(inv_lines, range(len(names))):
            line.set_data(x[sl], shares[sl, j])
        ax_inv.set_xlim(lo, max(lo + window, t))

        bids, asks = hist["books"][t]
        center = hist["mid"][t]
        if np.isnan(center):
            center = hist["last"][t] if not np.isnan(hist["last"][t]) else hist["mid"][0]
        center = int(round(center))
        for bars, levels in ((bid_bars, bids), (ask_bars, asks)):
            volume = dict((p - center, v) for p, v, _ in levels)
            for bar, off in zip(bars, offsets):
                bar.set_width(volume.get(off, 0))
        ax_book.set_yticks(offsets[::5], [f"{(center + o) * tick:.2f}" for o in offsets[::5]])
        ax_book.set_title(f"Order book (mid ${center * tick:.2f})")
        spread = hist["spread"][t]
        title.set_text(f"step {t}   last ${hist['last'][t] * tick:.2f}   "
                       f"spread {'-' if np.isnan(spread) else int(spread)} ticks")
        return []

    frames = (n - 1) // stride + 1
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400),
              progress_callback=lambda i, total: print(f"\rframe {i + 1}/{total}", end=""))
    print(f"\nsaved {path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--stride", type=int, default=10, help="simulation steps per video frame")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    _, hist, names = simulate(steps=args.steps, record_books=True)
    animate(hist, names, stride=args.stride, fps=args.fps)
