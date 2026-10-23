"""One analysis run of the background market with plots.

Run from the repo root: python -m scripts.run [--steps 10000] [--seed 0]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from order_book.history import flow_matrix, record
from order_book.simulator import Simulator
from scripts import OUTPUTS


def plot(sim, hist, path=OUTPUTS / "simulation.png", tick=0.01):
    names = list(sim.groups)
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    x = np.arange(len(hist["mid"]))
    a = ax[0, 0]
    a.fill_between(x, hist["bid"] * tick, hist["ask"] * tick, alpha=0.25, label="best bid/ask")
    a.plot(x, hist["last"] * tick, lw=0.8, label="last trade")
    a.plot(x, hist["value"] * tick, lw=0.8, color="black", alpha=0.6, label="hidden value")
    a.set(title="Price", xlabel="step", ylabel="price ($)")
    a.legend()

    a = ax[0, 1]
    a.bar(x, hist["volume"], width=1.0, color="grey", label="volume / step")
    a.set(title="Traded volume, spread and attention", xlabel="step", ylabel="lots")
    b = a.twinx()
    b.plot(x, hist["spread"], color="tab:red", lw=0.7, label="spread (ticks)")
    b.plot(x, hist["attention"], color="tab:blue", lw=0.7, label="attention")
    b.legend(loc="upper right")

    per_head = np.array([len(c) for c in sim.groups.values()])
    for a, key, title in ((ax[1, 0], "wealth", "Mark-to-market wealth per head vs start ($)"),
                          (ax[1, 1], "shares", "Shares per head")):
        data = hist[key] / per_head
        if key == "wealth":
            data = (data - data[0]) * tick
        for j, name in enumerate(names):
            a.plot(x, data[:, j], lw=1.5 if name.startswith("MM") else 0.9, label=name)
        a.set(title=title, xlabel="step")
        a.legend(fontsize=7, ncol=4)

    a = ax[2, 0]
    im = a.imshow(flow_matrix(sim), cmap="viridis")
    a.set(title="Traded lots: buyer (row) <- seller (col)",
          xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names)
    a.tick_params(axis="x", labelrotation=45)
    fig.colorbar(im, ax=a)

    a = ax[2, 1]
    for side, color in (("bid", "tab:green"), ("ask", "tab:red")):
        levels = sim.book.levels(side)
        if levels:
            p, v = zip(*levels)
            a.barh(np.array(p) * tick, v, height=tick * 0.9, color=color, label=side)
    a.set(title="Final book depth", xlabel="resting lots", ylabel="price ($)")
    a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sim = Simulator(args.seed)
    hist = record(sim, args.steps)
    plot(sim, hist)
    r = np.diff(hist["mid"])
    print(f"trades={sim.book.trade_id - 1}  price ${hist['mid'][0] / 100:.2f} -> ${hist['mid'][-1] / 100:.2f}  "
          f"mean spread={np.nanmean(hist['spread']):.2f} ticks  mid-change std={r.std():.2f} ticks  "
          f"saved {OUTPUTS / 'simulation.png'}")
