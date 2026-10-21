"""One analysis run of the background market with plots.

Run from the repo root: python -m simulation.run [--steps 10000] [--seed 0]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation.simulator import Simulator


def simulate(steps=10_000, seed=0, num_levels=20, record_books=False):
    """Run and record per-step public state and per-group holdings.

    Returns (sim, hist, names). ``hist["books"]`` holds (bids, asks) level
    tuples per step when ``record_books`` (used by the animation).
    """
    sim = Simulator(seed, num_levels)
    names = list(sim.groups)
    groups = [[sim.market.accounts[c] for c in cids] for cids in sim.groups.values()]
    keys = ("last", "mid", "bid", "ask", "spread", "volume", "value", "attention", "flow_signal")
    hist = {k: np.full(steps, np.nan) for k in keys}
    hist["shares"] = np.zeros((steps, len(names)))
    hist["wealth"] = np.zeros((steps, len(names)))
    hist["books"] = []
    ps = sim.market.public
    for t in range(steps):
        sim.step()
        hist["last"][t] = np.nan if ps.last is None else ps.last
        hist["mid"][t] = ps.mid
        hist["bid"][t] = np.nan if ps.best_bid is None else ps.best_bid
        hist["ask"][t] = np.nan if ps.best_ask is None else ps.best_ask
        hist["spread"][t] = ps.spread if ps.best_bid is not None and ps.best_ask is not None else np.nan
        hist["volume"][t], hist["value"][t] = ps.step_volume, sim.fundamental.value
        hist["attention"][t], hist["flow_signal"][t] = ps.attention, ps.flow_signal
        for j, accounts in enumerate(groups):
            hist["shares"][t, j] = sum(a.shares for a in accounts)
            hist["wealth"][t, j] = sum(a.cash + a.shares * ps.mid for a in accounts)
        if record_books:
            snap = sim.book.get_snapshot()
            hist["books"].append((snap["bids"], snap["asks"]))
    sim.check_conservation()
    return sim, hist, names


def flow_matrix(sim, names):
    """Traded lots between groups: rows buy from columns."""
    index = {cid: j for j, cids in enumerate(sim.groups.values()) for cid in cids}
    flows = np.zeros((len(names), len(names)))
    for _, _, quantity, _, buyer, seller in sim.market.tape:
        flows[index[buyer], index[seller]] += quantity
    return flows


def plot(sim, hist, names, path=Path(__file__).with_name("simulation.png"), tick=0.01):
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
    im = a.imshow(flow_matrix(sim, names), cmap="viridis")
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
    sim, hist, names = simulate(args.steps, args.seed)
    plot(sim, hist, names)
    r = np.diff(hist["mid"])
    print(f"trades={sim.book.trade_id - 1}  price ${hist['mid'][0] / 100:.2f} -> ${hist['mid'][-1] / 100:.2f}  "
          f"mean spread={np.nanmean(hist['spread']):.2f} ticks  mid-change std={r.std():.2f} ticks  "
          f"saved simulation/simulation.png")
