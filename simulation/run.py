"""Run the random-agent market simulation and plot the results.

Run from the repo root: python -m simulation.run
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agents import Market, MarketMaker, RandomAgent
from order_book import OrderBook


def simulate(steps=10_000, n_random=6, seed=0, mid=10_000, num_levels=20,
             record_books=True):
    rng = np.random.default_rng(seed)
    book = OrderBook(num_levels=num_levels, mid_price=mid, distance_from_mid=1)
    market = Market(book)
    mm = MarketMaker(0, market, np.random.default_rng(rng.integers(1 << 32)), 2_000_000, 200)
    traders = [RandomAgent(i, market, np.random.default_rng(rng.integers(1 << 32)),
                           int(rng.integers(200_000, 1_000_000)), int(rng.integers(20, 100)),
                           activity=float(rng.uniform(0.3, 0.8)))
               for i in range(1, n_random + 1)]
    agents = [mm, *traders]
    totals = (sum(a.cash for a in market.accounts.values()),
              sum(a.shares for a in market.accounts.values()))

    keys = ("last", "mid", "bid", "ask", "spread", "volume", "depth_bid", "depth_ask")
    hist = {k: [] for k in keys}
    hist["cash"], hist["shares"], hist["wealth"] = [], [], []
    books = []  # per-step L2 levels [(price, volume, n_orders), ...] for animation
    n_trades = 0
    for step in range(steps):
        market.step = step
        for i in rng.permutation(len(agents)):
            agents[i].act()
        trades = book._trades[n_trades:]
        n_trades = len(book._trades)
        snap = book.get_snapshot()
        if record_books:
            books.append((snap["bids"], snap["asks"]))
        mark = market.last_price or mid
        nan = lambda x: np.nan if x is None else x  # noqa: E731
        for k, v in zip(keys, (market.last_price, snap["mid_price"], snap["best_bid"], snap["best_ask"],
                               snap["spread"], sum(t["quantity"] for t in trades),
                               snap["total_bid_volume"], snap["total_ask_volume"])):
            hist[k].append(nan(v))
        hist["cash"].append([market.accounts[a.cid].cash for a in agents])
        hist["shares"].append([market.accounts[a.cid].shares for a in agents])
        hist["wealth"].append([market.accounts[a.cid].cash + market.accounts[a.cid].shares * mark
                               for a in agents])
        assert (sum(a.cash for a in market.accounts.values()),
                sum(a.shares for a in market.accounts.values())) == totals
        assert all(a.cash >= 0 and a.shares >= 0 for a in market.accounts.values())

    hist = {k: np.asarray(v, dtype=float) for k, v in hist.items()}
    hist["books"] = books
    names = ["MM"] + [f"R{a.cid}" for a in traders]
    flows = np.zeros((len(agents), len(agents)))
    for t in book._trades:
        flows[t["buyer_id"], t["seller_id"]] += t["quantity"]
    return book, hist, names, flows


def plot(book, hist, names, flows, path=Path(__file__).with_name("simulation.png"), tick=0.01):
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    x = np.arange(len(hist["last"]))
    a = ax[0, 0]
    a.fill_between(x, hist["bid"] * tick, hist["ask"] * tick, alpha=0.25, label="best bid/ask")
    a.plot(x, hist["mid"] * tick, lw=0.8, label="mid")
    a.plot(x, hist["last"] * tick, lw=0.8, label="last trade")
    a.set(title="Price", xlabel="step", ylabel="price ($)")
    a.legend()

    a = ax[0, 1]
    a.bar(x, hist["volume"], width=1.0, color="grey", label="volume / step")
    a.set(title="Traded volume and spread", xlabel="step", ylabel="lots")
    b = a.twinx()
    b.plot(x, hist["spread"], color="tab:red", lw=0.7, label="spread (ticks)")
    b.set_ylabel("spread (ticks)")

    for a, key, title in ((ax[1, 0], "wealth", "Mark-to-market wealth (vs start)"),
                          (ax[1, 1], "shares", "Inventory (shares)")):
        data = hist[key] - hist[key][0] if key == "wealth" else hist[key]
        if key == "wealth":
            data = data * tick
        for j, name in enumerate(names):
            a.plot(x, data[:, j], lw=1.6 if name == "MM" else 0.9, label=name)
        a.set(title=title, xlabel="step")
        a.legend(fontsize=8, ncol=2)

    a = ax[2, 0]
    im = a.imshow(flows, cmap="viridis")
    a.set(title="Traded lots: buyer (row) <- seller (col)",
          xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names)
    fig.colorbar(im, ax=a)

    a = ax[2, 1]
    snap = book.get_snapshot()
    for side, color in (("bids", "tab:green"), ("asks", "tab:red")):
        levels = snap[side]
        if levels:
            p, v, _ = zip(*levels)
            a.barh(np.array(p) * tick, v, height=tick * 0.9, color=color, label=side)
    a.set(title="Final book depth", xlabel="resting lots", ylabel="price ($)")
    a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


if __name__ == "__main__":
    book, hist, names, flows = simulate()
    plot(book, hist, names, flows)
    last = hist["last"][~np.isnan(hist["last"])]
    print(f"trades={len(book._trades)}  price ${last[0] / 100:.2f} -> ${last[-1] / 100:.2f}  "
          f"mean spread={np.nanmean(hist['spread']):.1f} ticks  saved simulation/simulation.png")
