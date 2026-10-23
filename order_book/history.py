"""Per-step recording of an analysis run, for plots and animations."""

import numpy as np

from order_book.simulator import Simulator

KEYS = ("last", "mid", "bid", "ask", "spread", "volume", "value", "attention", "flow_signal")


def record(sim: Simulator, steps: int, record_books: bool = False) -> dict:
    """Run ``steps`` and return per-step public state and per-group holdings.

    ``hist[key]`` is a length-``steps`` array for each key in ``KEYS``;
    ``hist["shares"]`` and ``hist["wealth"]`` are (steps, groups) in the order
    of ``sim.groups``; ``hist["books"]`` holds (bids, asks) level tuples per
    step when ``record_books`` (used by the animation).
    """
    groups = [[sim.market.accounts[c] for c in cids] for cids in sim.groups.values()]
    hist = {k: np.full(steps, np.nan) for k in KEYS}
    hist["shares"] = np.zeros((steps, len(groups)))
    hist["wealth"] = np.zeros((steps, len(groups)))
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
        hist["attention"][t], hist["flow_signal"][t] = ps.attention, sim.detectors["flow"].signal
        for j, accounts in enumerate(groups):
            hist["shares"][t, j] = sum(a.shares for a in accounts)
            hist["wealth"][t, j] = sum(a.cash + a.shares * ps.mid for a in accounts)
        if record_books:
            snap = sim.book.get_snapshot()
            hist["books"].append((snap["bids"], snap["asks"]))
    sim.check_conservation()
    return hist


def flow_matrix(sim: Simulator) -> np.ndarray:
    """Traded lots between groups from the analysis tape: rows buy from columns."""
    index = {cid: j for j, cids in enumerate(sim.groups.values()) for cid in cids}
    flows = np.zeros((len(sim.groups), len(sim.groups)))
    for _, _, quantity, _, buyer, seller in sim.market.tape:
        flows[index[buyer], index[seller]] += quantity
    return flows
