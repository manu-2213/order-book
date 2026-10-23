"""Zero-intelligence order flow from one pooled account."""

from order_book.agents.base import Agent
from order_book.config import NoiseFlowConfig


class NoiseFlow(Agent):
    """Poisson(rate x attention) orders per step; cost scales with events.

    Each step: cancel a Binomial(resting, cancel_prob) random subset of resting
    orders, then submit Poisson(rate * attention) new orders of 1..max_qty
    lots. A fraction ``market_frac`` are market orders (half size, so takers
    are smaller than makers as in real books), ``cross_frac`` are marketable
    limits at the opposite best, the rest rest Geometric ticks behind it
    (mean ``mean_offset`` + 1). The bid probability is tilted by
    ``balance_weight`` toward the pool's cash share of wealth, so the pool
    cannot run itself out of cash or stock in a long trend.
    """

    def __init__(self, *args, cfg: NoiseFlowConfig | None = None):
        super().__init__(*args)
        self.cfg = cfg or NoiseFlowConfig()
        self.p_geometric = 1 / (self.cfg.mean_offset + 1)

    def act(self) -> None:
        m, rng, book, cid, cfg = self.market, self.rng, self.market.book, self.cid, self.cfg
        ids = book.client_order_ids(cid)
        if ids:
            k = rng.binomial(len(ids), cfg.cancel_prob)
            if k:
                for order_id in rng.choice(list(ids), k, replace=False):
                    m.cancel(cid, int(order_id))
        n = rng.poisson(cfg.rate * m.public.attention)
        if not n:
            return
        account, mid = self.account, m.public.mid
        cash_share = account.cash / (account.cash + account.shares * mid)
        bids = rng.random(n) < 0.5 + cfg.balance_weight * (cash_share - 0.5)
        sizes = rng.integers(1, cfg.max_qty + 1, n)
        kinds = rng.random(n)
        offsets = rng.geometric(self.p_geometric, n)
        cross = cfg.market_frac + cfg.cross_frac
        for i in range(n):
            side = "bid" if bids[i] else "ask"
            quantity = int(sizes[i])
            if kinds[i] < cfg.market_frac:
                m.market(cid, side, (quantity + 1) // 2)
                continue
            bid, ask = book.best_bid, book.best_ask
            if side == "bid":
                opposite = ask if ask is not None else round(m.public.mid) + 1
                price = opposite if kinds[i] < cross else opposite - int(offsets[i])
            else:
                opposite = bid if bid is not None else round(m.public.mid) - 1
                price = opposite if kinds[i] < cross else opposite + int(offsets[i])
            m.limit(cid, side, price, quantity)
