"""The per-step public view of the market that every participant may read."""

import math

from order_book.book import OrderBook
from order_book.config import PublicStateConfig


class PublicState:
    """What everyone can see, computed once per step.

    EWMA statistics at a fast and a slow horizon (mid, squared mid change,
    traded volume), momentum = fast EMA - slow EMA for a fixed set of lookback
    pairs, rolling signed aggressive flow over ``flow_window`` steps with its
    imbalance in [-1, 1], the step's anonymous aggressive trades as
    ``step_trades`` [(side, price, quantity)], and an attention multiplier that
    rises with relative volatility and volume (activity clustering). Anything
    that *interprets* these statistics (e.g. a metaorder detector) lives in
    ``order_book.detectors``.
    """

    def __init__(self, mid: int, cfg: PublicStateConfig | None = None):
        cfg = cfg or PublicStateConfig()
        self.mid = float(mid)
        self.last: int | None = None
        self.best_bid = self.best_ask = None
        self.spread = 0
        self.ma_fast = self.ma_slow = float(mid)
        self.var_fast = self.var_slow = 1.0  # ticks^2 per step
        self.volume_fast = self.volume_slow = 0.0
        self.lookbacks = cfg.lookbacks
        self.flow_window = cfg.flow_window
        self.momentum = [0.0] * len(cfg.lookbacks)
        self.imbalance = 0.0
        self.attention = 1.0
        self.warmed_up = False  # slow EWMAs have seen a full span
        self.step_volume = 0
        self.step_trades: list[tuple] = []
        self.step_count = 0
        self._a_fast, self._a_slow = 2 / (cfg.fast + 1), 2 / (cfg.slow + 1)
        self._alphas = [(2 / (f + 1), 2 / (s + 1)) for f, s in cfg.lookbacks]
        self._emas = [[float(mid), float(mid)] for _ in cfg.lookbacks]
        self._buy = [0] * cfg.flow_window
        self._sell = [0] * cfg.flow_window
        self.buy_flow = self.sell_flow = 0
        self._n = 0

    def update(self, book: OrderBook, buy_volume: int, sell_volume: int, last: int | None,
               flow_buy: int | None = None, flow_sell: int | None = None,
               step_trades: list[tuple] | None = None) -> None:
        """``flow_buy``/``flow_sell`` feed the signed flow window (default: all aggressive volume)."""
        if flow_buy is None:
            flow_buy, flow_sell = buy_volume, sell_volume
        self.step_trades = step_trades if step_trades is not None else []
        bid, ask = book.best_bid, book.best_ask
        self.best_bid, self.best_ask = bid, ask
        if bid is not None and ask is not None:
            mid, self.spread = (bid + ask) / 2, ask - bid
        else:
            mid = float(last) if last is not None else self.mid
        change, self.mid, self.last = mid - self.mid, mid, last
        a, b = self._a_fast, self._a_slow
        self.ma_fast += a * (mid - self.ma_fast)
        self.ma_slow += b * (mid - self.ma_slow)
        self.var_fast += a * (change * change - self.var_fast)
        self.var_slow += b * (change * change - self.var_slow)
        volume = self.step_volume = buy_volume + sell_volume
        self.volume_fast += a * (volume - self.volume_fast)
        self.volume_slow += b * (volume - self.volume_slow)
        for k, (af, asl) in enumerate(self._alphas):
            ema = self._emas[k]
            ema[0] += af * (mid - ema[0])
            ema[1] += asl * (mid - ema[1])
            self.momentum[k] = ema[0] - ema[1]
        i = self._n % len(self._buy)
        self.buy_flow += flow_buy - self._buy[i]
        self.sell_flow += flow_sell - self._sell[i]
        self._buy[i], self._sell[i] = flow_buy, flow_sell
        total = self.buy_flow + self.sell_flow
        self.imbalance = (self.buy_flow - self.sell_flow) / total if total else 0.0
        self._n += 1
        self.step_count = self._n
        if self._n * self._a_slow > 1:  # past the slow warm-up; before it the ratios are meaningless
            self.warmed_up = True
            relative_vol = math.sqrt(self.var_fast / self.var_slow) if self.var_slow > 0 else 1.0
            relative_volume = self.volume_fast / self.volume_slow if self.volume_slow > 0 else 1.0
            self.attention = min(3.0, max(0.5, 1.0 + 0.5 * (relative_vol - 1) + 0.5 * (relative_volume - 1)))
