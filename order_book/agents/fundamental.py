"""Exogenous information arrival: the hidden value speculators form beliefs about."""

import numpy as np

from order_book.config import FundamentalConfig


class FundamentalValue:
    """Hidden value in ticks: value += reversion * (anchor - value) + sigma * N(0, 1).

    Only speculators see it, and only through private noise. It drives
    beliefs, not prices; the price still comes from the order flow. With
    ``cfg.private`` the speculators keep their own adaptive valuations
    instead and this shared path is advanced but unused.
    """

    def __init__(self, rng: np.random.Generator, initial: float, cfg: FundamentalConfig | None = None,
                 record: bool = True):
        self.cfg = cfg = cfg or FundamentalConfig()
        self.rng, self.value, self.sigma, self.reversion = rng, float(initial), cfg.sigma, cfg.reversion
        self.anchor = float(initial)
        self.history: list[float] | None = [self.value] if record else None

    def advance(self) -> None:
        self.value += self.reversion * (self.anchor - self.value) + self.sigma * self.rng.standard_normal()
        if self.history is not None:
            self.history.append(self.value)
