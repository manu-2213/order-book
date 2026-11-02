"""Base class for single-account participants.

Agents use public information (``market.public``, the book's best quotes and
levels) plus their own account and orders; they only touch the book through
``Market``.
"""

import numpy as np

from order_book.market import Market
from order_book.rng import keyed


class Agent:
    def __init__(self, cid: int, market: Market, rng: np.random.Generator, cash: int, shares: int):
        self.cid, self.market, self.rng = cid, market, rng
        self.account = market.open_account(cid, cash, shares)

    def draws(self, purpose: int, index: int = 0):
        """The generator for this step's draws of ``purpose`` (see ``order_book.rng``)."""
        return keyed(self.rng, self.market.step, purpose, index)

    def act(self) -> None:
        raise NotImplementedError
