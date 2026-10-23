"""Base class for single-account participants.

Agents use public information (``market.public``, the book's best quotes and
levels) plus their own account and orders; they only touch the book through
``Market``.
"""

import numpy as np

from order_book.market import Market


class Agent:
    def __init__(self, cid: int, market: Market, rng: np.random.Generator, cash: int, shares: int):
        self.cid, self.market, self.rng = cid, market, rng
        self.account = market.open_account(cid, cash, shares)

    def act(self) -> None:
        raise NotImplementedError
