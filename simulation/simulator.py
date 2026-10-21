"""The background market as one picklable, seedable object."""

import numpy as np

from agents import FundamentalValue, Institution, Market, MarketMaker, NoiseFlow, Speculators, run_step
from order_book import OrderBook

# Default population at mid = 10,000 ticks ($100): cash is sized so each account
# starts with equal cash and stock value. Client ids: 0-3 makers, 4 noise,
# 5-6 institutions, 7+ speculators.
POPULATION = {"makers": 4, "detecting_makers": 2, "institutions": 2, "speculators": 40}
ENDOWMENT = {"maker": 300, "noise": 2000, "institution": 1000, "speculator": 200}  # shares


class Simulator:
    """Book + accounting + population + hidden value, driven by one shared RNG.

    ``analysis=True`` keeps the full trade tape, book trade log and hidden-value
    history; ``analysis=False`` (training) keeps only the rolling public state.
    """

    def __init__(self, seed: int = 0, num_levels: int = 20, mid: int = 10_000,
                 analysis: bool = True, population: dict | None = None, endowment: dict | None = None):
        self.rng = np.random.default_rng(seed)
        self.book = OrderBook(num_levels, mid, 1, keep_trades=analysis)
        self.market = Market(self.book, analysis)
        self.fundamental = FundamentalValue(self.rng, mid, record=analysis)
        counts = {**POPULATION, **(population or {})}
        shares = {**ENDOWMENT, **(endowment or {})}
        args = (self.market, self.rng)
        self.groups: dict[str, list[int]] = {}
        self.agents: list = []
        for i in range(counts["makers"]):
            self.agents.append(MarketMaker(i, *args, shares["maker"] * mid, shares["maker"],
                                           detect=i < counts["detecting_makers"]))
            self.groups[f"MM{i + 1}"] = [i]
        cid = counts["makers"]
        self.agents.append(NoiseFlow(cid, *args, shares["noise"] * mid, shares["noise"]))
        self.groups["Noise"] = [cid]
        cid += 1
        for i in range(counts["institutions"]):
            self.agents.append(Institution(cid, *args, shares["institution"] * mid, shares["institution"]))
            self.groups[f"Inst{i + 1}"] = [cid]
            cid += 1
        self.speculators = Speculators(cid, *args, self.fundamental, n=counts["speculators"],
                                       cash=shares["speculator"] * mid, shares=shares["speculator"])
        self.agents.append(self.speculators)
        self.groups["Speculators"] = self.speculators.cids
        self.initial_cash = sum(a.cash for a in self.market.accounts.values())
        self.initial_shares = sum(a.shares for a in self.market.accounts.values())

    @property
    def step_count(self) -> int:
        return self.market.step

    def step(self) -> None:
        run_step(self.market, self.agents, self.rng, self.fundamental)

    def run(self, steps: int) -> None:
        for _ in range(steps):
            run_step(self.market, self.agents, self.rng, self.fundamental)

    def join(self, cash: int, shares: int) -> int:
        """Open an account for an external trader (e.g. the RL agent); returns its client id."""
        cid = max(self.market.accounts) + 1
        self.market.open_account(cid, cash, shares)
        self.initial_cash += cash
        self.initial_shares += shares
        return cid

    def reseed(self, seed: int) -> None:
        """Redirect the shared generator in place so every agent follows the new seed."""
        self.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state

    def check_conservation(self) -> None:
        accounts = self.market.accounts.values()
        assert sum(a.cash for a in accounts) == self.initial_cash
        assert sum(a.shares for a in accounts) == self.initial_shares
        assert all(a.free_cash >= 0 and a.free_shares >= 0 for a in accounts)
