"""The background market as one picklable, seedable object."""

import pickle

import numpy as np

from order_book.agents import (
    FundamentalValue, InstitutionalTrader, MarketMaker, MetaorderLog, NoiseFlow, OrderAnticipator, Speculators,
)
from order_book.book import OrderBook
from order_book.config import SimConfig
from order_book.detectors import Detector, FlowDetector
from order_book.market import Market


class Simulator:
    """Book + accounting + population + hidden value + detectors for one seeded run.

    Randomness: one root ``SeedSequence(seed)`` spawns an independent child
    stream for the scheduler, the hidden value and every agent, in a fixed
    order, so a run is reproduced exactly by (config, seed) and an agent's
    draws do not depend on what other agents do with theirs. ``new_stream``
    gives external participants (executors, adversaries) their own child
    without disturbing the background streams, which is what makes
    visible/invisible counterfactuals paired (common random numbers).

    Client ids: market makers first, then the noise pool, the institutional
    traders, the speculators and finally the order anticipators (last, so
    adding adversaries never changes a background agent's stream);
    ``groups`` maps a display name to the client ids it covers.
    ``detectors`` maps a name to a ``Detector`` updated every step; agents
    whose config names a detector hold a reference to it.
    ``analysis=True`` keeps the full trade tape, book trade log, hidden-value
    history and the metaorder ground-truth labels (``metaorders.records``);
    ``analysis=False`` (training) keeps only the rolling public state.
    """

    def __init__(self, seed: int = 0, config: SimConfig | None = None, analysis: bool = True):
        self.config = cfg = config or SimConfig()
        self.analysis = analysis
        self.seed = seed
        self._seed_seq = np.random.SeedSequence(seed)
        self.streams: list[np.random.Generator] = []
        self.rng = self.new_stream()  # scheduler: the order agents act in each step
        mid = cfg.book.mid
        self.book = OrderBook(cfg.book.num_levels, mid, cfg.book.distance_from_mid, keep_trades=analysis)
        self.market = Market(self.book, analysis, cfg.public)
        self.fundamental = FundamentalValue(self.new_stream(), mid, cfg.fundamental, record=analysis)
        self.detectors: dict[str, Detector] = {"flow": FlowDetector(cfg.flow_detector)}
        self.metaorders = MetaorderLog(keep=analysis)
        self.groups: dict[str, list[int]] = {}
        self.agents: list = []
        self.externals: list = []  # added later; act before the background agents, outside the permutation
        self._build_population()
        self.initial_cash = sum(a.cash for a in self.market.accounts.values())
        self.initial_shares = sum(a.shares for a in self.market.accounts.values())

    def new_stream(self) -> np.random.Generator:
        """Spawn the next child generator of the root seed; ``reseed`` covers it too."""
        stream = np.random.default_rng(self._seed_seq.spawn(1)[0])
        self.streams.append(stream)
        return stream

    def _detector(self, name: str | None) -> Detector | None:
        if name is None:
            return None
        if name not in self.detectors:
            raise KeyError(f"unknown detector {name!r}; available: {sorted(self.detectors)}")
        return self.detectors[name]

    def _build_population(self) -> None:
        cfg, mid, market = self.config, self.config.book.mid, self.market
        pop, shares = cfg.population, cfg.endowment
        cid = 0
        for i, mm_cfg in enumerate(pop.market_makers):
            self.agents.append(MarketMaker(cid, market, self.new_stream(), shares.maker * mid, shares.maker,
                                           cfg=mm_cfg, detector=self._detector(mm_cfg.detector)))
            self.groups[f"MM{i + 1}"] = [cid]
            cid += 1
        self.agents.append(NoiseFlow(cid, market, self.new_stream(), shares.noise * mid, shares.noise,
                                     cfg=cfg.noise))
        self.groups["Noise"] = [cid]
        cid += 1
        for i in range(pop.institutions):
            self.agents.append(InstitutionalTrader(cid, market, self.new_stream(), shares.institution * mid,
                                                   shares.institution, cfg=cfg.institution, log=self.metaorders))
            self.groups[f"Inst{i + 1}"] = [cid]
            cid += 1
        self.speculators = Speculators(cid, market, self.new_stream(), self.fundamental, pop.speculators,
                                       shares.speculator * mid, shares.speculator, cfg=cfg.speculators)
        self.agents.append(self.speculators)
        self.groups["Speculators"] = self.speculators.cids
        cid += pop.speculators
        for i in range(pop.anticipators):
            self.agents.append(OrderAnticipator(cid, market, self.new_stream(), shares.anticipator * mid,
                                                shares.anticipator, cfg=cfg.anticipator,
                                                detector=self._detector(cfg.anticipator.detector)))
            self.groups[f"Antic{i + 1}"] = [cid]
            cid += 1

    # ----- stepping -------------------------------------------------------- #

    @property
    def step_count(self) -> int:
        return self.market.step

    def step(self) -> None:
        """Advance one step: hidden value, external agents in fixed order, background agents in random
        order, publish state, run detectors."""
        self.fundamental.advance()
        for agent in self.externals:
            agent.act()
        agents = self.agents
        for i in self.rng.permutation(len(agents)):
            agents[i].act()
        self.market.end_step()
        public = self.market.public
        for detector in self.detectors.values():
            detector.update(public)

    def run(self, steps: int) -> None:
        step = self.step
        for _ in range(steps):
            step()

    # ----- external participants ------------------------------------------ #

    def join(self, cash: int, shares: int) -> int:
        """Open an account for an externally driven trader (e.g. the RL agent); returns its client id."""
        cid = max(self.market.accounts) + 1
        self.market.open_account(cid, cash, shares)
        self.initial_cash += cash
        self.initial_shares += shares
        return cid

    def add_external(self, agent_cls, cash: int, shares: int, **kwargs):
        """Add an agent that acts every step before the background population.

        It gets the next client id and its own stream, and is kept out of the
        scheduler's permutation, so the scheduler's and hidden value's draws
        are exactly those of the same seed without it and every background
        agent's stream is shared up to state-dependence (an agent that draws
        a state-dependent number of variates diverges only after the market
        has). Paired counterfactuals need only this and ``new_stream``.
        """
        cid = max(self.market.accounts) + 1
        agent = agent_cls(cid, self.market, self.new_stream(), cash, shares, **kwargs)
        self.initial_cash += cash
        self.initial_shares += shares
        self.externals.append(agent)
        self.groups[f"Ext{len(self.externals)}"] = [cid]
        return agent

    def reseed(self, seed: int) -> None:
        """Re-derive every stream from a new root seed, in place, so all holders follow it."""
        self.seed = seed
        self._seed_seq = np.random.SeedSequence(seed)
        for stream, child in zip(self.streams, self._seed_seq.spawn(len(self.streams))):
            stream.bit_generator.state = np.random.default_rng(child).bit_generator.state

    # ----- snapshots ------------------------------------------------------- #

    def snapshot(self) -> bytes:
        """Serialise the whole state; ``restore`` gives an independent copy that continues identically."""
        return pickle.dumps(self)

    @staticmethod
    def restore(blob: bytes) -> "Simulator":
        return pickle.loads(blob)

    # ----- invariants ------------------------------------------------------ #

    def check_conservation(self) -> None:
        accounts = self.market.accounts.values()
        assert sum(a.cash for a in accounts) == self.initial_cash
        assert sum(a.shares for a in accounts) == self.initial_shares
        assert all(a.free_cash >= 0 and a.free_shares >= 0 for a in accounts)
