"""Agent-based limit order book market.

    order_book.book          OrderBook: the matching engine, independent of agents
    order_book.market        Account, Market: accounting and order routing
    order_book.public_state  PublicState: what every participant may read
    order_book.config        frozen dataclasses for every parameter
    order_book.detectors     Detector interface and the rule-based FlowDetector
    order_book.agents        background participants and the metaorder labels
    order_book.simulator     Simulator: one seeded, picklable run with per-agent RNG streams
    order_book.presets       adversary levels A0/A1/A4 and other named configurations
    order_book.history       per-step recording for plots
    order_book.validation    stylised facts and impact statistics against the plan's targets
    order_book.impact        paired impact measurement with injected metaorders
    order_book.execution     execution task, executor agent, baseline strategies, calibration
    order_book.scoring       detector scoring (AUC, delay, false alarms) and the statistical protocol
    order_book.experiments   adversary levels, episode specs, market regimes and the episode runner
    order_book.runner        the process pool and the resumable result log
    order_book.provenance    SIMULATOR_VERSION and the manifest written next to every result
    order_book.rng           counter-based draws addressed by (seed, agent, step, purpose)
    order_book.ghost         a live copy of a book replaying its order flow (the ghost arm)
    order_book.counterfactual  Arm 0 and a treated world in lockstep, with four reaction channels
    order_book.env           Gymnasium wrapper (imports gymnasium; not re-exported here)

The studies built on this library live in papers/, one folder per paper (papers/README.md).
Full documentation with every equation and default: docs/simulator.tex (docs/simulator.pdf).
"""

from order_book.book import OrderBook, OrderView, Report, Side, Trade
from order_book.config import SimConfig
from order_book.market import Account, Market
from order_book.public_state import PublicState
from order_book.simulator import Simulator

__all__ = [
    "Account", "Market", "OrderBook", "OrderView", "PublicState", "Report", "Side", "SimConfig",
    "Simulator", "Trade",
]
