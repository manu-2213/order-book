"""Background-market participants.

Cost scales with events, not with the number of notional traders:

    NoiseFlow            Poisson order arrivals from one pooled account
    Speculators          one array-backed population of fundamentalists and chartists
                         with Brock-Hommes strategy switching
    InstitutionalTrader  metaorders in five execution styles, labelled for analysis
    MarketMaker          Avellaneda-Stoikov quoter with fragility, optionally reading a detector
    OrderAnticipator     trades on a detector's signal

Agents only use public information plus their own accounts and orders, and
only touch the book through ``Market``.
"""

from order_book.agents.base import Agent
from order_book.agents.fundamental import FundamentalValue
from order_book.agents.anticipator import OrderAnticipator
from order_book.agents.institutional import InstitutionalTrader, MetaorderLog, MetaorderRecord
from order_book.agents.market_maker import MarketMaker
from order_book.agents.noise import NoiseFlow
from order_book.agents.speculators import Speculators

__all__ = ["Agent", "FundamentalValue", "InstitutionalTrader", "MarketMaker", "MetaorderLog", "MetaorderRecord",
           "NoiseFlow", "OrderAnticipator", "Speculators"]
