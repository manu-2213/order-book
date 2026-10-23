"""Every tunable parameter of a run, as frozen dataclasses.

A run is fully determined by a ``SimConfig`` and a seed. Experiments differ
only by config; use ``dataclasses.replace`` to derive variants::

    cfg = replace(SimConfig(), population=replace(PopulationConfig(), institutions=4))

All sizes are lots, prices and distances are ticks, rates are per step.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class BookConfig:
    num_levels: int = 20          # bounded-window depth per side
    mid: int = 10_000             # initial mid in ticks ($100.00 at a $0.01 tick)
    distance_from_mid: int = 1    # initial anchor offset before any trade


@dataclass(frozen=True)
class PublicStateConfig:
    lookbacks: tuple[tuple[int, int], ...] = ((5, 30), (10, 60), (20, 120))  # (fast, slow) EMA pairs
    fast: int = 10                # EWMA span for the fast mid / variance / volume statistics
    slow: int = 100               # EWMA span for the slow statistics
    flow_window: int = 20         # steps of signed aggressive volume kept in the rolling window


@dataclass(frozen=True)
class FlowDetectorConfig:
    memory: int = 10              # EWMA span over the rolling flow imbalance
    threshold: float = 0.3        # |EWMA imbalance| below this gives no signal


@dataclass(frozen=True)
class FundamentalConfig:
    """Information arrival for speculators.

    ``private=False``: one exogenous hidden value (random walk, optional
    reversion) that every fundamentalist observes through private noise.
    ``private=True``: each speculator holds its own valuation that drifts with
    its own noise and adapts toward the traded mid, so no shared exogenous
    path drives prices.
    """

    sigma: float = 0.3            # ticks per step of the value's random walk
    reversion: float = 0.0        # pull toward the initial value per step (0 = pure random walk)
    private: bool = False         # private adaptive valuations instead of one shared value
    adapt: float = 0.01           # private mode: pull of each valuation toward the mid per step


@dataclass(frozen=True)
class NoiseFlowConfig:
    rate: float = 2.5             # Poisson orders per step before the attention multiplier
    cancel_prob: float = 0.015    # per resting order per step
    max_qty: int = 12             # order size is uniform on 1..max_qty
    mean_offset: float = 3.0      # mean ticks behind the opposite best for passive limits
    market_frac: float = 0.15     # share of orders sent as (half-size) market orders
    cross_frac: float = 0.05      # share sent as marketable limits at the opposite best
    balance_weight: float = 0.5   # tilt of the bid probability toward the pool's cash share


@dataclass(frozen=True)
class SpeculatorConfig:
    p_act: float = 0.05           # probability each member acts in a step
    bias_sd: float = 3.0          # sd of a member's permanent private bias (ticks)
    obs_noise: float = 1.0        # sd of fresh observation noise on the hidden value (ticks)
    threshold: float = 2.0        # fundamentalist: minimum |belief - mid| to trade
    aggressive_gap: float = 8.0   # fundamentalist: gap at which a market order is used
    max_qty: int = 20             # fundamentalist: maximum order size
    max_position: int | None = None  # absolute position cap; None means the share endowment
    chart_threshold: float = 6.0  # chartist: minimum |momentum| to enter (3 gives trending returns; see docs)
    chart_max_qty: int = 10       # chartist: maximum order and exit chunk size
    size_per_tick: float = 2.0    # lots per tick of signal beyond the threshold
    switch_every: int = 50        # steps between Brock-Hommes mode switches
    beta: float = 2.0             # intensity of choice in the switching logit
    fitness_memory: float = 0.1   # EWMA weight on each mode's realised block return


EXECUTION_STYLES = ("twap_market", "twap_passive", "twap_random", "pov", "front_loaded")


@dataclass(frozen=True)
class InstitutionalConfig:
    """Metaorder process of one institutional trader.

    Metaorders start as a Bernoulli-per-step (Poisson) process while idle.
    Size is log-normal, capped; participation is uniform; the duration is
    size / (participation x expected market volume per step), clamped. Each
    metaorder draws one of ``EXECUTION_STYLES``:

        twap_market   equal child market orders every ``interval`` steps
        twap_passive  each slice posted at own best, remainder crossed at the slice deadline
        twap_random   twap_market with size and interval jittered by +/- ``jitter``
        pov           tracks ``participation`` x market volume since the start
        front_loaded  Almgren-Chriss-like sinh trajectory with ``front_load`` = kappa T
    """

    start_rate: float = 0.0005    # probability per idle step of starting a metaorder
    size_median: float = 600.0    # lots; log-normal median
    size_sigma: float = 0.7       # log-normal sigma of the size
    size_max: int = 3000          # cap on the drawn size (lots)
    participation: tuple[float, float] = (0.02, 0.20)   # uniform target share of market volume
    duration: tuple[int, int] = (100, 4000)              # clamp on the implied duration (steps)
    interval: int = 5             # steps between child orders / slice length
    styles: tuple[str, ...] = EXECUTION_STYLES
    style_weights: tuple[float, ...] | None = None       # uniform over ``styles`` if None
    jitter: float = 0.5           # twap_random: relative +/- jitter on child size and interval
    front_load: float = 2.0       # front_loaded: kappa x T of the sinh schedule (larger = more front-loaded)
    volume_fallback: float = 5.0  # expected volume per step before the public EWMA has warmed up
    inventory_pull: float = 2.0   # strength of the side tilt toward the initial holding (tanh scale)


@dataclass(frozen=True)
class MarketMakerConfig:
    gamma: float = 0.003          # Avellaneda-Stoikov risk aversion
    kappa: float = 1.0            # order-arrival decay in the spread formula
    horizon: int = 50             # steps of inventory risk priced into the quotes
    min_var: float = 0.25         # floor on the fast variance (ticks^2 / step)
    max_skew: float = 10.0        # cap on both the inventory skew and the half-spread (ticks)
    size: tuple[int, int] = (5, 20)          # uniform quote size range
    refill_frac: float = 0.5      # requote a side once its size falls below this share of target
    max_inventory: int = 200      # pull the side that would push inventory past this
    detector: str | None = None   # key in ``Simulator.detectors`` to react to; None = no response
    detect_skew: float = 4.0      # reservation shift per unit of detector signal (ticks)
    detect_widen: float = 3.0     # extra half-spread on the pressured side per unit signal
    detect_size_cut: float = 0.7  # size reduction on the pressured side per unit signal
    fragility: float = 0.5        # size scale falls by fragility x (fast/slow vol ratio - 1); 0 at pull_ratio
    pull_ratio: float = 3.0       # pull both quotes when the fast/slow vol ratio exceeds this
    cooldown: int = 20            # steps the quotes stay pulled after a pull


@dataclass(frozen=True)
class OrderAnticipatorConfig:
    """Trades on a detector's signal: with the detected flow, within a position limit."""

    detector: str = "flow"
    threshold: float = 0.5        # |signal| needed to enter or add
    fade: float = 0.2             # |signal| below which the position is unwound
    rate: float = 2.0             # lots per step at full signal (4 destabilises the market; see docs)
    chunk: int = 10               # maximum lots per order
    max_position: int = 200       # absolute position cap (lots)
    contrarian_first: bool = False  # trade against the flow first, then with it (van Kervel-Menkveld)
    switch_after: int = 50        # contrarian mode: steps of confident signal before switching sides


def market_makers(n: int = 4, detecting: int = 2, detector: str = "flow",
                  base: MarketMakerConfig | None = None) -> tuple[MarketMakerConfig, ...]:
    """``n`` makers sharing ``base``; the first ``detecting`` of them react to ``detector``."""
    base = base or MarketMakerConfig()
    return tuple(replace(base, detector=detector if i < detecting else None) for i in range(n))


@dataclass(frozen=True)
class PopulationConfig:
    market_makers: tuple[MarketMakerConfig, ...] = field(default_factory=market_makers)
    institutions: int = 4
    speculators: int = 40
    anticipators: int = 0


@dataclass(frozen=True)
class EndowmentConfig:
    """Initial shares per account; cash is ``shares * mid`` so cash and stock start equal."""

    maker: int = 300
    noise: int = 6000
    institution: int = 5000
    speculator: int = 400
    anticipator: int = 1000


@dataclass(frozen=True)
class SimConfig:
    step_seconds: float = 1.0     # wall-clock meaning of one step; all rates are per step
    book: BookConfig = field(default_factory=BookConfig)
    public: PublicStateConfig = field(default_factory=PublicStateConfig)
    flow_detector: FlowDetectorConfig = field(default_factory=FlowDetectorConfig)
    fundamental: FundamentalConfig = field(default_factory=FundamentalConfig)
    noise: NoiseFlowConfig = field(default_factory=NoiseFlowConfig)
    speculators: SpeculatorConfig = field(default_factory=SpeculatorConfig)
    institution: InstitutionalConfig = field(default_factory=InstitutionalConfig)
    anticipator: OrderAnticipatorConfig = field(default_factory=OrderAnticipatorConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    endowment: EndowmentConfig = field(default_factory=EndowmentConfig)
