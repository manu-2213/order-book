"""Named configurations: adversary levels of the research plan (section 6) and the fundamental switch.

Only the detector and whether an anticipator is present vary between levels;
the market makers' response mapping is the same everywhere.
"""

from dataclasses import replace

from order_book.config import FlowDetectorConfig, MarketMakerConfig, SimConfig

FLOW_THRESHOLDS = {"lenient": 0.2, "medium": 0.3, "strict": 0.45}


def adversary(level: str, threshold: str = "medium", base: SimConfig | None = None,
              detecting: int | None = None) -> SimConfig:
    """``A0`` no detection; ``A1`` rule detector in some makers; ``A4`` A1 plus an order anticipator.

    Only the detection wiring changes: the makers' quote size, fragility and the rest of ``base`` are
    preserved (so a thin-market ``regime`` survives), and the number of detecting makers defaults to
    ``min(2, n)`` but can be overridden with ``detecting``.
    """
    cfg = base or SimConfig()
    pop = cfg.population
    n = len(pop.market_makers)
    wired = lambda d: tuple(replace(m, detector="flow" if i < d else None) for i, m in enumerate(pop.market_makers))
    if level == "A0":
        pop = replace(pop, market_makers=wired(0), anticipators=0)
    elif level in ("A1", "A4"):
        d = min(2, n) if detecting is None else min(detecting, n)
        pop = replace(pop, market_makers=wired(d), anticipators=1 if level == "A4" else 0)
        cfg = replace(cfg, flow_detector=FlowDetectorConfig(threshold=FLOW_THRESHOLDS[threshold]))
    else:
        raise ValueError(f"levels A2, A3 and A5 need detectors that are not implemented yet; got {level!r}")
    return replace(cfg, population=pop)


def regime(liquidity: float = 1.0, institutions: int = 4, makers: int = 4, mm_size_scale: float = 1.0,
           base: SimConfig | None = None, capital: int = 1) -> SimConfig:
    """A market of a given liquidity and institutional cover, before detection is wired in (plan E7).

    ``liquidity`` scales the noise order rate, the speculator count and the market-maker quote size
    together (``< 1`` = thinner market, so a fixed-participation order is a larger share of the standing
    book and each child moves the price more); ``institutions`` sets how much legitimate large-order
    flow hides the executor (``0`` = no cover, every detected one-sided burst is the executor);
    ``makers`` the number of market makers; ``mm_size_scale`` thins the book independently of the rest;
    ``capital`` scales what speculators and the noise pool hold, which a long metaorder can otherwise exhaust.
    Compose with ``adversary``: ``adversary("A4", base=regime(liquidity=0.5, institutions=1))``.
    """
    cfg = base or SimConfig()
    noise = replace(cfg.noise, rate=cfg.noise.rate * liquidity)
    speculators = max(1, round(cfg.population.speculators * liquidity))
    templates = cfg.population.market_makers or (MarketMakerConfig(),)
    scale = liquidity * mm_size_scale
    scaled = lambda m: replace(m, size=(max(1, round(m.size[0] * scale)), max(2, round(m.size[1] * scale))), detector=None)
    pop = replace(cfg.population, institutions=institutions, speculators=speculators,
                  market_makers=tuple(scaled(templates[i % len(templates)]) for i in range(makers)))
    endowment = replace(cfg.endowment, speculator=cfg.endowment.speculator * capital, noise=cfg.endowment.noise * capital)
    return replace(cfg, noise=noise, population=pop, endowment=endowment)


def private_valuations(base: SimConfig | None = None, adapt: float = 0.01) -> SimConfig:
    cfg = base or SimConfig()
    return replace(cfg, fundamental=replace(cfg.fundamental, private=True, adapt=adapt))


def no_institutions(base: SimConfig | None = None) -> SimConfig:
    cfg = base or SimConfig()
    return replace(cfg, population=replace(cfg.population, institutions=0))
