"""Named configurations: adversary levels of the research plan (section 6) and the fundamental switch.

Only the detector and whether an anticipator is present vary between levels;
the market makers' response mapping is the same everywhere.
"""

from dataclasses import replace

from order_book.config import FlowDetectorConfig, PopulationConfig, SimConfig, market_makers

FLOW_THRESHOLDS = {"lenient": 0.2, "medium": 0.3, "strict": 0.45}


def adversary(level: str, threshold: str = "medium", base: SimConfig | None = None) -> SimConfig:
    """``A0`` no detection; ``A1`` rule detector in two of four makers; ``A4`` A1 plus an order anticipator."""
    cfg = base or SimConfig()
    pop = cfg.population
    if level == "A0":
        pop = replace(pop, market_makers=market_makers(len(pop.market_makers), detecting=0), anticipators=0)
    elif level in ("A1", "A4"):
        pop = replace(pop, market_makers=market_makers(len(pop.market_makers), detecting=2),
                      anticipators=1 if level == "A4" else 0)
        cfg = replace(cfg, flow_detector=FlowDetectorConfig(threshold=FLOW_THRESHOLDS[threshold]))
    else:
        raise ValueError(f"levels A2, A3 and A5 need detectors that are not implemented yet; got {level!r}")
    return replace(cfg, population=pop)


def private_valuations(base: SimConfig | None = None, adapt: float = 0.01) -> SimConfig:
    cfg = base or SimConfig()
    return replace(cfg, fundamental=replace(cfg.fundamental, private=True, adapt=adapt))


def no_institutions(base: SimConfig | None = None) -> SimConfig:
    cfg = base or SimConfig()
    return replace(cfg, population=replace(cfg.population, institutions=0))
