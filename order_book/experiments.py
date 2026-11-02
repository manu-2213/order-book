"""The experiment harness every paper builds on: one execution episode in one market.

    AdversarySpec   which market the executor trades in (A0, A1 at three thresholds, A4)
    OBSERVERS       detectors attached as observers (they see the market, nobody reacts to them)
    EpisodeSpec     one execution episode: seed, strategy, size, adversary, visibility, task
    RegimeSpec      one market regime: how thin it is and how much cover it offers
    run_specs       runs episodes in parallel, one progress line and one JSONL record each

Seeds: 0-999 are training seeds (calibration, detector training, tuning); test
seeds start at 1000. The executor's side is fixed by the seed's parity so
that every strategy and adversary level is paired on the same seeds, sides
and burn-in states. Results are JSON-serialisable records. The studies built
on this live in ``papers/``.
"""

import json
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from order_book.config import FlowDetectorConfig, SimConfig
from order_book.detectors import FlowDetector, LearnedDetector, held_out_detectors
from order_book.execution import Calibration, ExecutionTask, ExecutorAgent, expected_volume_from_tape, make_strategy
from order_book.presets import FLOW_THRESHOLDS, adversary
from order_book.runner import iter_jobs
from order_book.simulator import Simulator

TRAIN_SEEDS = range(0, 1000)
TEST_SEEDS = range(1000, 1020)
SIZES = (0.05, 0.10, 0.20)
EXECUTOR_ENDOWMENT = 50_000          # shares; cash = shares x mid, so any task size is funded


# ----- adversaries ------------------------------------------------------------------------ #

@dataclass(frozen=True)
class AdversarySpec:
    name: str
    level: str                      # A0 | A1 | A4
    threshold: str = "medium"       # flow-detector threshold the makers (and anticipator) react to

    def config(self, base: SimConfig | None = None) -> SimConfig:
        return adversary(self.level, self.threshold, base)


ADVERSARIES = {
    "A0": AdversarySpec("A0", "A0"),
    "A1-lenient": AdversarySpec("A1-lenient", "A1", "lenient"),
    "A1-medium": AdversarySpec("A1-medium", "A1", "medium"),
    "A1-strict": AdversarySpec("A1-strict", "A1", "strict"),
    "A4": AdversarySpec("A4", "A4", "medium"),
}

A1_OBSERVERS = tuple(f"flow_{k}" for k in FLOW_THRESHOLDS)
A2_OBSERVERS = ("runs", "cusum", "volume_clock", "regularity")
A3_OBSERVER = "learned"
OBSERVERS = A1_OBSERVERS + A2_OBSERVERS + (A3_OBSERVER,)


def make_observers(names, learned_weights: str | None):
    out = {}
    for name in names:
        if name.startswith("flow_"):
            out[name] = FlowDetector(FlowDetectorConfig(threshold=FLOW_THRESHOLDS[name[5:]]))
        elif name in A2_OBSERVERS:
            out[name] = held_out_detectors()[name]
        elif name == A3_OBSERVER:
            if learned_weights is None:
                continue  # A3 needs trained weights (scripts/train_detector.py); skipped if absent
            out[name] = LearnedDetector(learned_weights)
        else:
            raise KeyError(f"unknown observer {name!r}")
    return out


# ----- episodes --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    strategy: str                   # a key of ``all_baselines()``; ignored when visibility == "none"
    size_fraction: float
    adversary: str                  # a key of ``ADVERSARIES``
    visibility: str = "visible"     # visible | invisible (detectors ignore the executor) | none (no executor)
    side: str | None = None         # None: "bid" on even seeds, "ask" on odd seeds
    horizon: int = 2_000
    decision_interval: int = 5
    burn_in: int = 2_000
    after: int = 500                # steps recorded after the horizon (impact reversion)
    observers: tuple[str, ...] = ()
    record_path: bool = False
    base_config: SimConfig | None = None
    regime: str = "default"         # label for grouping E7 regimes; the market itself is in base_config
    prebuilt_config: bool = False   # if set, base_config is the final SimConfig (detection already wired)
    rng: str = "stateful"           # "stateful" or "counter": see ``order_book.rng``

    @property
    def resolved_side(self) -> str:
        return self.side or ("bid" if self.seed % 2 == 0 else "ask")

    @property
    def window(self) -> int:
        return self.horizon + self.after


@dataclass
class EpisodeResult:
    spec: dict
    side: str
    quantity: int
    executed: int
    forced: int
    incomplete: int
    completion: float
    shortfall_bps: float | None
    arrival_mid: float
    final_mid: float
    aggressive_share: float
    orders_sent: int
    end: int | None
    expected_volume: float
    executor_window: tuple[int, int]            # [start, end) in window steps
    institutional_windows: list                  # (start, end, side) in window steps
    signals: dict = field(default_factory=dict)  # observer name -> list of per-step signals
    mid_path: list | None = None
    seconds: float = 0.0

    def to_json(self) -> dict:
        d = asdict(self)
        d["spec"] = {k: v for k, v in self.spec.items() if k != "base_config"}
        return d


def run_episode(spec: EpisodeSpec, calibration: Calibration | None = None,
                learned_weights: str | None = None) -> EpisodeResult:
    t0 = time.perf_counter()
    cfg = spec.base_config if spec.prebuilt_config else ADVERSARIES[spec.adversary].config(spec.base_config)
    if cfg.rng != spec.rng:
        cfg = replace(cfg, rng=spec.rng)
    sim = Simulator(spec.seed, cfg, analysis=True)
    observers = make_observers(spec.observers, learned_weights)
    for name, det in observers.items():
        sim.detectors[f"obs:{name}"] = det
    executor = None
    if spec.visibility != "none":
        # The executor is added before the burn-in so its stream exists; it does nothing until ``start``.
        # Its expected volume is set after the burn-in from the public tape (identical across pairings).
        task = ExecutionTask(side=spec.resolved_side, size_fraction=spec.size_fraction, horizon=spec.horizon,
                             decision_interval=spec.decision_interval)
        executor = sim.add_external(ExecutorAgent, EXECUTOR_ENDOWMENT * cfg.book.mid, EXECUTOR_ENDOWMENT,
                                    task=task, strategy=make_strategy(spec.strategy, calibration),
                                    start=spec.burn_in, expected_volume=1.0)
        if spec.visibility == "invisible":
            sim.market.hide(executor.cid)
    sim.run(spec.burn_in)
    volume = expected_volume_from_tape(sim.market.tape, spec.burn_in)
    if executor is not None:
        executor.expected_volume = volume
        executor.quantity = max(1, round(spec.size_fraction * volume * spec.horizon))
    ps, window = sim.market.public, spec.window
    mids = np.empty(window)
    signals = {name: np.empty(window) for name in observers}
    signals["flow"] = np.empty(window)  # the detector the market makers actually react to
    for t in range(window):
        sim.step()
        mids[t] = ps.mid
        signals["flow"][t] = sim.detectors["flow"].signal
        for name, det in observers.items():
            signals[name][t] = det.signal
    sim.check_conservation()
    inst = [(r.start - spec.burn_in, (r.end if r.end is not None else sim.market.step) - spec.burn_in, r.side)
            for r in sim.metaorders.records if (r.end is None or r.end > spec.burn_in)]
    spec_dict = {k: v for k, v in asdict(spec).items() if k != "base_config"}  # drop the heavy SimConfig
    if executor is None:
        return EpisodeResult(spec_dict, spec.resolved_side, 0, 0, 0, 0, 1.0, None, float(mids[0]), float(mids[-1]),
                             0.0, 0, None, volume, (0, spec.horizon), inst,
                             {k: v.tolist() for k, v in signals.items()},
                             mids.tolist() if spec.record_path else None, time.perf_counter() - t0)
    r = executor.result
    end_rel = (r.end - spec.burn_in) if r.end is not None else spec.horizon
    return EpisodeResult(
        spec_dict, r.side, r.quantity, r.executed, r.forced, r.incomplete, r.completion, r.shortfall_bps,
        r.arrival_mid, float(mids[-1]), r.aggressive_executed / r.executed if r.executed else 0.0, r.orders_sent,
        end_rel, volume, (0, min(end_rel, window)), inst, {k: v.tolist() for k, v in signals.items()},
        mids.tolist() if spec.record_path else None, time.perf_counter() - t0,
    )


# ----- runner ----------------------------------------------------------------------------- #

def _run(args):
    spec, calibration, learned_weights = args
    return run_episode(spec, calibration, learned_weights)


def run_specs(specs: list[EpisodeSpec], calibration: Calibration | None = None, learned_weights: str | None = None,
              workers: int = 1, verbose: bool = True, label: str = "run", sink=None) -> list[EpisodeResult]:
    """Run episodes (in parallel processes when ``workers > 1``), printing one progress line per episode
    and appending each record to ``sink`` (a file-like, JSON lines) when given."""
    results = []
    t0 = time.perf_counter()
    jobs = [(s, calibration, learned_weights) for s in specs]
    for i, r in enumerate(iter_jobs(_run, jobs, workers), 1):
        results.append(r)
        if sink is not None:
            sink.write(json.dumps(r.to_json()) + "\n")
            sink.flush()
        if verbose:
            s = r.spec
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (len(specs) - i)
            is_txt = "   n/a  " if r.shortfall_bps is None else f"{r.shortfall_bps:+7.2f}"
            print(f"[{label} {i:4d}/{len(specs)}] seed={s['seed']} {s['strategy']:<22s} Q/V={s['size_fraction']:.2f} "
                  f"{s['adversary']:<10s} {s['visibility']:<9s} {r.side} IS={is_txt}bps exec={r.executed:5d} "
                  f"forced={r.forced:4d} ({r.seconds:.1f}s, elapsed {elapsed:.0f}s, eta {eta:.0f}s)", flush=True)
    return results


# ----- market regimes ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class RegimeSpec:
    """One market regime: how thin it is and how much legitimate large-order flow hides the executor."""

    name: str
    liquidity: float = 1.0          # scales noise rate, speculator count and maker quote size
    institutions: int = 4           # institutional traders providing cover (0 = none)
    makers: int = 4
    mm_size_scale: float = 1.0      # thins the book independently of ``liquidity``
    capital: int = 1                # scales what speculators and the noise pool hold

    def base(self) -> SimConfig:
        from order_book.presets import regime
        return regime(self.liquidity, self.institutions, self.makers, self.mm_size_scale, capital=self.capital)


DEFAULT_REGIMES = (
    RegimeSpec("deep_cover", 1.0, 4),            # the E1-E3 market: full cover, deep book
    RegimeSpec("half_liquidity", 0.5, 4),        # thinner market, full cover
    RegimeSpec("low_cover", 1.0, 1),             # deep book, little cover
    RegimeSpec("thin_low_cover", 0.5, 1),        # thin and little cover
    RegimeSpec("no_cover", 1.0, 0),              # deep book, no institutions
    RegimeSpec("thin_no_cover", 0.4, 0, mm_size_scale=0.5),  # thin book, no cover: the executor dominates
)

# Ten times the capital for speculators and the noise pool. Long metaorders exhaust the default market's capital and the
# price runs away; these are the robustness market of the decomposition and the market the executor is trained in.
CAPITAL_REGIMES = (RegimeSpec("deep_cover_capital10", 1.0, 4, capital=10), RegimeSpec("low_cover_capital10", 1.0, 1, capital=10))
