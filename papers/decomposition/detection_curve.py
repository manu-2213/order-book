"""The detection curve: how often and how fast detectors fire as an executor changes how it trades.

A dialled TWAP (``DialTwap``) is swept along one axis at a time around a plain
TWAP: participation, timing jitter, size jitter and passive share. Each point
runs at each flow-detector threshold of the market, with the A1 rule detectors
at all three thresholds and the A3 learned detector attached as observers, and
visible and invisible to the makers, so one run gives

    p_fire    share of episodes in which the detector fired in the executor's
              direction while it was active, against the same on Arm 0's paths
    latency   median and 90th percentile steps to the first such firing, with
              episodes that never fired counted at the end of the window
    duty      share of active steps spent firing, against the same on Arm 0
    signal    mean signal in the executor's direction, against the same on Arm 0;
              needs no threshold, so it still moves where p_fire has saturated
    leakage   IS(visible) - IS(invisible), paired by seed

Over a long window a detector almost always fires at least once on natural flow,
so p_fire saturates near 1 on Arm 0 as well; duty, signal and latency against
their Arm-0 values are what discriminate. This explains the Arm 4 and 5 shares and tests Yang & Zhu's prediction that
executors should randomise once detectors are precise: if it holds, what timing
jitter saves in leakage grows as the threshold tightens (``randomisation_value``).
Episodes are reduced to these numbers in the worker, because the raw per-step
signals of thousands of episodes would not fit in memory.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from order_book.execution import Calibration
from order_book.experiments import A1_OBSERVERS, A3_OBSERVER, EpisodeSpec, RegimeSpec, TEST_SEEDS, run_episode
from order_book.runner import IncompleteShards, iter_jobs, load_results, load_shards, missing_shards, seed_of_shard, shard_dir
from order_book.scoring import bootstrap_ci, detection_delay, paired_difference

AXES = ("participation", "timing_jitter", "size_jitter", "passive_share")


@dataclass(frozen=True)
class DetectionCurveConfig:
    name: str = "detection_curve"
    thresholds: tuple[str, ...] = ("lenient", "medium", "strict")
    participation: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40)
    timing_jitter: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    size_jitter: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    passive_share: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    base_size: float = 0.20
    observers: tuple[str, ...] = A1_OBSERVERS + (A3_OBSERVER,)
    fire_threshold: float = 0.5
    seeds: tuple[int, ...] = tuple(TEST_SEEDS)
    regime: RegimeSpec = RegimeSpec("low_cover", 1.0, 1)  # the market where E7b found TWAP leakage measurable
    horizon: int = 2_000
    decision_interval: int = 5
    burn_in: int = 2_000
    after: int = 500
    rng: str = "counter"

    def points(self) -> dict:
        """axis -> [(dial value, strategy, size fraction)], each axis sweeping around the plain TWAP."""
        dial = lambda name, v: f"twap@{name}={v:g}" if v else "twap@"  # a dial at zero is the shared plain TWAP
        return {"participation": [(v, "twap@", v) for v in self.participation],
                "timing_jitter": [(v, dial("timing", v), self.base_size) for v in self.timing_jitter],
                "size_jitter": [(v, dial("size", v), self.base_size) for v in self.size_jitter],
                "passive_share": [(v, dial("passive", v), self.base_size) for v in self.passive_share]}

    def specs(self) -> list[EpisodeSpec]:
        base = self.regime.base()
        common = dict(horizon=self.horizon, decision_interval=self.decision_interval, burn_in=self.burn_in, after=self.after,
                      observers=self.observers, base_config=base, regime=self.regime.name, rng=self.rng)
        unique = sorted({(strategy, size) for axis in self.points().values() for _, strategy, size in axis})
        specs = []
        for threshold in self.thresholds:
            adversary = f"A1-{threshold}"
            for seed in self.seeds:
                specs.append(EpisodeSpec(seed, "none", 0.0, adversary, "none", **common))
                for strategy, size in unique:
                    for visibility in ("visible", "invisible"):
                        specs.append(EpisodeSpec(seed, strategy, size, adversary, visibility, **common))
        return specs


def score_episode(spec: EpisodeSpec, result, fire_threshold: float) -> dict:
    """The per-episode numbers: for each detector, whether it fired, when, and how much of the time."""
    start, end = result.executor_window
    if spec.visibility == "none":
        start, end = 0, spec.horizon
    length = max(end - start, 1)
    scores = {}
    for name, series in result.signals.items():
        delay = detection_delay(np.asarray(series), start, end, result.side, fire_threshold)
        sign = 1 if result.side == "bid" else -1
        window = sign * np.asarray(series[start:end], float)
        scores[name] = {"fired": delay is not None, "latency": delay if delay is not None else length,
                        "duty": float((window >= fire_threshold).mean()) if window.size else 0.0,
                        "signal": float(window.mean()) if window.size else 0.0}
    return {"seed": spec.seed, "strategy": spec.strategy, "size": spec.size_fraction, "adversary": spec.adversary,
            "visibility": spec.visibility, "side": result.side, "shortfall_bps": result.shortfall_bps,
            "quantity": result.quantity, "aggressive_share": result.aggressive_share, "scores": scores}


def _run(args):
    spec, calibration, weights, fire_threshold = args
    return score_episode(spec, run_episode(spec, calibration, weights), fire_threshold)


def _record_key(seed, strategy, size, adversary, visibility) -> tuple:
    return seed, strategy, size, adversary, visibility


def _spec_key(s: EpisodeSpec) -> tuple:
    return _record_key(s.seed, s.strategy, s.size_fraction, s.adversary, s.visibility)


def run_detection_curve(cfg: DetectionCurveConfig, calibration: Calibration | None = None, weights: str | None = None,
                        workers: int = 1, verbose: bool = True, path: Path | None = None, resume: bool = False,
                        seed: int | None = None) -> list[dict]:
    """Every episode of the curve, or of one ``seed``, one JSONL record each to ``path``. With ``resume``, logged
    episodes are not repeated."""
    records = load_results(path) if resume and path is not None else []
    finished = {_record_key(r["seed"], r["strategy"], r["size"], r["adversary"], r["visibility"]) for r in records}
    specs = [s for s in cfg.specs() if _spec_key(s) not in finished and (seed is None or s.seed == seed)]
    jobs = [(s, calibration, weights, cfg.fire_threshold) for s in specs]
    t0 = time.perf_counter()
    sink = open(path, "a" if resume else "w") if path is not None else None
    try:
        for i, record in enumerate(iter_jobs(_run, jobs, workers, chunksize=4), 1):
            records.append(record)
            if sink is not None:
                sink.write(json.dumps(record) + "\n")
                sink.flush()
            if verbose and (i % 50 == 0 or i == len(jobs)):
                elapsed = time.perf_counter() - t0
                print(f"[curve {i:5d}/{len(jobs)}] elapsed {elapsed:.0f}s, eta {elapsed / i * (len(jobs) - i):.0f}s", flush=True)
    finally:
        if sink is not None:
            sink.close()
    return records


def run_curve_shard(cfg: DetectionCurveConfig, shard: int, out: Path, calibration: Calibration | None = None,
                    weights: str | None = None, workers: int = 1, verbose: bool = True) -> list[dict]:
    """One seed of the curve, in its own folder, always resuming."""
    directory = shard_dir(out, shard)
    directory.mkdir(parents=True, exist_ok=True)
    return run_detection_curve(cfg, calibration, weights, workers, verbose, directory / "episodes.jsonl", True,
                               seed_of_shard(cfg.seeds, shard))


def merge_curve(cfg: DetectionCurveConfig, out: Path) -> list[dict]:
    """Every record of a sharded curve, also written to ``out/episodes.jsonl``. Raises ``IncompleteShards`` if any is missing."""
    records = load_shards(out)
    by_key = {_record_key(r["seed"], r["strategy"], r["size"], r["adversary"], r["visibility"]): r for r in records}
    specs = cfg.specs()
    missing = missing_shards({_spec_key(s): s.seed for s in specs}, set(by_key), cfg.seeds)
    if missing:
        raise IncompleteShards(missing, out)
    merged = [by_key[_spec_key(s)] for s in specs]
    with open(Path(out) / "episodes.jsonl", "w") as sink:
        for record in merged:
            sink.write(json.dumps(record) + "\n")
    return merged


def summarise_detection_curve(records: list[dict], cfg: DetectionCurveConfig) -> dict:
    """threshold -> axis -> one row per dial value, and the randomisation value per threshold."""
    index: dict = {}
    baseline: dict = {}
    for r in records:
        if r["visibility"] == "none":
            baseline.setdefault(r["adversary"], []).append(r)
        else:
            index.setdefault((r["adversary"], r["strategy"], r["size"]), {}).setdefault(r["visibility"], {})[r["seed"]] = r
    out: dict = {}
    for threshold in cfg.thresholds:
        adversary = f"A1-{threshold}"
        base_rows = baseline.get(adversary, [])
        by_axis: dict = {}
        for axis, points in cfg.points().items():
            rows = []
            for value, strategy, size in points:
                group = index.get((adversary, strategy, size))
                if not group or "visible" not in group:
                    continue
                visible = group["visible"]
                seeds = sorted(visible)
                detectors = {}
                for name in visible[seeds[0]]["scores"]:
                    fired = [visible[s]["scores"][name]["fired"] for s in seeds]
                    latency = [visible[s]["scores"][name]["latency"] for s in seeds]
                    duty = [visible[s]["scores"][name]["duty"] for s in seeds]
                    signal = [visible[s]["scores"][name]["signal"] for s in seeds]
                    arm0 = [b["scores"][name] for b in base_rows if name in b["scores"]]
                    detectors[name] = {
                        "p_fire": bootstrap_ci(np.array(fired, float)),
                        "latency_median": float(np.median(latency)), "latency_p90": float(np.percentile(latency, 90)),
                        "duty": bootstrap_ci(duty),
                        "p_fire_arm0": float(np.mean([a["fired"] for a in arm0])) if arm0 else float("nan"),
                        "duty_arm0": float(np.mean([a["duty"] for a in arm0])) if arm0 else float("nan"),
                        "signal": bootstrap_ci(signal),
                        "signal_arm0": float(np.mean([a["signal"] for a in arm0])) if arm0 else float("nan"),
                    }
                row = {"value": value, "strategy": strategy, "size": size, "episodes": len(seeds), "detectors": detectors,
                       "is_visible": bootstrap_ci([visible[s]["shortfall_bps"] for s in seeds]),
                       "realised_passive_share": 1 - float(np.mean([visible[s]["aggressive_share"] for s in seeds]))}
                if "invisible" in group:
                    common = sorted(set(visible) & set(group["invisible"]))
                    row["leakage"] = paired_difference([visible[s]["shortfall_bps"] for s in common],
                                                       [group["invisible"][s]["shortfall_bps"] for s in common])
                rows.append(row)
            by_axis[axis] = rows
        out[threshold] = {"axes": by_axis, "randomisation_value": _randomisation_value(index, adversary, cfg)}
    return out


def _randomisation_value(index: dict, adversary: str, cfg: DetectionCurveConfig) -> dict | None:
    """Leakage of the plain TWAP minus leakage at the largest timing jitter, paired by seed."""
    points = cfg.points()["timing_jitter"]
    plain = index.get((adversary, points[0][1], points[0][2]))
    jittered = index.get((adversary, points[-1][1], points[-1][2]))
    if not plain or not jittered or "invisible" not in plain or "invisible" not in jittered:
        return None
    seeds = sorted(set(plain["visible"]) & set(plain["invisible"]) & set(jittered["visible"]) & set(jittered["invisible"]))
    leak = lambda group: np.array([group["visible"][s]["shortfall_bps"] - group["invisible"][s]["shortfall_bps"] for s in seeds])
    return paired_difference(leak(plain), leak(jittered))


def format_detection_curve(summary: dict) -> str:
    lines = []
    for threshold, block in summary.items():
        lines.append(f"\n=== market threshold: {threshold} ===")
        for axis, rows in block["axes"].items():
            lines.append(f"-- {axis} --")
            lines.append(f"{'value':>7}{'detector':>15}{'P(fire)':>9}{'Arm 0':>7}{'lat med':>9}{'lat p90':>9}{'duty':>7}{'Arm 0':>7}{'signal':>8}{'Arm 0':>7}{'leak':>8}{'[95% CI]':>18}")
            for row in rows:
                leak = row.get("leakage")
                leak_txt = f"{leak['mean']:>8.2f}  [{leak['low']:+.2f},{leak['high']:+.2f}]" if leak else ""
                for name, d in row["detectors"].items():
                    lines.append(f"{row['value']:>7.2f}{name:>15}{d['p_fire'][0]:>9.2f}{d['p_fire_arm0']:>7.2f}"
                                 f"{d['latency_median']:>9.0f}{d['latency_p90']:>9.0f}{d['duty'][0]:>7.3f}{d['duty_arm0']:>7.3f}{d['signal'][0]:>8.3f}{d['signal_arm0']:>7.3f}{leak_txt}")
                    leak_txt = ""
        value = block["randomisation_value"]
        if value:
            lines.append(f"randomisation value (leakage plain - leakage at max timing jitter): {value['mean']:+.2f} "
                         f"[{value['low']:+.2f},{value['high']:+.2f}] bps, p={value['p_two_sided']:.3f}")
    return "\n".join(lines)
