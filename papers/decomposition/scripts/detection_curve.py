"""The detection curve: P(fire) and latency of the A1/A3 detectors vs participation, jitter and passive share.

    python -m papers.decomposition.scripts.detection_curve --workers 8                 # every axis x 3 thresholds x visible/invisible x 20 seeds
    python -m papers.decomposition.scripts.detection_curve --regime deep_cover --workers 8
    python -m papers.decomposition.scripts.detection_curve --quick                     # smoke test, seconds

Also reports the leakage at every point and the randomisation value per threshold (Yang & Zhu).
The A3 observer needs outputs/detectors/learned.npz (scripts/train_detector.py) and is skipped without it.
Outputs: outputs/decomposition/detection_curve/episodes.jsonl, outputs/decomposition/detection_curve/summary.json.
"""

import argparse
from dataclasses import replace

from papers.decomposition.detection_curve import (
    DetectionCurveConfig, format_detection_curve, merge_curve, run_curve_shard, run_detection_curve, summarise_detection_curve,
)
from order_book.experiments import DEFAULT_REGIMES
from order_book.provenance import write_manifest
from order_book.runner import shard_dir
from papers.decomposition import OUT
from scripts._common import calibration_or_default, learned_weights_or_none, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="test seeds 1000..1000+n-1")
    parser.add_argument("--regime", default="low_cover", choices=[r.name for r in DEFAULT_REGIMES])
    parser.add_argument("--thresholds", nargs="+", default=None, choices=("lenient", "medium", "strict"))
    parser.add_argument("--fire-threshold", type=float, default=0.5, help="signal in the executor's direction that counts as firing")
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--out", default="detection_curve", help="output directory under outputs/decomposition/")
    parser.add_argument("--resume", action="store_true", help="keep the episodes already in episodes.jsonl and do the rest")
    parser.add_argument("--shard", type=int, default=None, help="run only this seed (1-based, a Slurm array task id), then stop")
    parser.add_argument("--merge", action="store_true", help="combine finished shards and write the summary")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    regime = next(r for r in DEFAULT_REGIMES if r.name == args.regime)
    cfg = DetectionCurveConfig(seeds=tuple(range(1000, 1000 + args.seeds)), regime=regime, fire_threshold=args.fire_threshold,
                               burn_in=args.burn_in)
    if args.thresholds:
        cfg = replace(cfg, thresholds=tuple(args.thresholds))
    if args.quick:
        cfg = replace(cfg, thresholds=("medium",), participation=(0.1, 0.4), timing_jitter=(0.0, 0.5), size_jitter=(0.0, 0.5),
                      passive_share=(0.0, 0.5), seeds=(1000, 1001), horizon=300, burn_in=500, after=100)
    out = OUT / (f"{args.out}_quick" if args.quick else args.out)  # a smoke test never overwrites real results
    out.mkdir(parents=True, exist_ok=True)
    specs = cfg.specs()
    print(f"[curve] {len(specs)} episodes: thresholds={cfg.thresholds} regime={cfg.regime.name} seeds={cfg.seeds[0]}..{cfg.seeds[-1]}")
    calibration, weights = calibration_or_default(), learned_weights_or_none()
    if args.shard is not None:
        write_manifest(shard_dir(out, args.shard), cfg, True)
        run_curve_shard(cfg, args.shard, out, calibration, weights, args.workers)
        return
    if args.merge:
        records = merge_curve(cfg, out)
        write_manifest(out, cfg)
    else:
        write_manifest(out, cfg, args.resume)
        records = run_detection_curve(cfg, calibration, weights, args.workers, True, out / "episodes.jsonl", args.resume)
    summary = summarise_detection_curve(records, cfg)
    print(format_detection_curve(summary))
    write_json({"config": {k: v for k, v in cfg.__dict__.items()}, "summary": summary}, out / "summary.json")


if __name__ == "__main__":
    main()
