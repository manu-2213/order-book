"""The paired-seed protocol: what counter-based draws buy over shared stateful seeds.

    python -m papers.decomposition.scripts.crn_variance --workers 8        # 5 arms x 3 strategies x 2 sizes x 40 seeds x 2 pairing schemes
    python -m papers.decomposition.scripts.crn_variance --quick            # smoke test, seconds

For each contrast between arms and each pairing scheme: the variance reduction against independent seeds (Klein et
al.) and Sharma's positive-correlation condition. Then how much of the background's order flow survives pairing over
the window, which is where a shifted stateful stream shows up directly.
Outputs: outputs/decomposition/protocol/episodes.jsonl, outputs/decomposition/protocol/summary.json.
"""

import argparse
from dataclasses import replace

import numpy as np

from order_book.counterfactual import FULL, PairedSpec, flow_coupling
from order_book.provenance import write_manifest
from order_book.runner import iter_jobs, shard_dir
from papers.decomposition.decomposition import ProtocolConfig, format_protocol, merge_grid, run_grid, run_shard, summarise_protocol
from papers.decomposition import OUT
from scripts._common import write_json


def _coupling(spec):
    return spec.rng, spec.strategy, flow_coupling(spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40, help="test seeds 1000..1000+n-1")
    parser.add_argument("--resume", action="store_true", help="keep the runs already in episodes.jsonl and do the rest")
    parser.add_argument("--shard", type=int, default=None, help="run only this seed (1-based, a Slurm array task id), then stop")
    parser.add_argument("--merge", action="store_true", help="combine finished shards and write the summary")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = ProtocolConfig(seeds=tuple(range(1000, 1000 + args.seeds)))
    if args.quick:
        cfg = replace(cfg, strategies=("twap_market",), sizes=(0.2,), seeds=(1000, 1001, 1002, 1003), horizon=300, burn_in=500, after=100)
    out = OUT / ("protocol_quick" if args.quick else "protocol")  # a smoke test never overwrites real results
    out.mkdir(parents=True, exist_ok=True)
    specs = cfg.specs()
    print(f"[protocol] {len(specs)} runs: {cfg.modes} x {cfg.strategies} x {cfg.sizes} x 5 arms x {len(cfg.seeds)} seeds")
    if args.shard is not None:
        write_manifest(shard_dir(out, args.shard), cfg, True)
        run_shard(specs, cfg.seeds, args.shard, out, None, args.workers, True, "protocol")
        return
    if args.merge:
        results = merge_grid(specs, cfg.seeds, out)
        write_manifest(out, cfg)
    else:
        write_manifest(out, cfg, args.resume)
        results = run_grid(specs, out / "episodes.jsonl", None, args.workers, True, "protocol", args.resume)
    summary = summarise_protocol(results)
    print("\n" + format_protocol(summary))

    base = cfg.regime.base()
    jobs = [PairedSpec(seed, strategy, cfg.sizes[-1], FULL, horizon=cfg.horizon, burn_in=cfg.burn_in, after=cfg.after,
                       base_config=base, regime=cfg.regime.name, rng=mode)
            for mode in cfg.modes for strategy in cfg.strategies for seed in cfg.seeds[:10]]
    rows = list(iter_jobs(_coupling, jobs, args.workers))
    coupling: dict = {}
    for mode, strategy, overlap in rows:
        coupling.setdefault(mode, {}).setdefault(strategy, []).append(overlap)
    summary["flow_coupling"] = {mode: {s: np.mean(v, axis=0).tolist() for s, v in by.items()} for mode, by in coupling.items()}
    print("\nbackground order flow shared by Arm 0 and Arm 5, per fifth of the window (Jaccard):")
    for mode, by in summary["flow_coupling"].items():
        for strategy, overlap in by.items():
            print(f"  {mode:<10}{strategy:<16}" + " ".join(f"{x:.3f}" for x in overlap))
    write_json({"config": {k: v for k, v in cfg.__dict__.items()}, "summary": summary}, out / "summary.json")


if __name__ == "__main__":
    main()
