"""Hyperparameter sweep for Phase 3 RL executor with verbose live terminal output and Weights & Biases logging.

Features:
  - Streams every parallel worker's per-iteration and per-eval lines live to stdout AND to each run's train.log.
  - Prints overall sweep progress [done/total], active workers, elapsed wall time, and overall sweep ETA.
  - Resumable: skips runs that already have a completed history.json & policy.pt unless --overwrite is passed.
  - Logs each run to Weights & Biases (--wandb), and can retroactively upload existing completed runs (--sync-existing).
  - Prints a ranked summary table at the end.

Examples:
    caffeinate -i .venv/bin/python -m scripts.sweep_rl --workers 6 --wandb
    .venv/bin/python -m scripts.sweep_rl --sync-existing --wandb   # upload existing runs + rank
    .venv/bin/python -m scripts.sweep_rl --quick                   # fast smoke test
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from scripts import OUTPUTS
from scripts._common import write_json


@dataclass(frozen=True)
class SweepJob:
    lr: float
    entropy: float
    epochs: int
    size: float
    iterations: int
    episodes: int
    institutions: int
    horizon: int
    burn_in: int
    eval_every: int
    eval_seeds: int
    seed: int
    root: str = "rl/sweep"

    @property
    def tag(self) -> str:
        lr_s = {1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}.get(self.lr, f"{self.lr:g}")
        return f"lr{lr_s}_ent{self.entropy:g}_ep{self.epochs}_sz{self.size:.2f}"

    @property
    def out_rel(self) -> str:
        return f"{self.root}/{self.tag}"

    @property
    def out_dir(self) -> Path:
        return OUTPUTS / self.root / self.tag

    def is_done(self) -> bool:
        hj = self.out_dir / "history.json"
        pt = self.out_dir / "policy.pt"
        if not (hj.exists() and pt.exists()):
            return False
        try:
            data = json.loads(hj.read_text())
            return len(data.get("history", [])) >= self.iterations
        except Exception:
            return False


def sync_run_to_wandb(job_dir: Path, project: str, entity: str | None, group: str | None, mode: str) -> None:
    """Upload a completed local history.json to Weights & Biases."""
    import wandb
    hj = job_dir / "history.json"
    if not hj.exists():
        return
    data = json.loads(hj.read_text())
    cfg = data["config"]
    twap = data["baseline_twap"]
    tag = job_dir.name
    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=tag,
        id=f"sweep_{tag}",
        resume="allow",
        config={**cfg, "baseline_twap_shortfall_bps": twap["shortfall"]},
        mode=mode,
        reinit="finish_previous",
    )
    eval_by_it = {e["iteration"]: e for e in data.get("evals", [])}
    best_so_far = None
    for row in data.get("history", []):
        it = row["iteration"]
        payload = {
            "iteration": it,
            "train/shortfall_bps": row.get("shortfall"),
            "train/completion": row.get("completion"),
            "train/return": row.get("return"),
            "train/transitions": row.get("transitions"),
            "train/entropy": row.get("entropy"),
            "train/policy_loss": row.get("policy_loss"),
            "train/value_loss": row.get("value_loss"),
            "train/kl": row.get("kl"),
        }
        if it in eval_by_it:
            ev = eval_by_it[it]
            gap = ev.get("gap_vs_twap_bps", ev["shortfall"] - twap["shortfall"])
            if best_so_far is None or ev["shortfall"] < best_so_far["shortfall"]:
                best_so_far = {**ev, "gap_vs_twap_bps": gap}
            payload.update({
                "eval/shortfall_bps": ev["shortfall"],
                "eval/completion": ev["completion"],
                "eval/forced_share": ev.get("forced_share", float("nan")),
                "eval/gap_vs_twap_bps": gap,
                "eval/twap_shortfall_bps": twap["shortfall"],
                "eval/best_shortfall_bps": best_so_far["shortfall"],
            })
        run.log(payload, step=it)
    if best_so_far is not None:
        run.summary["best_eval_shortfall_bps"] = best_so_far["shortfall"]
        run.summary["best_eval_completion"] = best_so_far["completion"]
        run.summary["best_eval_iteration"] = best_so_far["iteration"]
        run.summary["best_gap_vs_twap_bps"] = best_so_far["gap_vs_twap_bps"]
    run.finish()
    print(f"[wandb-sync] uploaded {tag} ({len(data.get('history', []))} iters)", flush=True)


def rank_sweep(root_dir: Path) -> dict:
    """Rank all completed runs in ``root_dir`` per order size."""
    by_size: dict[str, list[dict]] = {}
    for hj in sorted(root_dir.glob("*/history.json")):
        try:
            d = json.loads(hj.read_text())
        except Exception:
            continue
        tag = hj.parent.name
        cfg = d.get("config", {})
        twap = d.get("baseline_twap", {}).get("shortfall", float("nan"))
        evals = d.get("evals", [])
        if not evals:
            continue
        high_compl = [e for e in evals if e.get("completion", 0.0) >= 0.98]
        best = min(high_compl or evals, key=lambda e: e["shortfall"])
        last = evals[-1]
        size_key = f"{float(cfg.get('size', 0.2)):.2f}"
        by_size.setdefault(size_key, []).append({
            "tag": tag,
            "best_shortfall": best["shortfall"],
            "best_completion": best["completion"],
            "best_iteration": best["iteration"],
            "last_shortfall": last["shortfall"],
            "last_completion": last["completion"],
            "twap_shortfall": twap,
            "gap_vs_twap": best["shortfall"] - twap,
            "iterations_done": len(d.get("history", [])),
        })
    for rows in by_size.values():
        rows.sort(key=lambda r: r["best_shortfall"])
    return by_size


def format_ranking(by_size: dict) -> str:
    lines = []
    for sz, rows in sorted(by_size.items()):
        lines.append(f"\n=== SWEEP RANKING (Q/V = {sz}) ===")
        lines.append(f"{'config':<34}{'bestIS':>8}{'bCompl':>8}{'bIter':>6}"
                     f"{'lastIS':>8}{'lCompl':>8}{'TWAP':>8}{'gap':>8}")
        for r in rows:
            lines.append(
                f"{r['tag']:<34}{r['best_shortfall']:+8.2f}{r['best_completion']:8.3f}{r['best_iteration']:6d}"
                f"{r['last_shortfall']:+8.2f}{r['last_completion']:8.3f}{r['twap_shortfall']:+8.2f}"
                f"{r['gap_vs_twap']:+8.2f}"
            )
    return "\n".join(lines)


def _run_worker(job: SweepJob, idx: int, total: int, args, state: dict, lock: threading.Lock) -> None:
    job.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.out_dir / "train.log"
    cmd = [
        sys.executable, "-u", "-m", "scripts.train_rl",
        "--iterations", str(job.iterations),
        "--episodes", str(job.episodes),
        "--epochs", str(job.epochs),
        "--lr", str(job.lr),
        "--entropy", str(job.entropy),
        "--size", str(job.size),
        "--institutions", str(job.institutions),
        "--horizon", str(job.horizon),
        "--burn-in", str(job.burn_in),
        "--eval-every", str(job.eval_every),
        "--eval-seeds", str(job.eval_seeds),
        "--seed", str(job.seed),
        "--out", job.out_rel,
        "--wandb-project", args.wandb_project,
        "--wandb-group", args.wandb_group,
        "--wandb-name", job.tag,
        "--wandb-mode", "online" if args.wandb and args.wandb_mode == "disabled" else args.wandb_mode,
    ]
    if args.wandb:
        cmd.append("--wandb")
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    t_job = time.perf_counter()
    with lock:
        state["active"].add(job.tag)
        print(f"[sweep start {idx:2d}/{total}] {job.tag} (active={len(state['active'])})", flush=True)

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            with lock:
                state["steps_done"] += 1 if "] IS " in line else 0
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()

    dt = time.perf_counter() - t_job
    with lock:
        state["active"].discard(job.tag)
        state["jobs_done"] += 1
        done = state["jobs_done"]
        elapsed = time.perf_counter() - state["t0"]
        eta = elapsed / done * ( state["jobs_to_run"] - done ) if done else 0.0
        print(f"[sweep done  {done:2d}/{state['jobs_to_run']} | total { state['already_done'] + done }/{total}] "
              f"{job.tag} rc={proc.returncode} in {dt:.0f}s "
              f"(wall {elapsed/60:.1f}m, sweep ETA {eta/60:.1f}m, active={len(state['active'])})",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--entropies", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    parser.add_argument("--epochs-list", type=int, nargs="+", default=[4, 10])
    parser.add_argument("--sizes", type=float, nargs="+", default=[0.20, 0.40])
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--institutions", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=2_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-seeds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--root", type=str, default="rl/sweep")
    parser.add_argument("--overwrite", action="store_true", help="re-run even if history.json already completed")
    parser.add_argument("--sync-existing", action="store_true", help="upload already-completed runs to wandb")
    parser.add_argument("--rank-only", action="store_true", help="only print the ranking of completed runs")
    parser.add_argument("--wandb", action="store_true", help="log runs to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="stealth-execution")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="phase3-ppo-sweep")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.lrs, args.entropies, args.epochs_list, args.sizes = [3e-4], [0.01], [2], [0.20]
        args.iterations, args.episodes, args.horizon, args.burn_in = 3, 2, 300, 400
        args.eval_every, args.eval_seeds, args.workers = 3, 2, 1

    root_dir = OUTPUTS / args.root
    root_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        SweepJob(lr=lr, entropy=ent, epochs=ep, size=sz, iterations=args.iterations,
                 episodes=args.episodes, institutions=args.institutions, horizon=args.horizon,
                 burn_in=args.burn_in, eval_every=args.eval_every, eval_seeds=args.eval_seeds,
                 seed=args.seed, root=args.root)
        for lr, ent, ep, sz in itertools.product(args.lrs, args.entropies, args.epochs_list, args.sizes)
    ]

    if args.sync_existing:
        wb_mode = "online" if (args.wandb and args.wandb_mode == "disabled") else (
            args.wandb_mode if args.wandb_mode != "disabled" else "online"
        )
        done_jobs = [j for j in jobs if j.is_done()]
        print(f"[sweep] syncing {len(done_jobs)} completed runs to W&B (project={args.wandb_project}, "
              f"group={args.wandb_group}, mode={wb_mode})", flush=True)
        for j in done_jobs:
            sync_run_to_wandb(j.out_dir, args.wandb_project, args.wandb_entity, args.wandb_group, wb_mode)

    if not args.rank_only:
        todo = jobs if args.overwrite else [j for j in jobs if not j.is_done()]
        already = len(jobs) - len(todo)
        print(f"[sweep] total={len(jobs)} already_completed={already} remaining={len(todo)} "
              f"workers={args.workers} iterations={args.iterations} episodes/it={args.episodes}", flush=True)
        if todo:
            state = {
                "t0": time.perf_counter(), "active": set(), "jobs_done": 0,
                "jobs_to_run": len(todo), "already_done": already, "steps_done": 0,
            }
            lock = threading.Lock()
            sem = threading.BoundedSemaphore(args.workers)
            threads = []
            for idx, job in enumerate(todo, 1):
                sem.acquire()
                def _target(j=job, i=idx):
                    try:
                        _run_worker(j, i, len(jobs), args, state, lock)
                    finally:
                        sem.release()
                t = threading.Thread(target=_target, daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

    ranking = rank_sweep(root_dir)
    print(format_ranking(ranking), flush=True)
    write_json({"ranking": ranking}, root_dir / "summary.json")


if __name__ == "__main__":
    main()
