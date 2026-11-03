# Running on the cluster

The commands to copy are in `hpc_commands.txt`, a local file git does not track.

The same workflow as DeepHedging: sync with `rsync` from the Mac, one short Slurm file per run, a smoke test first, and `submit_all.sh` for the grid.
Each command says where it runs. **None of this has been run on the cluster yet**, so the first run below is also the test of these scripts.

The differences from DeepHedging: order_book is CPU-only and parallelises over processes, so jobs use the default `compute` partition with many cores, no GPU and no account. There
are no banks to build. Every job asks for 1 hour, so it qualifies for the short queue. Job files follow [the cluster's guide](https://docs.hpc.qmul.ac.uk/using/submittingjobs/): cores with `-n` and read back as `$SLURM_NTASKS`, not `-c`; memory with
`--mem-per-cpu`; output named with `%x.o%j`. The one addition is `-N 1`, because workers are processes on one machine and the guide's example does not guarantee that.

| Item | Value |
|---|---|
| Login | `acw796@login.hpc.qmul.ac.uk` |
| Repo on the cluster | `/data/home/acw796/order-book` |
| Everything cluster-specific | `hpc/settings.sh` |
| Venv | `.venv` on the cluster, built by `hpc/setup_venv.sh` |
| Results | `outputs/<paper>/<run>/`, each with `manifest.json` |
| Logs | `papers/<paper>/slurm/logs/<job>.o<id>` and `.e<id>` |
| Limits | 240 h maximum runtime; `compute` nodes have 36 or 48 cores and about 362 GB |

## Before the first run

**On the Mac, check the rsync filter still lists what the job needs:** `hpc/rsync.filter` names the two PPO policies, the calibration and the learned detector. If a file is missing the
job fails or quietly runs without it: without `outputs/detectors/learned.npz` the A3 observer is skipped.

## Launch sequence

**1. Mac: push.** Add `-n` for a dry run first.

```bash
bash hpc/push.sh -n
bash hpc/push.sh
```

**2. Cluster: build the venv**, once.

```bash
cd /data/home/acw796/order-book && bash hpc/setup_venv.sh
```

**3. Cluster: smoke test**, at most an hour. It runs the suite and every script end to end at tiny sizes.

```bash
bash hpc/submit.sh decomposition smoke_test
squeue --me
```

Then read `papers/decomposition/slurm/logs/smoke_test.o<id>`:

- **`ALL PASSED`**: the job got to the end. `submit.sh` refuses real jobs without this line.
- **`FINGERPRINT MATCHES` or `FINGERPRINT DIFFERS`**: whether the cluster reproduces the Mac's runs bit for bit. `DIFFERS` is not a failure. It means results here are valid but not
  byte-comparable with the pilot, so compare them statistically and do not mix runs from the two machines in one grid.
- **The benchmark lines**: real steps per second, to check the walltimes below.
- **`logs/smoke_test.e<id>`**: tracebacks.

**4. Cluster: submit.**

```bash
bash papers/decomposition/slurm/submit_all.sh          # e8_deep_cover, e8_low_cover, protocol, detection_curve
bash papers/decomposition/slurm/submit_all.sh extras   # e8_horizons, detection_curve_deep, replications
bash papers/decomposition/slurm/submit_all.sh all      # all of them
```

**Every job is split into 1-hour tasks**, because those start at once in the short queue. Each grid is a Slurm array with one seed per task, writing to `outputs/<paper>/<job>/shards/<task>/`, and
`submit.sh` queues a `<job>_merge` job after it. The merge checks every run is there, writes `episodes.jsonl` and `summary.json`, and emails you when it ends.

If a task is killed at the hour, the merge fails and its log gives the command to resubmit what is missing. A task keeps its finished runs, so a retry only does the rest:

```bash
ARRAY=3,7 bash hpc/submit.sh decomposition e8_deep_cover
```

Estimates come from the pilot's timings on the Mac, where the slowest seed was 1.4 times the median and no single run took more than 3 seconds. They leave room for a slower cluster, but correct them with the smoke test's speed.

| Job | Tasks | Runs | CPU minutes per task | Cores | Time per task |
|---|---|---|---|---|---|
| `e8_deep_cover` | 20 | 24,000 | 15-20 | 4 | 5 min |
| `e8_low_cover` | 20 | 28,000 | 20-30 | 4 | 7 min |
| `e8_horizons` | 20 | 18,000 | about 25 | 8 | 3 min |
| `protocol` | 40 | 2,400 | 1 | 4 | under 1 min |
| `detection_curve`, `detection_curve_deep` | 20 each | 2,100 each | 2 | 4 | under 1 min |
| `replications` | 20 | 1,360 | 2 | 4 | under 1 min |
| `e8_capital_deep`, `e8_capital_low` | 20 each | 24,000 each | 15-20 | 4 | 5 min |
| `e8_capital_horizons` | 20 | 18,000 | about 25 | 8 | 3 min |
| `train_rl` | 10 variants | 1,000 PPO or GRPO iterations each | about 370 per hour per task (9.6 s an iteration) | 16 | 3 chained hours; `CHAIN=5` leaves a margin, unused links exit at once |
| `rl_eval` | 1 | 10 policies x up to 6 cells x 120 seeds, plus TWAP once per episode; also writes `analysis.json` and `episodes.jsonl` | about 25 | 16 | 20 min |
| `e8_rl` | 20 | 14,400 | about 10 | 8 | 3 min |
| `rl_ceiling` | 1 | 4,720 | about 10 | 16 | 15 min |
| `train_rl_large_tick` | 6 variants | 1,000 iterations each, large-tick market | as `train_rl` | 16 | 3 chained hours, `CHAIN=5` |
| `rl_eval_large_tick`, `e8_rl_large_tick` (20 tasks, 10,800 runs), `rl_ceiling_large_tick` | | the same results in the large-tick market | | | as above |
| `runaway_causes` | 1 | 1,120 | about 20 | 16 | 20 min |
| `market_probe` | 1 | 13 configs x 20 seeds | a few | 16 | a few min |
| `rl_heldout` (stealth_execution) | 1 | 2 policies x 120 seeds x 2 | about 10 | 16 | a few min |

**The final run.** Every number the paper quotes comes from one simulator version, so after a version bump everything is rerun:
regenerate the inputs on the Mac first (`python -m scripts.calibrate`, then `python -m scripts.train_detector`; about 15 minutes),
push, run the smoke test, then `bash papers/decomposition/slurm/submit_all.sh final`, which submits the grids, the single jobs, the
stealth paper's `rl_heldout`, the training chain and, behind it, the results jobs. Delete the old `outputs/decomposition` and
`outputs/stealth_execution/rl_heldout` on the cluster first: shards and `train_rl` resume, and `write_manifest` refuses to resume
across versions, so stale folders fail loudly rather than mix.

`train_rl` is the one job that outlives an hour. Each task checkpoints before the walltime and the next link of the chain resumes it;
a link that finds its run finished exits at once, so an extra link costs nothing. `submit_all.sh rl` chains 5 links and queues `rl_eval` and
`e8_rl` behind the last one (`AFTER=<job id>` in `hpc/submit.sh`), so one command gives the policies, their scores and their decomposition.
`submit_all.sh large_tick` does the same for the large-tick market (six variants, then `large_tick_results`); `final` runs both.
If a task's log does not end in `done` after the chain, `CHAIN=2 bash hpc/submit.sh decomposition train_rl` adds hours and `submit_all.sh rl_results`
redoes the results from the finished policies. Delete `outputs/decomposition/e8_rl` on the cluster before any rerun with a different sweep: shards resume.

`e2`, `e3`, `e7`, `e7b` and `rl_heldout` in `papers/stealth_execution/slurm/` are not split: each is a single 1-hour job on 16 cores. They are not resumable, so if one hits the hour, raise its `-t` or cores.

**Decide before the final grid:** the default burn-in of 2,000 keeps results comparable with E3, but volume is still rising then. A longer burn-in means adding `--burn-in 5000` to
both the array and merge job files, and the results are then not comparable with the pilot.

## Monitoring and results

```bash
squeue --me                                                          # cluster
tail -f papers/decomposition/slurm/logs/e8_deep_cover.o<id>.<task>   # cluster: progress of one task
jobstats                                                             # cluster: memory requested against used; lower --mem-per-cpu if far apart
scancel <id>                                                         # cluster
ARRAY=3,7 bash hpc/submit.sh decomposition e8_deep_cover             # cluster: resubmit tasks a merge reported missing
bash hpc/pull.sh decomposition                                       # Mac: summaries and manifests
bash hpc/pull.sh decomposition --all                                 # Mac: also the per-episode logs
```

To build the notes from the full run, on the **Mac**, after pulling:

```bash
.venv/bin/python -m papers.decomposition.scripts.make_tables --root outputs/decomposition --deep e8_deep_cover --heldout outputs/stealth_execution/rl_heldout/summary.json --horizons e8_horizons --curve-deep detection_curve_deep
cd papers/decomposition/paper && latexmk -pdf notes.tex
```

The tables update themselves. The prose quotes the pilot and has to be revised by hand. `--low e8_low_cover` already matches the job's output folder.

## Traps

Inherited from DeepHedging, plus this repo's own.

- **Submit through `hpc/submit.sh`, not `sbatch`.** It creates `logs/` first, without which a job dies with no log, and checks the smoke test. `sbatch *.slurm` submits only the first file anyway.
- **rsync never deletes on the cluster.** Removed or renamed local files linger there, and stale job files can be submitted by mistake. Clean up with an explicit `rm`. Do not use
  `--delete`: it would wipe cluster-only results.
- **The Mac's rsync is Apple's openrsync.** `push.sh` was tested with it against a local folder, not against the cluster. If it errors against the cluster, `brew install rsync`.
- **Do not push engine code while jobs are queued.** A job that starts later would run different code from the ones before it. On a change in behaviour, manifests refuse to resume across simulator versions.
  A change that keeps the fingerprints but alters behaviour they do not reach would slip through, so wait, or use a new output folder.
- **Results are never pushed.** `hpc/rsync.filter` sends only the listed inputs from `outputs/`, so a push cannot overwrite a result.
- **No `.git` on the cluster.** `push.sh` writes the commit to `.source_version`, and manifests cite it. Commit before pushing, or manifests will say `dirty: true`.
- **Tasks can be resubmitted as often as needed:** results are flushed line by line.
- **Email only comes from the merge.** The cluster does not send it for array jobs, so a killed task only shows up when the merge fails.
- **`e2`, `e3`, `e7` and `e7b` do not resume or split.** Only the decomposition's scripts do.
