# Running on the cluster

The same workflow as DeepHedging: sync with `rsync` from the Mac, one short Slurm file per run, a smoke test first, and `submit_all.sh` for the grid.
Each command says where it runs. **None of this has been run on the cluster yet**, so the first run below is also the test of these scripts.

The differences from DeepHedging: order_book is CPU-only and parallelises over processes, so jobs use the default `compute` partition with many cores, no GPU and no account. There
are no banks to build. Job files follow [the cluster's guide](https://docs.hpc.qmul.ac.uk/using/submittingjobs/): cores with `-n` and read back as `$SLURM_NTASKS`, not `-c`; memory with
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
bash papers/decomposition/slurm/submit_all.sh extras   # e8_horizons, replications
bash papers/decomposition/slurm/submit_all.sh all      # all of them
```

Each job uses `--workers $SLURM_NTASKS`, 32 by default; to change it, edit `-n` in the job file. Queue time depends mostly on cores and memory, not on runtime. The CPU hours come from the pilot's speed on the Mac: about 0.8 s per run in deep cover, 1.4 s in low cover with the learned
executors, and in proportion to the horizon.

| Job | Runs | CPU hours | Walltime asked |
|---|---|---|---|
| `e8_deep_cover` | 21,600 | 5-7 | 24 h |
| `e8_low_cover` | 28,000 | 11 | 24 h |
| `e8_horizons` | 18,000 | 7-10, mostly the 8,000-step horizon | 48 h |
| `protocol` | 2,400 | 1 | 12 h |
| `detection_curve` | 4,200 | 1 | 12 h |
| `replications` | 1,840 | 1 | 12 h |

These are estimates; correct them with the smoke test's speed. Very large orders in some seeds run slower than average.

**Decide before the final grid:** the default burn-in of 2,000 keeps results comparable with E3, but volume is still rising then. A longer burn-in means adding `--burn-in 5000` to
the job files, and the results are then not comparable with the pilot.

## Monitoring and results

```bash
squeue --me                                                          # cluster
tail -f papers/decomposition/slurm/logs/e8_deep_cover.o<id>          # cluster: progress and an ETA per run
jobstats                                                             # cluster: memory requested against used; lower --mem-per-cpu if far apart
scancel <id>                                                         # cluster
bash hpc/submit.sh decomposition e8_deep_cover                       # cluster: resume after a walltime kill or crash
bash hpc/pull.sh decomposition                                       # Mac: summaries and manifests
bash hpc/pull.sh decomposition --all                                 # Mac: also the per-episode logs
```

To build the notes from the full run, on the **Mac**, after pulling:

```bash
.venv/bin/python -m papers.decomposition.scripts.make_tables --root outputs/decomposition --deep e8_deep_cover --heldout outputs/stealth_execution/rl_heldout/summary.json
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
- **A resumed run's walltime is per job.** Re-submit as often as needed: results are flushed line by line.
- **`e2`, `e3`, `e7` and `e7b` do not resume.** Only the decomposition's scripts do.
