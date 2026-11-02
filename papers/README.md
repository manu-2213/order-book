# Papers

One simulator, many studies. `order_book/` is the library; each folder here is one paper built on it.

```
papers/<name>/
  __init__.py       OUT = outputs/<name>, where its runs go
  README.md         the question, how to run it, what has been found
  experiments.py    its configs and summaries (may be split, as decomposition is)
  scripts/          entry points: python -m papers.<name>.scripts.<script>, each with --quick and --workers
  tests/            claims specific to this paper; python -m pytest collects them
  slurm/            arrays of 1-hour tasks with a merge job, smoke_test.slurm, submit_all.sh (see hpc/README.md)
  paper/            LaTeX, and tables generated from results; private, so not tracked by git
  results/          small summaries committed with the paper, so its tables rebuild from a clean clone
```

## The rules

1. **Papers import the library and never each other**, and the library never imports a paper. `tests/test_layout.py` enforces both.
   Once a second paper needs something, move it into `order_book/`.
2. **Every result says where it came from.** Call `write_manifest(out, cfg)` next to the results; it records the simulator version,
   commit, configuration and machine.
3. **The simulator's behaviour is versioned.** Anything that changes what a run produces fails `tests/test_simulator_version.py`
   until you bump `SIMULATOR_VERSION` and record the new fingerprint. A paper quotes the version it ran on, and manifests refuse to
   resume across versions, so a study can stay on a tag while the market changes for the next one.
4. **One runner.** Use `iter_jobs`, `load_results` and the shard helpers from `order_book/runner.py`; a paper supplies configs, summaries and tables.
5. **`--quick` never touches real results.** It writes to `<name>_quick`.
6. **Jobs ask for one hour**, because those start at once. Split a long grid by seed with `--shard`, and `--merge` to combine it. `tests/test_slurm_files.py` checks every job file.
7. **Seeds below 1000 are for tuning, from 1000 up for testing.** Nothing is selected on a seed it is scored on.

## Adding a paper

```sh
mkdir -p papers/my_study/{scripts,tests,slurm,paper,results}
touch papers/my_study/scripts/__init__.py papers/my_study/tests/__init__.py
printf '"""One line on the question."""\n\nfrom scripts import OUTPUTS\n\nOUT = OUTPUTS / "my_study"\n' > papers/my_study/__init__.py
cp papers/decomposition/slurm/protocol*.slurm papers/my_study/slurm/   # then rename them and edit the commands
```

Inputs the cluster needs from `outputs/`, like a trained policy, have to be listed in `hpc/rsync.filter`.

| Paper | Status |
|---|---|
| `stealth_execution` | Run; its RL gains do not survive unseen seeds, see its README |
| `decomposition` | Full run done; price runaways in some cells partly explained, see its README |
