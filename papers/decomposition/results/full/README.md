# Full-run summaries

What the paper's tables and figures are generated from, so `make_paper` works from a clean clone. Per-episode logs stay in `outputs/`, which is not tracked.

Each folder holds `summary.json` and the `manifest.json` that names its commit, simulator version and configuration. `tails.json` holds what the summaries leave out: per-cell medians, runaway counts and, for four cells, the shortfall per seed. It is recomputed from the logs by `make_paper --refresh`.

| Folder | What | Seeds |
|---|---|---|
| `e8_deep_cover`, `e8_low_cover`, `e8_horizons` | the main market | 1000-1019 |
| `e8_capital_deep`, `e8_capital_low`, `e8_capital_horizons` | ten times the capital for speculators and the noise pool | 1020-1039 |
| `protocol` | paired-seed protocol, both RNG schemes | 1000-1039 |
| `detection_curve`, `detection_curve_deep` | detection curve | 1000-1019 |
| `replications` | the three replication hooks | 1000-1019 |
| `runaway_causes` | switching causes of the runaways off after the arrival | 1000-1019 |
| `rl_heldout` | the sweep's two best PPO policies on selected and unseen seeds | see the file |
