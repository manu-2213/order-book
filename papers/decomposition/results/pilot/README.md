# Pilot summaries

The `summary.json` of each pilot run, committed so the notes' tables rebuild from a clean clone. Per-episode logs stay in `outputs/`, which is
not tracked.

These runs predate manifests, so nothing here records its own provenance. All ran on the Mac on 2026-11-02, simulator version 1, on the working tree
just before the move into `papers/`.

| Folder | What | How |
|---|---|---|
| `e8_pilot` | `twap_market`, `twap_passive`; 5 sizes; deep cover; 20 seeds | `decompose --strategies twap_market twap_passive --sizes 0.02 0.05 0.1 0.2 0.4` |
| `e8_low_cover` | 12 baselines and 2 PPO policies; 2 sizes; low cover; 20 seeds | `decompose --regime low_cover --sizes 0.2 0.4` |
| `e8_rl_stateful` | TWAP and PPO, ladder only, stateful draws; low cover | `decompose --regime low_cover --rng stateful --mode ladder --no-controls` |
| `protocol` | 3 strategies, 2 sizes, 40 seeds, both schemes | `crn_variance` |
| `detection_curve` | low cover, 3 thresholds, 20 seeds | `detection_curve` |
| `replications` | all three hooks, deep cover, 20 seeds | `replications` |
| `rl_heldout` | the sweep's two best policies on selected and unseen seeds | `rl_heldout` (stealth_execution) |
