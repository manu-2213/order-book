# Stealth execution

*Hiding in the Order Flow: Learning to Execute Against Liquidity Providers that Detect Metaorders.* Can an executor trade without
being detected, what does detection cost, and can a learned policy do better than classical schedules?

The harness is `order_book/experiments.py`; the studies are in `experiments.py` here; results go to `outputs/stealth_execution/`. The
methods and tables are still in the simulator reference (`docs/simulator.tex`, Parts III and IV) and have not been moved here.

E3 was regenerated after the fix to `Market.hide`, so `footprint` now has exactly zero leakage under A0 and everything else is unchanged.
E2, E7 and E7b are as first run: their stored results predate the fix, which matters only if an adaptive strategy is in them (`footprint`
is in E2's grid, but E2 runs visible only, so it does not change).

## Measurement without RL (plan sections 5.1, 6, 7, 9, 10)

- **Execution task** (`order_book/execution/task.py`): side, Q = fraction x expected volume x T (T = 2000, N = 5),
  hard limits (2 resting orders, 5 ticks through the best, optional participation cap), forced
  completion at T counted in full. `ExecutorAgent` is an external agent with its own RNG stream.
- **Baselines** (`order_book/execution/strategies.py`): immediate, twap_market, twap_passive, twap_random, pov,
  almgren_chriss_{1e-7,1e-6,1e-5}, obizhaeva_wang, limit_market, footprint, mimic. Model-based ones read
  a `Calibration` (sigma, temporary/permanent impact, resilience, depth, fill probabilities) fitted under
  A0 by `scripts/calibrate.py`.
- **Adversaries**: A0, A1 (lenient/medium/strict flow detector), A4 (A1 + anticipator) as markets; A1
  thresholds, A2 (runs, CUSUM, volume clock, regularity) and A3 (learned, `scripts/train_detector.py`)
  as observers scored on ground truth.
- **E2** (`scripts/e2_detectability.py`): ROC AUC / delay / false alarms of every detector on every
  classic strategy and size in the A1-medium market.
- **E3** (`scripts/e3_leakage.py`): every baseline x size x {A0, A1-medium, A4} x {visible, invisible}
  on the same seeds (+ no-executor runs); leakage = IS(visible) - IS(invisible) paired by seed with
  bootstrap CI, p-value and Holm correction; prints the Phase 2 gate verdict.
- **E7** (`scripts/e7_regimes.py`): the same counterfactual across a grid of market regimes
  (`regime(liquidity, institutions, makers, mm_size_scale)`) and executor sizes, mapping where leakage
  appears as liquidity and institutional cover fall and the order becomes a large share of flow (H4).

```sh
python -m scripts.calibrate                        # ~3 min; writes outputs/calibration.json
python -m scripts.train_detector                   # verbose per epoch; writes outputs/detectors/learned.npz
python -m papers.stealth_execution.scripts.e2_detectability --workers 8     # 420 episodes -> outputs/stealth_execution/e2/
python -m papers.stealth_execution.scripts.e3_leakage --workers 8 --paths   # 4,380 episodes -> outputs/stealth_execution/e3/ (gate verdict at the end)
python -m papers.stealth_execution.scripts.e7_regimes --workers 8           # leakage across market regimes -> outputs/stealth_execution/e7/
python -m papers.stealth_execution.scripts.e7_strength --workers 8          # detector-strength sweep in low-cover -> outputs/stealth_execution/e7b/
python -m scripts.<any> --quick                    # smoke test in seconds
```

## The RL executor (`order_book/execution/`)

The learned executor is built on the Phase 2 machinery: it is one more `Strategy` carried by the same
`ExecutorAgent`, with the same hard limits and forced completion, so it is scored by the same
experiments as the baselines.

- **MDP** (`execution/features.py`): a 13-feature, public-data-only observation (own progress, schedule
  gap, footprint share, imbalance, momentum, running shortfall, ...) and a 15-way `size x placement`
  action grid; the reward is telescoping implementation shortfall in bps (an episode's rewards sum to
  minus its quantity-weighted shortfall).
- **Environment** (`execution/env.py`): `ExecutionEnv.rollout` runs whole episodes in the E7b candidate
  market (`phase3_config`: liquid `low_cover`, two detecting makers); `PolicyStrategy`/`PolicyNet` are
  the actor-critic policy.
- **Training** (`execution/ppo.py`, `scripts/train_rl.py`): PyTorch PPO over whole episodes, verbose per
  iteration, with periodic greedy evaluation against TWAP.

```sh
python -m papers.stealth_execution.scripts.train_rl --iterations 300 --episodes 32 --wandb   # -> outputs/stealth_execution/rl/policy.pt, best_policy.pt, history.json
python -m papers.stealth_execution.scripts.sweep_rl --workers 6 --wandb --sync-existing      # parallel sweep, live stdout + W&B, resumable
python -m papers.stealth_execution.scripts.train_rl --quick                                  # smoke test, ~1 min
# score the trained policy against the baselines through the leakage counterfactual:
python -m papers.stealth_execution.scripts.e7_strength --strategies rl@outputs/stealth_execution/rl/policy.pt twap_market --workers 8
```

**Selection bias in the sweep results.** `train_rl` and `sweep_rl` evaluate on seeds 1000-1039 and pick checkpoints and
hyperparameters from those scores, so the gaps in the simulator reference are selected on their own test set. This is still how those
scripts work. On 80 unseen
seeds (`python -m papers.stealth_execution.scripts.rl_heldout`) PPO is behind TWAP: +0.39 bps [+0.03, +0.76] at Q/V 0.2, +0.21 bps [-0.00, +0.44] at 0.4.
Retrain with selection on seeds below 1000 before quoting a gain.
