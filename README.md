# Order book

A deterministic, single-instrument matching engine. It generates no liquidity,
price process, mean reversion, or agent behavior. The caller serializes actions;
call order determines arrival priority. Prices and quantities are integer ticks
and lots. Currency conversion belongs outside the engine.

## Rules

- Best price first, FIFO within each price. Trades execute at the resting price.
- Limit orders match only at their limit or better; remaining size rests until
  filled, cancelled, or explicitly evicted by the bounded-window policy.
- Market orders consume available liquidity and cancel the remainder (IOC).
- Self-trade prevention defaults to cancelling the incoming remainder.
  `cancel_oldest` removes the encountered resting order and continues;
  `cancel_both` removes it and cancels the incoming remainder; `none` permits
  self-matching. Fills made before an STP encounter remain valid.
- Reducing remaining size retains priority. Increasing size or changing price
  replaces the order with a new ID and new priority. Invalid replacements leave
  the original intact. A remaining size of zero cancels.
- Cancellation/modification with `client_id` checks ownership. Omitting it is an
  administrative operation: an agent-facing adapter must supply the caller's ID.
- Inputs must be integers (NumPy integers supported, booleans rejected), with
  positive prices/quantities and nonnegative client IDs. Inputs are int64-bounded.
  Aggregated quantities use Python integers; array conversion raises explicitly
  if a volume exceeds int64.

## Bounded-window policy

This preserves the requested bounded book, but **is not a normal exchange rule**.
Each side has an anchor: its best price when populated; one tick inside the
opposite best when it is empty; the last trade price when both sides are empty
(the constructor's `mid_price +/- distance_from_mid` before any trade). For N
levels and anchor B/A, bids below `max(1, B-N+1)` and asks above
`min(INT64_MAX, A+N-1)` are rejected. More aggressive orders may cross the
opposite side. An improvement can evict deeper orders outside the new window;
the triggering report lists them in `evicted_orders`. Removing an order does not
restore evicted ones, and empty sides never manufacture quotes: `mid_price` and
`spread` are `None` unless both sides rest.

An agent can exploit eviction by improving a quote to remove others' deep
liquidity. Fast moves also strip the far side's depth, which amplifies them.

## API and RL state

```python
from order_book import OrderBook

book = OrderBook(num_levels=20, mid_price=10000, distance_from_mid=5)
ask_id = book.place_ask(client_id=1, ask=10005, quantity=10)
report = book.buy(client_id=2, quantity=4)        # or book.limit("bid", 2, 10005, 4)
assert report["filled_quantity"] == 4
assert book.get_order(ask_id)["quantity"] == 6
book.cancel_order(ask_id, client_id=1)
```

Everything hot is O(1): `best_bid`, `best_ask`, `mid_price`, `spread`,
`get_volume_at_price`, `get_client_orders` (per-client index), `get_order`.
`levels(side)` lists `(price, volume)` best first; `best_excluding(side, id)`
is everyone else's best quote. Reports name the maker order of every fill
(`buy_order_id` / `sell_order_id`), the orders removed by STP
(`stp_cancelled_orders`) and by eviction (`evicted_orders`), so an external
accountant can keep reservations exact. Reports and trade records are shared,
read-only objects, not copies; `keep_trades=False` disables the trade log.

An L2 array (`get_book_arrays`) is a partial observation: FIFO order, owners,
order ids and anchors also drive transitions. Balances, inventory, rewards and
scheduling live outside the book (see `order_book/market.py`).

Tests: `python -m pytest`

## Documentation

`docs/simulator.tex` (compiled: `docs/simulator.pdf`, 26 pages) is the complete, self-contained technical
and mathematical reference for the project. It includes:
1. **Part I — Conceptual Primer & Codebase Navigation**: a beginner-friendly introduction to limit order
   books, metaorders, implementation shortfall, mechanical market impact vs. information leakage, the
   exact file-by-file module map, internal data structures (`OrderedDict` FIFO queues, sorted price lists,
   running account reservations), and the 6-stage `Simulator.step()` execution pipeline.
2. **Part II — The Simulator**: exact mathematical equations for `PublicState`, the `SeedSequence`
   common-random-number tree, every background agent (`FundamentalValue`, `NoiseFlow`, `Speculators` with
   Brock–Hommes switching, `InstitutionalTrader`, Avellaneda–Stoikov `MarketMaker` with volatility
   fragility and detection response, `OrderAnticipator`), every detector (`FlowDetector`, `RunsDetector`,
   `CusumDetector`, `VolumeClockDetector`, `RegularityDetector`, and the 3-class `LearnedDetector`), and
   Phase 1 stylised-facts and square-root impact validation (`outputs/validation_default.json`).
3. **Part III — Phase 2 Measurement & Counterfactual Experiments**: mathematical definitions of all 12
   baseline strategies, A0 parameter calibration (`outputs/calibration.json`), Mann–Whitney ROC AUC and
   paired-bootstrap protocol, and full empirical tables for **E2** (`outputs/e2/summary.json`), **E3**
   (`outputs/e3/summary.json`), **E7** (`outputs/e7/summary.json`), and **E7b** (`outputs/e7b/summary.json`).
4. **Part IV — Phase 3 RL Executor & Hyperparameter Sweep**: the 13-feature sign-normalised observation
   space, 15-way `(size_mult, placement)` action grid, telescoping implementation-shortfall reward
   theorem, `PolicyNet` actor–critic architecture, GAE + clipped PPO trainer, parallel W&B sweep runner
   (`scripts/sweep_rl.py`), and the sweep results (`outputs/rl/sweep/summary.json`) showing the learned
   policy beating TWAP at both $Q/V = 0.20$ ($+2.40$ vs. $+2.82$ bps) and $Q/V = 0.40$ ($+4.74$ vs.
   $+5.12$ bps).

Build the PDF with `cd docs && latexmk -pdf simulator.tex`.

## Layout

```
order_book/              the library (pip install -e .)
  book.py                OrderBook: matching engine, knows nothing about agents
  market.py              Account, Market: accounting and order routing
  public_state.py        PublicState: the per-step public view everyone may read
  config.py              frozen dataclasses for every parameter (SimConfig)
  detectors/             Detector interface; FlowDetector (rule-based, level A1)
  agents/                NoiseFlow, Speculators, InstitutionalTrader (+ MetaorderLog labels),
                         MarketMaker, OrderAnticipator, FundamentalValue
  simulator.py           Simulator: one seeded run; per-agent RNG streams; snapshot/restore;
                         external agents for paired counterfactuals
  presets.py             adversary levels A0 / A1 (lenient, medium, strict) / A4; private valuations
  history.py             per-step recording of an analysis run
  validation.py          stylised facts, impact statistics, pass/fail table (plan section 4.2)
  impact.py              paired square-root-impact measurement with injected metaorders
  execution/             ExecutionTask, ExecutorAgent, Strategy; 12 baselines; Calibration + fitting;
                         Phase 3 RL executor (features, policy, env, ppo)
  detectors/             + features.py, statistical.py (A2: runs, cusum, volume_clock, regularity),
                         learned.py (A3: logistic classifier on public features)
  scoring.py             ROC AUC, detection delay, false alarms; bootstrap CI, paired diff, CVaR, Holm
  experiments.py         AdversarySpec, EpisodeSpec, E2/E3/E7/E7b configs, runner and summaries
  env.py                 OrderBookEnv: generic-trader Gymnasium prototype (execution uses execution/env.py)
scripts/                 CLI entry points, run as python -m scripts.<name>; write to outputs/
tests/                   pytest suite (python -m pytest)
docs/                    LaTeX documentation
```

Dependency direction is strictly downward: `book` <- `market`/`public_state`
<- `detectors` <- `agents` <- `simulator` <- `presets`/`history`/`validation`/`impact`/`execution`/`env`
<- `experiments` <- `scripts`. The book never imports anything above it; agents never touch the book except
through `Market`; detectors read only `PublicState`.

## Configuration

A run is fully determined by a `SimConfig` and a seed. Every agent parameter,
detector threshold, population count and endowment is a field of a frozen
dataclass in `order_book/config.py`; experiments differ only by config:

```python
from dataclasses import replace
from order_book import SimConfig, Simulator
from order_book.config import PopulationConfig, market_makers
from order_book.presets import adversary

cfg = replace(SimConfig(), population=replace(PopulationConfig(), institutions=8,
                                              market_makers=market_makers(n=6, detecting=3)))
sim = Simulator(seed=0, config=cfg, analysis=True)
sim.run(10_000)
sim.check_conservation()
labels = sim.metaorders.records        # ground-truth metaorders (analysis mode only)
a4 = Simulator(seed=0, config=adversary("A4", "strict"))
```

## The market in one paragraph each

- **Randomness.** One root `SeedSequence(seed)` spawns an independent child stream for the scheduler,
  the hidden value and every agent in fixed order. `(config, seed)` reproduces the event log exactly;
  `reseed()` re-derives every stream in place; `new_stream()` / `add_external()` give an executor or an
  injected metaorder its own stream outside the scheduler's permutation, so a run with and without it
  shares the background randomness (common random numbers) and their difference is the order's impact.
- **Market / PublicState.** Accounting with exact reservations and conservation; a per-step public view
  (mid, spread, fast/slow EWMAs of mid, variance and volume, three momentum indicators, rolling signed
  flow imbalance, attention). No interpretation lives here.
- **Detectors.** `update(public_state)` once per step; `signal` in [-1, 1] and `p_buy`/`p_sell`.
  `FlowDetector` is the rule-based level A1 (EWMA of the flow imbalance, thresholded).
- **NoiseFlow.** Poisson(rate x attention) zero-intelligence orders from one pooled account.
- **Speculators.** 40 fundamentalists / chartists with Brock-Hommes switching; shared hidden value or
  private adaptive valuations (`FundamentalConfig.private`).
- **InstitutionalTrader** (x4). Poisson metaorder starts, log-normal sizes (median 600 lots, cap
  3,000), participation 2-20% of market volume, five execution styles (TWAP market, TWAP passive with
  catch-up, randomised TWAP, POV, front-loaded Almgren-Chriss-like), inventory-reverting side choice,
  forced completion at the planned end, ground-truth labels in analysis mode.
- **MarketMaker** (x4, two reading the flow detector). Avellaneda-Stoikov quotes, post-only, queue
  keeping; fixed detection response (skew, widen, size cut) per unit signal; fragility (size falls
  with fast/slow volatility ratio, quotes pulled above a ratio of 3 for a cooldown).
- **OrderAnticipator** (A4 only). Trades with the detected flow within a position limit, unwinds when
  the signal fades; optional contrarian-first mode (van Kervel-Menkveld).

## Validation (plan section 4.2)

`python -m scripts.validate --seeds 5 --steps 30000` with the default configuration passes every
target: median spread 1 tick, no empty sides, |return autocorrelation| < 0.05 from lag 5
(max 0.044), kurtosis 12.7 -> 6.9 from 1 to 100 steps, positive decaying autocorrelation of absolute
returns and of trade signs, square-root impact exponent 0.69 at completion / 0.64 at peak on the paired
measurement (0.60 / 0.54 on 181 natural metaorders), partial impact reversion, zero budget rejects,
~15,000 steps/s. See the document for how the measurements are defined, what the validation found
and fixed (institutional inventory drift, measurement power), and the parameter sweep behind the
defaults. The private-valuation preset runs but still needs its own calibration.

Commands (from the repo root; outputs go to `outputs/`):

```sh
pip install -e .                          # or: uv pip install -e .
python -m pytest                          # 134 tests, ~8 s
python -m scripts.validate                # stylised facts + paired impact, 5 seeds x 10k steps
python -m scripts.validate --seeds 10 --steps 50000 --preset private
python -m scripts.impact --seeds 20       # paired square-root impact, saves paths
python -m scripts.benchmark               # steps/s + cProfile top 20
python -m scripts.run                     # 10k-step analysis run -> outputs/simulation.png
python -m scripts.animate                 # video with the live book -> outputs/simulation.mp4
python -m scripts.batch                   # 100 seeds x 10k steps in parallel -> outputs/price_paths.*
python -m scripts.burn_in --seeds 8       # RL start-state pool -> outputs/states.pkl
```

## Phase 2: measurement without RL (plan sections 5.1, 6, 7, 9, 10)

- **Execution task** (`execution/task.py`): side, Q = fraction x expected volume x T (T = 2000, N = 5),
  hard limits (2 resting orders, 5 ticks through the best, optional participation cap), forced
  completion at T counted in full. `ExecutorAgent` is an external agent with its own RNG stream.
- **Baselines** (`execution/strategies.py`): immediate, twap_market, twap_passive, twap_random, pov,
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
python -m scripts.e2_detectability --workers 8     # 420 episodes -> outputs/e2/
python -m scripts.e3_leakage --workers 8 --paths   # 4,380 episodes -> outputs/e3/ (gate verdict at the end)
python -m scripts.e7_regimes --workers 8           # leakage across market regimes -> outputs/e7/
python -m scripts.e7_strength --workers 8          # detector-strength sweep in low-cover -> outputs/e7b/
python -m scripts.<any> --quick                    # smoke test in seconds
```

## Phase 3: RL executor (`order_book/execution/`)

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
python -m scripts.train_rl --iterations 300 --episodes 32 --wandb   # -> outputs/rl/policy.pt, best_policy.pt, history.json
python -m scripts.sweep_rl --workers 6 --wandb --sync-existing      # parallel sweep, live stdout + W&B, resumable
python -m scripts.train_rl --quick                                  # smoke test, ~1 min
# score the trained policy against the baselines through the leakage counterfactual:
python -m scripts.e7_strength --strategies rl@outputs/rl/policy.pt twap_market --workers 8
```

## Generic-trader prototype (`order_book/env.py`)

`OrderBookEnv` is the Gymnasium prototype from before the research plan: 6 discrete actions, mark-to-
market reward, episodes from a bank of burned-in snapshots. It still exposes the flow detector's
signal in the observation, which the plan forbids for the executor; it is kept only so the snapshot
and vectorisation machinery stays tested, and is superseded for execution by
`order_book/execution/env.py` (the Phase 3 MDP above).
