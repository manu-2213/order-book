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
scheduling live outside the book (see `agents/market.py`).

Tests: `python -m unittest unit_tests -v`

## Background market (`agents/`, `simulation/`)

`agents/market.py`
- `Market` owns all accounts and routes orders. Resting bids reserve cash and
  resting asks reserve shares as running totals updated on place, fill (both
  sides), cancel, STP and eviction, so `free_cash`/`free_shares` are O(1), nobody
  shorts or borrows, and cash/shares are conserved. `market()` walks the ask
  levels so a buy is always funded. Analysis mode keeps a tape of
  `(step, price, qty, aggressor, buyer, seller)`; training mode keeps nothing.
- `PublicState` is updated once per step: mid, spread, best quotes, fast/slow
  EWMAs of mid, squared mid change (volatility) and volume, momentum
  (fast EMA - slow EMA) for three lookback pairs, rolling signed aggressive flow
  (ring buffer, no tape scan) with an EWMA imbalance -> `flow_signal` in [-1, 1],
  and an `attention` multiplier in [0.5, 3] from relative volatility and volume.

`agents/agents.py` -- cost per event, not per notional trader
- `NoiseFlow`: Poisson(rate x attention) orders per step from one pooled account;
  sides, sizes and geometric offsets drawn vectorised; 15% market (half size),
  5% marketable, the rest passive; Binomial random cancellations; bid probability
  tilted toward the pool's cash share so it never runs dry.
- `Speculators`: one array-backed population (40). Each member is a
  fundamentalist (noisy private view of a hidden random-walk value; passive
  limits for small gaps, size-bounded market orders for large ones) or a
  chartist (one of the three public momentum indicators; market entries, chunked
  exits). One vectorised draw picks who acts (5%/step); every 50 steps modes
  switch Brock-Hommes style on EWMA fitness of realised returns.
- `Institution` (x2): rare large metaorders executed as evenly sized TWAP child
  market orders; returns immediately while idle.
- `MarketMaker` (x4, two with flow detection): Avellaneda-Stoikov reservation
  price and spread from `PublicState` volatility, skew and half-spread capped at
  10 ticks, post-only, requotes only on a >= 1 tick move or when size runs down
  (keeps queue priority), pulls a side at +/-200 inventory. Detection skews and
  widens against persistent one-sided flow.

Default endowments (shares; cash = shares x mid so cash and stock start equal):
makers 300, noise pool 2000, institutions 1000, speculators 200. Edit
`POPULATION`/`ENDOWMENT` in `simulation/simulator.py`.

Commands (from the repo root):

```sh
python -m simulation.benchmark            # steps/s + cProfile top 20
python -m simulation.run                  # 10k-step analysis run -> simulation/simulation.png
python -m simulation.animate              # video with the live book -> simulation/simulation.mp4
python -m simulation.batch                # 100 seeds x 10k steps in parallel -> price_paths.png/.npz
python -m simulation.env --seeds 8        # burn in and save the RL start-state pool
```

Performance (one core, 20-agent population, 20k steps): 950 steps/s before
this rewrite -> ~17,000 steps/s in training mode, ~16,000 with the full tape.
Remaining cost is roughly 40% market makers, 30% noise flow, 25% book.

## RL environment (`simulation/env.py`)

`OrderBookEnv` is a Gymnasium env: the agent acts every `decision_interval`
steps (6 discrete actions: hold, market buy/sell, join best bid/ask, cancel
all), reward is the mark-to-market wealth change in ticks, episodes start from a
saved pool of burned-in simulator states (`burn_in(seeds, steps)`, several
seeds) reseeded per episode. The env is picklable and seeded through
`reset(seed=...)`, so `gymnasium.vector.AsyncVectorEnv` works. The observation
is `PublicState` features, own position/resting size/free cash and the top
`depth` volumes per side. If this is ever too slow, the next step is a numba
array book; it has not been needed.
