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
Each side has an independent anchor: its actual best price when populated, or
its last anchor when empty. Initially the anchors are `mid_price +/- distance_from_mid`;
these are reference values, not quotes or liquidity.

For N levels and anchor B/A, bids below `max(1, B-N+1)` and asks above
`min(INT64_MAX, A+N-1)` are rejected. More aggressive orders are allowed to cross
the opposite side. After submission the anchor tracks the actual best, including
the first order. An improvement can cancel deeper orders outside the new window.
The triggering report lists those orders, their owners, and remaining quantities
in `evicted_orders`. Removing an order does not restore previously evicted orders.

An agent can exploit eviction by submitting an improved quote to remove other
agents' deep liquidity. For research without that artificial effect, use an
unbounded matching book and truncate only observations instead; that is a
different policy, not enabled here.

Corrections from the initial implementation: windows contain N ticks rather
than N+1, both edges are inclusive, and empty sides never manufacture opposite
quotes or impose a spread. `mid_price` and `spread` are `None` unless both sides
have resting orders.

## API and RL state

```python
from order_book import OrderBook

book = OrderBook(num_levels=20, mid_price=10000, distance_from_mid=5)
ask_id = book.place_ask(client_id=1, ask=10005, quantity=10)
report = book.buy(client_id=2, quantity=4)
assert report["filled_quantity"] == 4
assert book.get_order(ask_id)["quantity"] == 6
book.cancel_order(ask_id, client_id=1)
```

Limit submissions return an ID; market submissions return a report.
`last_order_report` covers the latest submission, cancellation, or modification.
Reports separate incoming fills/resting/cancelled size from resting orders
removed by STP or eviction. Replacements identify `replaced_order_id`; their
requested quantity is the newly requested remaining size. Report sides are
consistently `bid`/`ask`; trade aggressor sides are `buy`/`sell`.
Read outputs, including tables, trades, and reports, are detached copies.

- `get_snapshot(depth)` returns populated price levels `(price, volume, count)`,
  best first. Whole-book totals do not change with requested depth.
- `get_book_arrays(depth)` returns contiguous ticks, including zero-volume gaps,
  anchored at each side's reference anchor. Defaults to N ticks. Prices outside
  the positive int64 domain are padded with zero and have a false `*_valid` mask.
  Empty-side grids are references with zero volume, **not executable quotes**.
  Explicit depth zero produces empty outputs; invalid depths raise.
- `get_client_orders(client_id)` exposes an agent's outstanding orders.
- Trade history grows until the book is discarded; create a fresh book per
  episode. There is no implicit history truncation.

An L2 array is a **partial observation**, not the complete Markov engine state:
FIFO order, owner identities (which affect STP), outstanding order IDs, and
admission anchors also affect future transitions. The engine retains those
internally; it does not yet provide checkpoint export/restore. Queue-priority
information can be obtained from ordered table copies for diagnostics.
Keep balances, inventory, rewards, action scheduling, latency, fees, and episode
termination in the external environment. Do not value inventory at a synthetic
midpoint when a side is empty.

Run the tests using the project's Python environment:

```sh
python -m unittest unit_tests -v
```

## Agent simulation

- `agents/` — `Market` (accounting wrapper), `Agent`, `RandomAgent`, `MarketMaker`.
- `simulation/run.py` — `simulate()` loop and `plot()`.
- `simulation/animate.py` — renders `simulation/simulation.mp4`: trailing price and
  inventory charts plus the live order book (±25 ticks around mid), one frame per
  `--stride` steps (default 10 → 1000 frames, ~100 s to render). Uses the ffmpeg
  binary bundled with `imageio-ffmpeg`: `.venv/bin/python -m simulation.animate`.

`.venv/bin/python -m simulation.run` (from the repo root) runs 10,000 steps on a book with
`num_levels=20`, one `MarketMaker` (client 0) and six `RandomAgent`s (clients 1–6), then
writes `simulation/simulation.png` (price, spread/volume,
per-agent wealth and inventory, buyer→seller trade matrix, final depth).

- `Market` wraps the book: it routes all orders and settles cash/shares for **both**
  sides of every trade. All accounting is kept outside the book.
- No shorting or borrowing: resting bids reserve cash and resting asks reserve shares.
  Market buys are capped by walking the visible asks. Total cash and shares are
  asserted to be conserved every step.
- `RandomAgent`: zero-intelligence trader. It acts with probability `activity`; 15% of
  its orders are market orders and the rest are limit orders placed between −2 and +15
  ticks from the reference price (mid, else last trade); negative offsets cross the spread.
  Orders are cancelled after `ttl` steps.
- `MarketMaker`: each step it cancels its quotes and requotes both sides around the
  reference with a random half-spread and size, skewed against its inventory.
- Agent order is shuffled every step. Everything is seeded (`simulate(seed=...)`).

### Comparing 100 independent price paths

From the repository root:

```sh
.venv/bin/python -m simulation.batch
```

Runs 100 independent simulations of 10,000 timesteps with 20 levels per side,
using up to four worker processes. Each run has its own book, six random traders,
one market maker, and a different seed (including different initial trader
endowments and activity). This samples 100 outcomes, not all possible outcomes.
No book snapshots are retained in batch mode.

The output is `simulation/price_paths.png`: all last-trade paths overlaid,
with a median and empirical 5th-95th percentile band. This band describes these
sampled simulations, not a forecast confidence interval. Before the first trade,
price is missing rather than replaced with an artificial trade.
`simulation/price_paths.npz` stores `prices` with shape `(runs, steps)`, `seeds`,
`num_levels` and `tick_size` for further analysis.

Options: `--runs 100 --steps 10000 --levels 20 --seed 0 --workers 4`.
Use `--workers 1` for sequential execution.
