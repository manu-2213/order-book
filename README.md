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
