import random
import unittest

import numpy as np

from order_book import OrderBook


class TestOrderBook(unittest.TestCase):
    def setUp(self):
        self.book = OrderBook(20, 10000, 5)

    def assert_accounted(self, report):
        self.assertEqual(
            report["requested_quantity"],
            report["filled_quantity"] + report["resting_quantity"] + report["cancelled_quantity"],
        )

    def assert_invariants(self, book):
        orders = []
        for side, table in (("bid", book.bid_table), ("ask", book.ask_table)):
            window = book.bid_prices if side == "bid" else book.ask_prices
            for price, queue in table.items():
                self.assertTrue(queue)
                self.assertLessEqual(min(window), price)
                self.assertLessEqual(price, max(window))
                ids = [o["order_id"] for o in queue]
                self.assertEqual(ids, sorted(ids))
                for order in queue:
                    self.assertGreater(order["quantity"], 0)
                    self.assertEqual(book.get_order(order["order_id"]), order)
                    orders.append(order)
        self.assertEqual(len(orders), len(book._orders))
        self.assertEqual({o["order_id"] for o in orders}, set(book._orders))
        for side in ("bid", "ask"):
            table = book.bid_table if side == "bid" else book.ask_table
            self.assertEqual(sorted(table), book._prices[side])
            for price, queue in table.items():
                self.assertEqual(book.get_volume_at_price(side, price), sum(o["quantity"] for o in queue))
        by_client = {}
        for o in orders:
            by_client.setdefault(o["client_id"], set()).add(o["order_id"])
        for cid, ids in by_client.items():
            self.assertEqual({o["order_id"] for o in book.get_client_orders(cid)}, ids)
        if book.best_bid is not None and book.best_ask is not None:
            self.assertLess(book.best_bid, book.best_ask)
            self.assertEqual(book.mid_price, (book.best_bid + book.best_ask) / 2)
        else:
            self.assertIsNone(book.mid_price)
            self.assertIsNone(book.spread)
        self.assert_accounted(book.last_order_report)

    def test_empty_and_one_sided_state(self):
        self.assertIsNone(self.book.mid_price)
        self.assertIsNone(self.book.best_bid)
        self.assertIsNone(self.book.best_ask)
        np.testing.assert_array_equal(self.book.bid_prices, [9995, 9976])
        np.testing.assert_array_equal(self.book.ask_prices, [10005, 10024])
        bid = self.book.place_bid(1, 9990, 5)
        self.assertEqual(self.book.bid_prices[0], 9990)
        self.assertIsNone(self.book.mid_price)
        self.book.place_ask(2, 10010, 4)
        self.assertEqual(self.book.ask_prices[0], 10010)
        self.assertEqual(self.book.mid_price, 10000)
        self.book.cancel_order(bid, 1)
        self.assertIsNone(self.book.mid_price)
        self.assertIsNone(self.book.spread)

    def test_exact_window_boundaries_and_reported_evictions(self):
        for side in ("bid", "ask"):
            with self.subTest(side=side):
                book = OrderBook(20, 10000, 5)
                place = book.place_bid if side == "bid" else book.place_ask
                edge, outside, improved = ((9976, 9975, 9998) if side == "bid"
                                          else (10024, 10025, 10002))
                with self.assertRaises(ValueError):
                    place(1, outside, 1)
                old = place(1, edge, 7)
                place(2, improved, 3)
                self.assertIsNone(book.get_order(old))
                self.assertEqual(book.last_order_report["evicted_orders"][0]["order_id"], old)
                self.assertEqual(book.last_order_report["evicted_orders"][0]["quantity"], 7)
                self.assert_invariants(book)

    def test_all_admitted_window_ticks_appear_in_arrays(self):
        for side in ("bid", "ask"):
            with self.subTest(side=side):
                book = OrderBook(20, 10000, 5)
                place = book.place_bid if side == "bid" else book.place_ask
                anchor = 9995 if side == "bid" else 10005
                step = -1 if side == "bid" else 1
                for i in range(20):
                    place(i, anchor + i * step, i + 1)
                arrays = book.get_book_arrays()
                self.assertEqual(arrays[f"{side}_volumes"].tolist(), list(range(1, 21)))
                self.assertEqual(
                    arrays[f"{side}_prices"].tolist(), [anchor + i * step for i in range(20)],
                )
                self.assert_invariants(book)

    def test_empty_side_window_follows_opposite_best_without_fabricating_quote(self):
        first = self.book.place_ask(1, 10010, 2)
        self.book.place_ask(2, 10015, 4)
        np.testing.assert_array_equal(self.book.bid_prices, [10009, 9990])
        self.book.buy(3, 2)
        self.assertIsNone(self.book.get_order(first))
        self.assertEqual(self.book.ask_prices[0], 10015)
        np.testing.assert_array_equal(self.book.bid_prices, [10014, 9995])
        self.assertIsNone(self.book.mid_price)
        with self.assertRaises(ValueError):
            self.book.place_bid(3, 9994, 1)
        remaining = self.book.get_client_orders(2)[0]["order_id"]
        self.book.cancel_order(remaining, 2)
        self.assertIsNone(self.book.best_ask)
        # Both sides empty: both windows anchor on the last trade price.
        self.assertEqual(self.book.ask_prices[0], 10010)
        self.assertEqual(self.book.bid_prices[0], 10010)
        self.assertEqual(OrderBook(20, 10000, 5).ask_prices[0], 10005)  # no trade yet

    def test_default_stp_market_cancels_incoming_not_resting(self):
        order_id = self.book.place_ask(1, 10005, 5)
        report = self.book.buy(1, 3)
        self.assertEqual(report["filled_quantity"], 0)
        self.assertEqual(report["cancelled_quantity"], 3)
        self.assertEqual(self.book.get_order(order_id)["quantity"], 5)
        self.assert_invariants(self.book)

    def test_price_priority_fifo_partial_and_market_ioc_both_sides(self):
        for buying in (True, False):
            with self.subTest(buying=buying):
                book = OrderBook(20, 10000, 5)
                place = book.place_ask if buying else book.place_bid
                market = book.buy if buying else book.sell
                best, worse = (10005, 10008) if buying else (9995, 9992)
                deep = place(1, worse, 4)
                first = place(2, best, 3)
                second = place(3, best, 5)
                report = market(9, 6)
                key = "sell_order_id" if buying else "buy_order_id"
                self.assertEqual([t[key] for t in report["trades"]], [first, second])
                self.assertEqual([t["quantity"] for t in report["trades"]], [3, 3])
                self.assertEqual(report["avg_fill_price"], best)
                self.assertIsNone(book.get_order(first))
                self.assertEqual(book.get_order(second)["quantity"], 2)
                report = market(9, 10)
                self.assertEqual([t[key] for t in report["trades"]], [second, deep])
                self.assertEqual(report["filled_quantity"], 6)
                self.assertEqual(report["cancelled_quantity"], 4)
                self.assertAlmostEqual(report["avg_fill_price"], (2 * best + 4 * worse) / 6)
                self.assertEqual(report["trades"][0]["buyer_id"], 9 if buying else 3)
                self.assertEqual(report["trades"][0]["seller_id"], 3 if buying else 9)
                self.assertEqual(market(9, 2)["cancelled_quantity"], 2)
                self.assert_invariants(book)

    def test_crossing_limits_keep_price_protection_and_rest_both_sides(self):
        for buying in (True, False):
            with self.subTest(buying=buying):
                book = OrderBook(20, 10000, 5)
                maker = book.place_ask if buying else book.place_bid
                taker = book.place_bid if buying else book.place_ask
                prices = (10005, 10007, 10010) if buying else (9995, 9993, 9990)
                for client, price in enumerate(prices):
                    maker(client, price, 10)
                limit = 10008 if buying else 9992
                order_id = taker(9, limit, 25)
                report = book.last_order_report
                self.assertEqual([t["price"] for t in report["trades"]], list(prices[:2]))
                self.assertEqual(report["filled_quantity"], 20)
                self.assertEqual(book.get_order(order_id)["quantity"], 5)
                self.assertEqual(book.get_order(order_id)["price"], limit)
                self.assert_invariants(book)

    def test_all_stp_modes_after_external_partial_fill_both_sides(self):
        for buying in (True, False):
            for mode in sorted(OrderBook.VALID_STP_MODES):
                with self.subTest(buying=buying, mode=mode):
                    book = OrderBook(20, 10000, 5, mode)
                    maker = book.place_ask if buying else book.place_bid
                    taker = book.place_bid if buying else book.place_ask
                    price = 10005 if buying else 9995
                    maker(2, price, 2)
                    own = maker(1, price, 3)
                    maker(3, price, 4)
                    taker(1, price, 8)
                    report = book.last_order_report
                    expected = {
                        "none": (8, 0, 0), "cancel_newest": (2, 0, 6),
                        "cancel_oldest": (6, 2, 0), "cancel_both": (2, 0, 6),
                    }[mode]
                    self.assertEqual(
                        tuple(report[k] for k in ("filled_quantity", "resting_quantity", "cancelled_quantity")),
                        expected,
                    )
                    self.assertEqual(
                        [o["order_id"] for o in report["stp_cancelled_orders"]],
                        [own] if mode in {"cancel_oldest", "cancel_both"} else [],
                    )
                    self.assertEqual(book.get_order(own) is not None, mode == "cancel_newest")
                    if mode != "none":
                        self.assertTrue(all(t["buyer_id"] != t["seller_id"] for t in book.trades))
                    self.assert_invariants(book)

    def test_cancel_modify_priority_and_crossing(self):
        first = self.book.place_bid(1, 9995, 10)
        second = self.book.place_bid(2, 9995, 4)
        self.assertEqual(self.book.modify_order(first, 5, client_id=1), first)
        self.assertEqual(self.book.last_order_report["cancelled_quantity"], 5)
        increased = self.book.modify_order(first, 8, client_id=1)
        self.assertEqual([o["order_id"] for o in self.book.bid_table[9995]], [second, increased])
        self.assertEqual(self.book.last_order_report["replaced_order_id"], first)
        self.book.place_ask(3, 10005, 3)
        crossed = self.book.modify_order(increased, new_price=10005, client_id=1)
        self.assertEqual(self.book.last_order_report["filled_quantity"], 3)
        self.assertEqual(self.book.get_order(crossed)["quantity"], 5)
        self.assertIsNone(self.book.modify_order(crossed, 0, client_id=1))
        self.assertIsNone(self.book.get_order(crossed))
        self.assert_invariants(self.book)

    def test_rejected_modifications_are_atomic(self):
        for side in ("bid", "ask"):
            book = OrderBook(20, 10000, 5)
            place = book.place_bid if side == "bid" else book.place_ask
            order_id = place(1, 9995 if side == "bid" else 10005, 10)
            before, next_id = book.get_order(order_id), book.order_id
            changes = [
                {"new_quantity": -1}, {"new_quantity": 1.9}, {"new_quantity": True},
                {"new_price": 9990.5}, {"new_price": 0},
                {"new_price": 9975 if side == "bid" else 10025},
            ]
            for change in changes:
                with self.subTest(side=side, change=change):
                    with self.assertRaises(ValueError):
                        book.modify_order(order_id, client_id=1, **change)
                    self.assertEqual(book.get_order(order_id), before)
                    self.assertEqual(book.order_id, next_id)
            for operation in (book.cancel_order, book.modify_order):
                with self.assertRaises(PermissionError):
                    operation(order_id, client_id=2)
                with self.assertRaises(KeyError):
                    operation(0, client_id=1)

    def test_invalid_inputs_and_numpy_integers(self):
        for value in (0, -1, True, 1.5, "2", 2**63):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    OrderBook(value, 10000, 5)
                with self.assertRaises(ValueError):
                    self.book.buy(1, value)
                with self.assertRaises(ValueError):
                    self.book.place_bid(1, value, 1)
        with self.assertRaises(ValueError):
            OrderBook(20, 5, 5)
        with self.assertRaises(ValueError):
            self.book.place_bid(1, None, 1)
        with self.assertRaises(ValueError):
            self.book.buy(1, 1, "invalid")
        order_id = self.book.place_bid(np.int64(1), np.int64(9995), np.int64(3))
        self.assertEqual(self.book.get_order(order_id)["quantity"], 3)

    def test_snapshots_arrays_depth_and_mutation_isolation(self):
        bid = self.book.place_bid(1, 9995, 5)
        self.book.place_bid(2, 9993, 7)
        ask = self.book.place_ask(3, 10005, 4)
        snapshot = self.book.get_snapshot(1)
        self.assertEqual(snapshot["bids"], [(9995, 5, 1)])
        self.assertEqual(snapshot["total_bid_volume"], 12)
        self.assertEqual(self.book.get_snapshot(0)["bids"], [])
        arrays = self.book.get_book_arrays()
        self.assertEqual(arrays["bid_prices"].shape, (20,))
        self.assertEqual(arrays["bid_prices"].dtype, np.int64)
        self.assertEqual(arrays["bid_volumes"][:3].tolist(), [5, 0, 7])
        self.assertEqual(self.book.get_book_arrays(0)["ask_prices"].size, 0)
        for invalid in (-1, 2.5, True):
            for query in (self.book.get_snapshot, self.book.get_book_arrays):
                with self.assertRaises(ValueError):
                    query(invalid)
        self.book.bid_table[9995][0]["quantity"] = 0
        self.book.get_order(bid)["quantity"] = 0
        self.assertEqual(self.book.get_order(bid)["quantity"], 5)
        # Reports and trade records are shared, read-only objects (no deep copies).
        result = self.book.buy(4, 2)
        self.assertIs(result, self.book.last_order_report)
        self.assertEqual(result["trades"][0]["price"], 10005)
        self.assertEqual(result["trades"][0]["sell_order_id"], ask)
        self.book.trades.clear()
        self.assertEqual(len(self.book.trades), 1)
        self.assertEqual(len(OrderBook(20, 10000, 5, keep_trades=False).trades), 0)

    def test_low_prices_and_aggregate_volume_overflow(self):
        book = OrderBook(5, 3, 1)
        book.place_bid(1, 1, 1)
        arrays = book.get_book_arrays()
        self.assertEqual(arrays["bid_prices"].tolist(), [1, 0, 0, 0, 0])
        self.assertEqual(arrays["bid_valid"].tolist(), [True, False, False, False, False])
        book.place_ask(2, 4, 2**63 - 1)
        book.place_ask(3, 4, 1)
        with self.assertRaises(OverflowError):
            book.get_book_arrays()
        self.assertEqual(book.get_snapshot()["total_ask_volume"], 2**63)

    def test_deterministic_random_sequences_preserve_invariants(self):
        rng = random.Random(42)
        for _ in range(600):
            action, client, quantity = rng.randrange(6), rng.randrange(5), rng.randrange(1, 20)
            if action < 2:
                side = "bid" if action == 0 else "ask"
                anchor = int((self.book.bid_prices if side == "bid" else self.book.ask_prices)[0])
                price = anchor + rng.randrange(0, 5) * (1 if side == "bid" else -1)
                place = self.book.place_bid if side == "bid" else self.book.place_ask
                place(client, price, quantity)
            elif action < 4:
                market = self.book.buy if action == 2 else self.book.sell
                market(client, quantity)
            elif self.book._orders:
                order = rng.choice(list(self.book._orders.values()))
                if action == 4:
                    self.book.cancel_order(order.order_id, order.client_id)
                else:
                    self.book.modify_order(order.order_id, quantity, client_id=order.client_id)
            else:
                continue
            self.assert_invariants(self.book)


if __name__ == "__main__":
    unittest.main()


class TestMarketAndSimulator(unittest.TestCase):
    """Reservation bookkeeping must match the book exactly under every removal path."""

    def assert_reservations_consistent(self, market):
        reserved = {cid: [0, 0] for cid in market.accounts}
        for side, table in (("bid", market.book.bid_table), ("ask", market.book.ask_table)):
            for queue in table.values():
                for o in queue:
                    reserved[o["client_id"]][side == "ask"] += o["price"] * o["quantity"] if side == "bid" else o["quantity"]
        for cid, account in market.accounts.items():
            self.assertEqual((account.reserved_cash, account.reserved_shares), tuple(reserved[cid]), cid)
            self.assertGreaterEqual(account.free_cash, 0)
            self.assertGreaterEqual(account.free_shares, 0)

    def test_settlement_reservations_and_removal_paths(self):
        from agents import Market
        book = OrderBook(5, 100, 1, self_trade_prevention="cancel_oldest")
        market = Market(book)
        for cid in range(3):
            market.open_account(cid, 10_000, 50)
        self.assertIsNotNone(market.limit(0, "bid", 99, 10))  # reserves 990
        self.assertEqual(market.accounts[0].reserved_cash, 990)
        self.assertIsNotNone(market.limit(1, "ask", 101, 5))  # reserves 5 shares
        market.market(2, "ask", 4)  # hits client 0's bid: maker cash reservation released pro rata
        self.assertEqual(market.accounts[0].reserved_cash, 99 * 6)
        self.assertEqual((market.accounts[0].shares, market.accounts[2].cash), (54, 10_000 + 4 * 99))
        market.limit(0, "bid", 95, 2)
        market.limit(0, "bid", 100, 1)  # improving the best evicts the 95 bid (window of 5)
        self.assertEqual(book.get_volume_at_price("bid", 95), 0)
        market.market(0, "ask", 3)  # own bid at best: cancel_oldest removes it and releases cash
        self.assert_reservations_consistent(market)
        market.cancel_all(0)
        market.cancel_all(1)
        self.assertEqual((market.accounts[0].reserved_cash, market.accounts[1].reserved_shares), (0, 0))
        self.assertIsNone(market.limit(2, "ask", 200, 1))  # outside the window: dropped, counted
        self.assertEqual(market.window_rejects["ask"], 1)
        market.limit(1, "ask", 101, 50)
        self.assertEqual(market.accounts[1].reserved_shares, 50)
        self.assertIsNone(market.limit(1, "ask", 102, 1))  # nothing free left
        self.assertEqual(market.budget_rejects["ask"], 1)
        market.accounts[2].cash = 250
        report = market.market(2, "bid", 10)  # can afford only 2 at 101
        self.assertEqual(report["filled_quantity"], 2)
        self.assert_reservations_consistent(market)

    def test_simulator_conserves_wealth_is_seeded_and_picklable(self):
        import pickle
        from simulation.simulator import Simulator
        a, b = Simulator(seed=3, analysis=False), Simulator(seed=3, analysis=False)
        a.run(300)
        b.run(300)
        self.assertEqual(a.market.public.mid, b.market.public.mid)
        self.assertEqual(a.book.trade_id, b.book.trade_id)
        a.check_conservation()
        self.assert_reservations_consistent(a.market)
        self.assertIsNone(a.market.tape)
        clone = pickle.loads(pickle.dumps(a))
        clone.run(200)
        a.run(200)
        self.assertEqual(clone.market.public.mid, a.market.public.mid)
        clone.reseed(7)
        a.reseed(8)
        clone.run(200)
        a.run(200)
        self.assertNotEqual(clone.book.order_id, a.book.order_id)
        clone.check_conservation()

    def test_env_is_seeded_bounded_and_picklable(self):
        import pickle
        from simulation.env import OrderBookEnv, burn_in
        states = burn_in([0, 1], steps=200)
        env = OrderBookEnv(states, decision_interval=5, horizon=3)
        obs, _ = env.reset(seed=4)
        self.assertEqual(obs.shape, env.observation_space.shape)
        results = [env.step(a) for a in (1, 3, 4)]
        trajectory = [r[1] for r in results]
        self.assertEqual([r[3] for r in results], [False, False, True])  # truncated at the horizon
        clone = pickle.loads(pickle.dumps(OrderBookEnv(states, decision_interval=5, horizon=3)))
        clone.reset(seed=4)
        self.assertEqual([clone.step(a)[1] for a in (1, 3, 4)], trajectory)
        env.sim.check_conservation()
        self.assert_reservations_consistent(env.sim.market)
