import unittest
import numpy as np
from order_book import OrderBook


class TestOrderBook(unittest.TestCase):
    def setUp(self):
        self.ob = OrderBook(num_levels=20, mid_price=10000, distance_from_mid=5)

    def test_initial_state(self):
        np.testing.assert_array_equal(self.ob.bid_prices, np.array([9995, 9975]))
        np.testing.assert_array_equal(self.ob.ask_prices, np.array([10005, 10025]))
        self.assertEqual(self.ob.mid_price, 10000)
        self.assertIsNone(self.ob.best_bid)
        self.assertIsNone(self.ob.best_ask)
        self.assertIsNone(self.ob.spread)

    def test_resting_limit_orders_and_out_of_bounds(self):
        oid1 = self.ob.place_ask(client_id=1, ask=10022, quantity=100)
        self.assertEqual(oid1, 100_000)
        self.assertEqual(self.ob.ask_table[10022][0]["order_id"], 100_000)
        self.assertEqual(self.ob.ask_table[10022][0]["quantity"], 100)

        oid2 = self.ob.place_bid(client_id=2, bid=9990, quantity=50)
        self.assertEqual(oid2, 100_001)
        self.assertEqual(self.ob.bid_table[9990][0]["quantity"], 50)

        # Out-of-range bid (< 9975) and ask (>= 10025)
        with self.assertRaises(ValueError):
            self.ob.place_bid(client_id=2, bid=9970, quantity=10)
        with self.assertRaises(ValueError):
            self.ob.place_ask(client_id=1, ask=10025, quantity=10)

    def test_window_shift_and_eviction_on_aggressive_non_crossing_orders(self):
        # Place deep bid at 9976 (within [9995, 9975])
        deep_bid_id = self.ob.place_bid(client_id=1, bid=9976, quantity=25)
        self.assertIsNotNone(self.ob.get_order(deep_bid_id))

        # Place tighter bid at 9998 -> shifts bid_prices to [9998, 9978], evicting 9976
        tight_bid_id = self.ob.place_bid(client_id=2, bid=9998, quantity=40)
        np.testing.assert_array_equal(self.ob.bid_prices, np.array([9998, 9978]))
        self.assertNotIn(9976, self.ob.bid_table)
        self.assertIsNone(self.ob.get_order(deep_bid_id))
        self.assertIsNotNone(self.ob.get_order(tight_bid_id))
        self.assertEqual(self.ob.mid_price, (10005 + 9998) / 2)

        # Same on ask side: place deep ask at 10024, then tighter ask at 10002
        deep_ask_id = self.ob.place_ask(client_id=3, ask=10024, quantity=30)
        tight_ask_id = self.ob.place_ask(client_id=4, ask=10002, quantity=60)
        np.testing.assert_array_equal(self.ob.ask_prices, np.array([10002, 10022]))
        self.assertNotIn(10024, self.ob.ask_table)
        self.assertIsNone(self.ob.get_order(deep_ask_id))
        self.assertIsNotNone(self.ob.get_order(tight_ask_id))
        self.assertEqual(self.ob.mid_price, (10002 + 9998) / 2)

    def test_price_time_priority_and_market_orders(self):
        # Two asks at 10005 (FIFO order) and one ask at 10008
        a1 = self.ob.place_ask(client_id=10, ask=10005, quantity=30)
        a2 = self.ob.place_ask(client_id=11, ask=10005, quantity=50)
        a3 = self.ob.place_ask(client_id=12, ask=10008, quantity=40)

        # Market buy for 60: should fill all 30 of a1 and 30 of a2 at 10005
        report = self.ob.buy(client_id=20, quantity=60)
        self.assertEqual(report["filled_quantity"], 60)
        self.assertEqual(report["cancelled_quantity"], 0)
        self.assertEqual(report["avg_fill_price"], 10005.0)
        self.assertEqual(len(report["trades"]), 2)
        self.assertEqual(report["trades"][0]["sell_order_id"], a1)
        self.assertEqual(report["trades"][0]["quantity"], 30)
        self.assertEqual(report["trades"][1]["sell_order_id"], a2)
        self.assertEqual(report["trades"][1]["quantity"], 30)

        # a2 should have 20 left at 10005
        self.assertIsNone(self.ob.get_order(a1))
        self.assertEqual(self.ob.get_order(a2)["quantity"], 20)

        # Market buy for 100 (exceeds remaining 20 + 40 = 60 liquidity -> partial fill + IOC cancel)
        report2 = self.ob.buy(client_id=21, quantity=100)
        self.assertEqual(report2["filled_quantity"], 60)
        self.assertEqual(report2["cancelled_quantity"], 40)
        self.assertIsNone(self.ob.best_ask)
        self.assertIsNone(self.ob.get_order(a3))

    def test_crossing_limit_order_matches_and_rests_remainder(self):
        self.ob.place_ask(client_id=1, ask=10005, quantity=40)
        self.ob.place_ask(client_id=2, ask=10007, quantity=30)
        self.ob.place_ask(client_id=3, ask=10010, quantity=50)

        # Crossing limit bid at 10008 for quantity 90:
        # Matches 40 @ 10005 and 30 @ 10007 (total 70), then rests 20 @ 10008!
        bid_oid = self.ob.place_bid(client_id=9, bid=10008, quantity=90)
        report = self.ob.last_order_report
        self.assertEqual(report["order_id"], bid_oid)
        self.assertEqual(report["filled_quantity"], 70)
        self.assertEqual(report["resting_quantity"], 20)
        self.assertEqual(self.ob.best_bid, 10008)
        self.assertEqual(self.ob.best_ask, 10010)
        self.assertEqual(self.ob.get_volume_at_price("bid", 10008), 20)

    def test_cancel_and_modify_order(self):
        b1 = self.ob.place_bid(client_id=1, bid=9995, quantity=50)
        b2 = self.ob.place_bid(client_id=2, bid=9995, quantity=40)

        # Decreasing quantity keeps queue priority (b1 stays ahead of b2)
        mod_id = self.ob.modify_order(order_id=b1, new_quantity=30, client_id=1)
        self.assertEqual(mod_id, b1)
        self.assertEqual(self.ob.bid_table[9995][0]["order_id"], b1)
        self.assertEqual(self.ob.bid_table[9995][0]["quantity"], 30)

        # Increasing quantity loses time priority (moves behind b2)
        new_b1 = self.ob.modify_order(order_id=b1, new_quantity=80, client_id=1)
        self.assertNotEqual(new_b1, b1)
        self.assertEqual(self.ob.bid_table[9995][0]["order_id"], b2)
        self.assertEqual(self.ob.bid_table[9995][1]["order_id"], new_b1)

        # Ownership check on cancel
        with self.assertRaises(PermissionError):
            self.ob.cancel_order(order_id=b2, client_id=999)

        cancelled = self.ob.cancel_order(order_id=b2, client_id=2)
        self.assertEqual(cancelled["cancelled_quantity"], 40)
        self.assertIsNone(self.ob.get_order(b2))

    def test_self_trade_prevention(self):
        # Default STP is "cancel_newest"
        a1 = self.ob.place_ask(client_id=42, ask=10005, quantity=50)
        report = self.ob.buy(client_id=42, quantity=30)
        self.assertEqual(report["filled_quantity"], 0)
        self.assertEqual(report["cancelled_quantity"], 30)
        self.assertEqual(self.ob.get_order(a1)["quantity"], 50)

        # "cancel_oldest" cancels resting order and continues matching
        ob_oldest = OrderBook(
            num_levels=20,
            mid_price=10000,
            distance_from_mid=5,
            self_trade_prevention="cancel_oldest",
        )
        old_id = ob_oldest.place_ask(client_id=42, ask=10005, quantity=50)
        other_id = ob_oldest.place_ask(client_id=99, ask=10006, quantity=50)
        rep2 = ob_oldest.buy(client_id=42, quantity=30)
        self.assertIn(old_id, rep2["stp_cancelled_order_ids"])
        self.assertIsNone(ob_oldest.get_order(old_id))
        self.assertEqual(rep2["filled_quantity"], 30)
        self.assertEqual(rep2["trades"][0]["sell_order_id"], other_id)

    def test_snapshot_and_book_arrays(self):
        self.ob.place_bid(client_id=1, bid=9995, quantity=25)
        self.ob.place_bid(client_id=2, bid=9993, quantity=15)
        self.ob.place_ask(client_id=3, ask=10005, quantity=40)

        snap = self.ob.get_snapshot()
        self.assertEqual(snap["best_bid"], 9995)
        self.assertEqual(snap["best_ask"], 10005)
        self.assertEqual(snap["spread"], 10)
        self.assertEqual(snap["total_bid_volume"], 40)
        self.assertEqual(snap["total_ask_volume"], 40)

        arrays = self.ob.get_book_arrays()
        self.assertEqual(len(arrays["bid_volumes"]), 20)
        self.assertEqual(len(arrays["ask_volumes"]), 20)
        self.assertEqual(arrays["bid_volumes"][0], 25)
        self.assertEqual(arrays["bid_volumes"][2], 15)
        self.assertEqual(arrays["ask_volumes"][0], 40)


if __name__ == "__main__":
    unittest.main()
