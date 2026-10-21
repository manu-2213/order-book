import numpy as np
from collections import defaultdict


class OrderBook:
    """
    Limit order book class. Values are stored as ints (100.05 becomes 10005).

    """
    def __init__(self, num_levels: int, mid_price: int, distance_from_mid: int):

        # Controls the number of levels (in ticks) of the order book. Orders falling
        # outside of this range are cancelled / rejected.

        self.num_levels = num_levels
        self.ask_prices = np.array([mid_price + distance_from_mid, mid_price + distance_from_mid + num_levels])
        self.bid_prices = np.array([mid_price - distance_from_mid, mid_price - distance_from_mid - num_levels])
        self.mid_price = mid_price
        self.ask_table = defaultdict(list)
        self.bid_table = defaultdict(list)

        self.order_id = 100_000 # identifies specific order and time

    def buy(self, client_id: int, ):
        pass

    def sell(self, client_id: int, ):
        pass

    def place_bid(self, client_id: int, bid: int, quantity: int,):

        if bid >= self.ask_prices[0]:
            return self.buy(client_id)

        elif bid < self.bid_prices[1]:
            raise ValueError(f"This bid order falls outside the limit of the order book",
                             f"The current mid price is: {self.mid_price}. This order boook",
                             f"has {self.num_levels} levels. With the highest bid sitting at",
                             f" {self.bid_prices[0] / 100} USD.")

        else:
            self.bid_table[bid].append({
                            "client_id": client_id,
                            "quantity": quantity,
                            "order_id": self.order_id
                        })
            
            self.order_id += 1

            if  bid > self.bid_prices[0]:
                new_bid_prices = np.array([bid, bid - self.num_levels])
                
                for b in range(self.bid_prices[1], self.bid_prices[0] + 1):
                    if b < new_bid_prices[1]:
                        if b in self.bid_table:
                            del self.bid_table[b]
    
                self.bid_prices = new_bid_prices

                self.mid_price = (self.ask_prices[0] + self.bid_prices[0]) / 2

        return self.order_id - 1
          
    def place_ask(self, client_id: int, ask: int, quantity: int, ):

        if ask <= self.bid_prices[0]:
            return self.sell(client_id)

        elif ask >= self.ask_prices[1]:
            raise ValueError(f"This ask order falls outside the limit of the order book",
                             f"The current mid price is: {self.mid_price}. This order boook",
                             f"has {self.num_levels} levels. With the lowest ask sitting at",
                             f" {self.ask_prices[0] / 100} USD.")

        else:
            self.ask_table[ask].append({
                            "client_id": client_id,
                            "quantity": quantity,
                            "order_id": self.order_id
                        })
            
            self.order_id += 1

            if ask < self.ask_prices[0]:
                new_ask_prices = np.array([ask, ask + self.num_levels])
                
                for b in range(self.ask_prices[0], self.ask_prices[1] + 1):
                    if b > new_ask_prices[1]:
                        if b in self.ask_table:
                            del self.ask_table[b]
    
                self.ask_prices = new_ask_prices

                self.mid_price = (self.ask_prices[0] + self.bid_prices[0]) / 2

            return self.order_id - 1

    