from .simple_orders import SimpleOrders



from typing import Union, List
from itertools import product
from gym.spaces import Discrete

from nextrade.orders import Order, OrderListener, TradeSide, TradeType


class SwapSimpleOrders(SimpleOrders):
    def compile(self):
        self.actions = list(product(self._criteria,
                                    self._trade_sizes,
                                    self._durations,
                                    [TradeSide.LONG_OPEN,TradeSide.LONG_CLOSE,
                                    TradeSide.SHORT_OPEN,TradeSide.SHORT_CLOSE]))
        self.actions = list(product(self.exchange_pairs, self.actions))
        self.actions = [None] + self.actions

        self._action_space = Discrete(len(self.actions))


    def get_order(self, action: int, portfolio: 'Portfolio') -> Order:

        return None
