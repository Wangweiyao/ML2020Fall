import numpy as np
import copy
from typing import *

from nextrade.orders import TradeSide
from .trading_environment import TradingEnvironment

MAKER_FEE_RATE = 0.02 / 100
TAKER_FEE_RATE = 0.05 / 100
CONTRACT_FACE_VALUE = 100.
LEVERAGE_RATE = 125.
ADJUSTMENT_FACTOR = 0.3
EPS = 1e-6
DUMMY_SLIP_RATIO = 0.001
INF = 1e10


def apply_dummp_slippage(direction: TradeSide, price) -> float:
    slip_ratio = 1.
    if direction in [TradeSide.LONG_OPEN, TradeSide.SHORT_CLOSE]:
        slip_ratio = 1. + DUMMY_SLIP_RATIO
    elif direction in [TradeSide.LONG_CLOSE, TradeSide.SHORT_OPEN]:
        slip_ratio = 1. - DUMMY_SLIP_RATIO
    return slip_ratio * price


class SwapTradingEnvironment(object):

    def __init__(self, env_config: Dict, **kwargs):
        self.start_time = env_config['start_time']
        self.end_time = env_config['end_time']
        self.data = copy.deepcopy(env_config['data'][self.start_time:self.end_time])
        self.preprocess = env_config['preprocess']        
        self.feed = self.preprocess(self.data)
        self.data_len = len(self.feed.inputs[0]._array) # get length of data by query the first node which is a stream

        self._swap_compile(env_config)
        self.env_config = env_config
        self.history = {'price': np.zeros(self.data_len-1), \
                        'equity': np.zeros(self.data_len-1), \
                        'long_position': np.zeros(self.data_len-1), \
                        'short_position': np.zeros(self.data_len-1), \
                        'margin_ratio': np.zeros(self.data_len-1)}
        self.step_counter = 0

    def _swap_compile(self, env_config):
        self._leverage_ratio = env_config.get('leverage_ratio', LEVERAGE_RATE)
        self._contract_face_value = env_config.get('contract_face_value', CONTRACT_FACE_VALUE)
        self._adjustment_factor = env_config.get('adjustment_factor', ADJUSTMENT_FACTOR)
        self._maker_fee_rate = env_config.get('maker_fee_rate', MAKER_FEE_RATE)
        self._taker_fee_rate = env_config.get('taker_fee_rate', TAKER_FEE_RATE)
        self._reset_account(env_config)

    def _reset_account(self, env_config={}):
        # Number of shares of long contracts.
        self._btc_swap_long_position = env_config.get('btc_swap_long_position', 0.)
        # Number of shares of short contracts
        self._btc_swap_short_position = env_config.get('btc_swap_short_position', 0.)
        # Equivalent quantities of long BTC. Necessary to compute average long position price.
        self._btc_long_quantity = env_config.get('btc_long_quantity', 0.)
        # Equivalent quantities of short BTC. Necessary to compute average short position price.
        self._btc_short_quantity = env_config.get('btc_short_quantity', 0.)
        # Average long position price.
        self._btc_long_position_price = env_config.get('btc_long_position_price', INF)
        # Average short position price.
        self._btc_short_position_price = env_config.get('btc_short_position_price', -INF)
        # Quantities of deposit BTC and realized BTC.
        self._btc = env_config.get('btc', 0.)
        # Margin ratio of account
        self._margin_ratio = env_config.get('margin_ratio', INF)

    @property
    def _margin_used(self):  # measured in BTC, without leverage
        return self._margin_used_at_price(self._close)

    @property
    def _swap_buying_power(self) -> int:  # measured in number of contracts
        return (max(self._equity - self._margin_used,
                    0.)) * self._leverage_ratio * self._close // self._contract_face_value

    @property
    def _equity(self):  # measured in BTC
        return (self._btc + self._unrealized_btc)

    @property
    def _equity_usd(self):  # measured in USD
        return self._equity * self._close

    @property
    def _unrealized_btc(self):
        return self._long_unrealized_btc + self._short_unrealized_btc

    @property
    def _long_unrealized_btc(self):
        return self._long_unrealized_btc_at_price(self._close)

    @property
    def _short_unrealized_btc(self):
        return self._short_unrealized_btc_at_price(self._close)

    def _margin_used_at_price(self, price):  # measured in BTC, without leverage
        return (self._btc_swap_long_position + self._btc_swap_short_position) * \
               self._contract_face_value / price / self._leverage_ratio

    def _unrealized_btc_at_price(self, price):
        return self._long_unrealized_btc_at_price(price) + self._short_unrealized_btc_at_price(price)

    def _long_unrealized_btc_at_price(self, price):
        return (1 / self._btc_long_position_price - 1 / price) * \
               self._btc_swap_long_position * self._contract_face_value \
            if self._btc_swap_long_position > EPS else 0.

    def _short_unrealized_btc_at_price(self, price):
        return (1 / price - 1 / self._btc_short_position_price) * \
               self._btc_swap_short_position * self._contract_face_value \
            if self._btc_swap_short_position > EPS else 0.

    def _commision(self, num_contracts):  # measured in BTC
        return num_contracts * self._contract_face_value * self._taker_fee_rate / self._close

    def _check_account(self) -> Tuple[bool, str]:  # -> error, msg
        # check it will not blow the account for each price
        for price in [self._low, self._high, self._close]:
            if self._margin_used_at_price(price)  == 0:
                self._margin_ratio = INF
            else:
                self._margin_ratio = (self._btc + self._unrealized_btc_at_price(price)) / \
                    self._margin_used_at_price(price) - self._adjustment_factor
            if self._margin_ratio < 0.:
                self.history['price'][self.step_counter] = price
                self.history['equity'][self.step_counter] = self._equity_usd
                self.history['long_position'][self.step_counter] = self._btc_swap_long_position
                self.history['short_position'][self.step_counter] = self._btc_swap_short_position
                self.history['margin_ratio'][self.step_counter] = self._margin_ratio
                self._reset_account()
                return True, 'Forced liquidation!'

        return False, ''

    def _swap_step(self, direction, num_contracts) -> Tuple[bool, str]:  # -> error, msg
        if direction == None or num_contracts <= EPS:
            return False, ''

        if (direction == TradeSide.LONG_OPEN or direction == TradeSide.SHORT_OPEN) \
                and num_contracts > self._swap_buying_power:
            return True, 'Buying power is not enough'

        trade_worth = num_contracts * self._contract_face_value

        if direction == TradeSide.LONG_OPEN:
            btc_quantity = trade_worth / apply_dummp_slippage(direction, self._close)
            self._btc_swap_long_position += num_contracts
            self._btc_long_position_price = \
                (trade_worth + self._btc_long_position_price * self._btc_long_quantity) / \
                (btc_quantity + self._btc_long_quantity)
            self._btc_long_quantity += btc_quantity

        if direction == TradeSide.SHORT_OPEN:
            btc_quantity = trade_worth / apply_dummp_slippage(direction, self._close)
            self._btc_swap_short_position += num_contracts
            self._btc_short_position_price = \
                (trade_worth + self._btc_short_position_price * self._btc_short_quantity) / \
                (btc_quantity + self._btc_short_quantity)
            self._btc_short_quantity += btc_quantity

        if direction == TradeSide.LONG_CLOSE and self._btc_swap_long_position > EPS:
            if num_contracts > self._btc_swap_long_position:
                return True, 'Selling power is not enough'

            position_ratio = num_contracts / self._btc_swap_long_position
            self._btc += position_ratio * self._long_unrealized_btc
            self._btc_swap_long_position *= (1. - position_ratio)
            self._btc_long_quantity *= (1. - position_ratio)

        if direction == TradeSide.SHORT_CLOSE and self._btc_swap_short_position > EPS:
            if num_contracts > self._btc_swap_short_position:
                return True, 'Selling power is not enough'

            position_ratio = num_contracts / self._btc_swap_short_position
            self._btc += position_ratio * self._short_unrealized_btc
            self._btc_swap_short_position *= (1. - position_ratio)
            self._btc_short_quantity *= (1. - position_ratio)

        # Deduct commission fee
        self._btc -= self._commision(num_contracts)

        return False, ''

    def account_obs(self) -> Dict:
        return {
            'btc': self._btc,
            'long_unrealized_btc': self._long_unrealized_btc,
            'short_unrealized_btc': self._short_unrealized_btc,
            'unrealized_btc': self._unrealized_btc,
            'equity': self._equity_usd,
            'swap_buying_power': self._swap_buying_power,
            'margin_ratio': self._margin_ratio,
            'btc_swap_long_position': self._btc_swap_long_position,
            'btc_swap_short_position': self._btc_swap_short_position,
            'btc_long_position_price': self._btc_long_position_price,
            'btc_short_position_price': self._btc_short_position_price
        }

    def _on_market_obs(self, market_obs: Dict):
        # update prices (for next iteration)
        self._open, self._high, self._low, self._close = \
            market_obs['open'], market_obs['high'], market_obs['low'], market_obs['close']

    def step(self, direction: TradeSide, num_contracts: int) -> Tuple[Dict, Dict, bool, str]:
        """
        Arguments:
            direction:
            num_contracts
        Returns:
            market observation (Dict):
            account observation (Dict):
            done (bool):
            message (str):
        """
        error, msg = self._swap_step(direction, num_contracts)  # error in step does not need handle
        error_, msg_ = self._check_account()
        if error_:
            return {}, {}, True, msg_  # if account is blown, return Done = True

        done = not self.feed.has_next()

        self.history['price'][self.step_counter] = self._close
        self.history['equity'][self.step_counter] = self._equity_usd
        self.history['long_position'][self.step_counter] = self._btc_swap_long_position
        self.history['short_position'][self.step_counter] = self._btc_swap_short_position
        self.history['margin_ratio'][self.step_counter] = self._margin_ratio
        self.step_counter += 1

        market_obs = self.feed.next()
        self._on_market_obs(market_obs)

        return market_obs, self.account_obs(), done, msg

    def reset(self) -> [Dict, Dict]:
        self._swap_compile(self.env_config)
        self.feed.reset()

        self.history = {'price': np.zeros(self.data_len-1), \
                        'equity': np.zeros(self.data_len-1), \
                        'long_position': np.zeros(self.data_len-1), \
                        'short_position': np.zeros(self.data_len-1), \
                        'margin_ratio': np.zeros(self.data_len-1)}
        self.step_counter = 0

        market_obs = self.feed.next()
        self._on_market_obs(market_obs)
        self._check_account()

        return market_obs, self.account_obs(), False, None
