# Copyright 2019 The nextrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import gym
import uuid
import logging
import numpy as np
import pandas as pd
import copy
import random

from gym.spaces import Box
from typing import Union, List, Tuple, Dict

import nextrade.actions as actions
import nextrade.rewards as rewards
import nextrade.wallets as wallets

from nextrade.base import TimeIndexed, Clock
from nextrade.actions import ActionScheme
from nextrade.rewards import RewardScheme
from nextrade.data import DataFeed, Stream
from nextrade.data.internal import create_internal_feed
from nextrade.orders import Broker
from nextrade.wallets import Portfolio
from nextrade.environments import ObservationHistory
from nextrade.environments.render import get

from nextrade.exchanges import Exchange
from nextrade.exchanges.services.execution.simulated import execute_order
from nextrade.data import Stream, DataFeed, Module
from nextrade.instruments import USD, BTC, ETH
from nextrade.wallets import Wallet, Portfolio

class TradingEnvironment(gym.Env, TimeIndexed):
    """A trading environments made for use with Gym-compatible reinforcement learning algorithms."""

    agent_id: str = None
    episode_id: str = None

    def __init__(self,
                 env_config: Dict,
                 **kwargs):
        """
        Arguments:
            portfolio: The `Portfolio` of wallets used to submit and execute orders from.
            action_scheme:  The component for transforming an action into an `Order` at each timestep.
            reward_scheme: The component for determining the reward at each timestep.
            feed (optional): The pipeline of features to pass the observations through.
            renderers (optional): single or list of renderers for output by name or as objects.
                String Values: 'screenlog', 'filelog', or 'plotly'. None for no rendering.
            price_history (optional): OHLCV price history feed used for rendering
                the chart. Required if render_mode is 'plotly'.
            kwargs (optional): Additional arguments for tuning the environments, logging, etc.
        """
        super().__init__()

        self.start_time = env_config['start_time']
        self.end_time = env_config['end_time']
        self.data = copy.deepcopy(env_config['data'][self.start_time:self.end_time])
        self.preprocess = env_config['preprocess']

        self.rl_env = env_config.get('rl_env', False)
        self.exchange_name = env_config.get('exchange_name', 'test')
        self.exchange = env_config.get('exchange',
                            Exchange(self.exchange_name, service=execute_order)(
                                Stream(list(self.data["close"])).rename("USD-BTC"),
                            ))
        self.portfolio = env_config.get('portfolio',
                            Portfolio(USD, [
                            Wallet(self.exchange, 10000 * USD),
                            Wallet(self.exchange, 0 * BTC),
                            ]))        
        self.action_scheme = env_config.get('action_scheme', 'simple')
        self.reward_scheme = env_config.get('reward_scheme', 'simple')
        self.window_size = env_config.get('window_size', 1)
        self.use_internal = env_config.get('use_internal', False)
        self.slice_len = env_config.get('slice_len', 60 * 24) # 1 day
        self.renderers = env_config.get('renders', ['screenlog'])

        self.feed = self.preprocess(self.data)
        self._price_history = self.data.copy().reset_index()
        self.data_len = len(self.feed.inputs[0]._array) # get length of data by query the first node which is a stream

        if self.feed:
            self._external_keys = [*self.feed.next().keys()]
            self._external_keys.remove('Timestamp')
            self.feed.reset()

        self.history = ObservationHistory(window_size=self.window_size)
        self._broker = Broker()

        self.clock = Clock()
        self.action_space = None
        self.observation_space = None

        renderers = self.renderers
        if not renderers:
            renderers = []
        elif type(renderers) is not list:
            renderers = [renderers]

        self._renderers = []
        for renderer in renderers:
            if isinstance(renderer, str):
                renderer = get(renderer)
            self._renderers.append(renderer)

        self._enable_logger = kwargs.get('enable_logger', False)
        self._observation_dtype = kwargs.get('dtype', np.float32)
        self._observation_lows = kwargs.get('observation_lows', -np.iinfo(np.int64).max)
        self._observation_highs = kwargs.get('observation_highs', np.iinfo(np.int64).max)
        self._max_allowed_loss = kwargs.get('max_allowed_loss', 0.1)

        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

        self._max_episodes = None
        self._max_steps = None

        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

        self.compile()

    @property
    def max_episodes(self) -> int:
        return self._max_episodes

    @max_episodes.setter
    def max_episodes(self, max_episodes: int):
        self._max_episodes = max_episodes

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @max_steps.setter
    def max_steps(self, max_steps: int):
        self._max_steps = max_steps

    def compile(self):
        """
        Sets the observation space and the action space of the environment.
        Creates the internal feed and sets initialization for different components.
        """
        components = [self._broker, self.portfolio, self.action_scheme,
                      self.reward_scheme] + self.portfolio.exchanges

        for component in components:
            component.clock = self.clock

        self.action_scheme.exchange_pairs = self.portfolio.exchange_pairs
        self.action_scheme.compile()
        self.action_space = self.action_scheme.action_space

        if not self.feed:
            self.feed = create_internal_feed(self.portfolio)
        else:
            self.feed = self.feed + create_internal_feed(self.portfolio)
        initial_obs = self.feed.next()
        n_features = len(initial_obs.keys()) if self.use_internal else len(self._external_keys)

        self.observation_space = Box(
            low=self._observation_lows,
            high=self._observation_highs,
            shape=(self.window_size, n_features + 3), # 2 for return history
            dtype=self._observation_dtype
        )

        self.feed.reset()

        self.best_return = -1.

    @property
    def portfolio(self) -> Portfolio:
        """The portfolio of instruments currently held on this exchange."""
        return self._portfolio

    @portfolio.setter
    def portfolio(self, portfolio: Union[Portfolio, str]):
        self._portfolio = wallets.get(portfolio) if isinstance(portfolio, str) else portfolio

    @property
    def broker(self) -> Broker:
        """The broker used to execute orders within the environment."""
        return self._broker

    @property
    def episode_trades(self) -> Dict[str, 'Trade']:
        """A dictionary of trades made this episode, organized by order id."""
        return self._broker.trades

    @property
    def action_scheme(self) -> ActionScheme:
        """The component for transforming an action into an `Order` at each time step."""
        return self._action_scheme

    @action_scheme.setter
    def action_scheme(self, action_scheme: Union[ActionScheme, str]):
        self._action_scheme = actions.get(action_scheme) if isinstance(
            action_scheme, str) else action_scheme

    @property
    def reward_scheme(self) -> RewardScheme:
        """The component for determining the reward at each time step."""
        return self._reward_scheme

    @reward_scheme.setter
    def reward_scheme(self, reward_scheme: Union[RewardScheme, str]):
        self._reward_scheme = rewards.get(reward_scheme) if isinstance(
            reward_scheme, str) else reward_scheme

    @property
    def price_history(self) -> pd.DataFrame:
        return self._price_history

    @price_history.setter
    def price_history(self, price_history):
        self._price_history = price_history

    def step(self, action: int) -> Tuple[np.array, float, bool, dict]:
        """Run one timestep within the environments based on the specified action.

        Arguments:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (pandas.DataFrame): Provided by the environments's exchange, often OHLCV or tick trade history data points.
            reward (float): An size corresponding to the benefit earned by the action taken this timestep.
            done (bool): If `True`, the environments is complete and should be restarted.
            info (dict): Any auxiliary, diagnostic, or debugging information to output.
        """

        orders = self.action_scheme.get_order(action, self.portfolio)

        if orders:
            if not isinstance(orders, list):
                orders = [orders]

            for order in orders:
                self._broker.submit(order)

        self._broker.update()
        
        self.net_worth = self.portfolio.performance.net_worth

        self.curr_return = self.net_worth.iloc[-1] / self.net_worth.iloc[0] - 1

        if self.curr_return > self.best_return:
            self.best_return = self.curr_return

        obs_row = self.feed.next()
        timestamp = obs_row['Timestamp']

        if not self.use_internal:
            obs_row = {k: obs_row[k] for k in self._external_keys}
        else:
            all_keys = [*self.feed.next().keys()]
            all_keys.remove('Timestamp')
            obs_row = {k: obs_row[k] for k in all_keys}

        obs_row["curr_return"] = self.curr_return
        obs_row["best_return"] = self.best_return

        if self.rl_env:
            self.history.push(obs_row)
            obs = self.history.observe()
            obs = obs.astype(self._observation_dtype)
        else:
            obs = obs_row

        reward = self.reward_scheme.get_reward(self._portfolio)
        reward = np.nan_to_num(reward)

        if np.bitwise_not(np.isfinite(reward)):
            raise ValueError('Reward returned by the reward scheme must by a finite float.')

        done = (self.portfolio.profit_loss < self._max_allowed_loss) or not self.feed.has_next()
            
        info = {
            'step': self.clock.step,
            'timestamp': timestamp
            #'portfolio': self.portfolio,
            #'broker': self._broker,
            #'order': orders[0] if orders else None,
        }

        if self._enable_logger:
            self.logger.debug('Order:       {}'.format(order))
            self.logger.debug('Observation: {}'.format(obs))
            self.logger.debug('P/L:         {}'.format(self._portfolio.profit_loss))
            self.logger.debug('Reward ({}): {}'.format(self.clock.step, reward))
            self.logger.debug('Performance: {}'.format(self._portfolio.performance.tail(1)))

        self.clock.increment()
        return obs, reward, done, info

    def reset(self) -> np.array:
        """Resets the state of the environments and returns an initial observation.

        Returns:
            The episode's initial observation.
        """
        self.episode_id = uuid.uuid4()
        self.clock.reset()
        self.feed.reset()
        self.action_scheme.reset()
        self.reward_scheme.reset()
        self.portfolio.reset()
        self.history.reset()
        self._broker.reset()

        for renderer in self._renderers:
            renderer.reset()

        # reset the stream inputs node cursor to a random place
        # other nodes have already been reseted in previous lines
        if self.slice_len > 0:
            start_pt = random.randint(0, self.data_len - self.slice_len)
            end_pt = start_pt + self.slice_len

            for node in self.feed.inputs:
                if isinstance(node, Stream):
                    node.reset(start=start_pt, end=end_pt)
                
            for wallet in self.portfolio._wallets:
                for node in self.portfolio._wallets[wallet]._exchange.inputs:
                    if isinstance(node, Stream):
                        node.reset(start=start_pt, end=end_pt)

        obs_row = self.feed.next()

        if not self.use_internal:
            obs_row = {k: obs_row[k] for k in self._external_keys}
        else:
            all_keys = [*self.feed.next().keys()]
            all_keys.remove('Timestamp')
            obs_row = {k: obs_row[k] for k in all_keys}

        obs_row["curr_return"] = 0.
        obs_row["best_return"] = 0.

        if self.rl_env:
            self.history.push(obs_row)
            obs = self.history.observe()
            obs = obs.astype(self._observation_dtype)
        else:
            obs = obs_row

        self.clock.increment()

        return obs

    def render(self, episode: int = None):
        """Renders the environment.

        Arguments:
            episode: Current episode number (0-based).
        """
        current_step = self.clock.step - 1
        for renderer in self._renderers:
            price_history = None if self._price_history is None else self._price_history[
                self._price_history.index < current_step]
            renderer.render(episode=episode,
                            max_episodes=self._max_episodes,
                            step=current_step,
                            max_steps=self._max_steps,
                            price_history=price_history,
                            net_worth=self._portfolio.performance.net_worth,
                            performance=self._portfolio.performance.drop(columns=['base_symbol']),
                            trades=self._broker.trades)

    def save(self):
        """Saves the environment.

        Arguments:
            episode: Current episode number (0-based).
        """
        for renderer in self._renderers:
            renderer.save()

    def close(self):
        """Utility method to clean environment before closing."""
        for renderer in self._renderers:
            if callable(hasattr(renderer, 'close')):
                renderer.close()  # pylint: disable=no-member
