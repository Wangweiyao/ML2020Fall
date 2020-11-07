import os
import gym
import torch
import pprint
import argparse
import numpy as np
import copy
import mlflow

from tianshou.env import VectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer

from .discrete_net import MLP

from nextrade.agents import Agent
from nextrade.environments import TradingEnvironment

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--reward-threshold', type=float, default=100000000)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=16)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


class DQNAgent(Agent):

    def __init__(self,
                 train_env_config: EnvConfig,
                 test_env_config: EnvConfig):

        # get args
        self.args=get_args()
        print("initialize envs")
        # multiple envs for training and testing
        self.train_envs = VectorEnv(
            [lambda: TradingEnvironment(train_env_config) for _ in range(self.args.training_num)])
        self.test_envs = VectorEnv(
            [lambda: TradingEnvironment(test_env_config) for _ in range(self.args.test_num)])

        # state and action space
        self.dummy_env = TradingEnvironment(train_env_config)
        self.observation_shape = self.dummy_env.observation_space.shape
        self.n_actions = self.dummy_env.action_space.n

        # seed
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        self.train_envs.seed(self.args.seed)
        self.test_envs.seed(self.args.seed)

        print("initialize models")
        # model
        self.net = MLP(self.args.layer_num, self.observation_shape, self.n_actions, self.args.device)
        self.net = self.net.to(self.args.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.policy = DQNPolicy(
            self.net, self.optim, self.args.gamma, self.args.n_step,
            use_target_network=self.args.target_update_freq > 0,
            target_update_freq=self.args.target_update_freq)

        # collector
        print("initialize collectors")
        self.train_collector = Collector(
            self.policy, self.train_envs, ReplayBuffer(self.args.buffer_size)) #self.train_envs
        self.test_collector = Collector(self.policy, self.test_envs) #self.test_envs
        # policy.set_eps(1)
        print("collect...")
        self.train_collector.collect(n_step=self.args.batch_size)

    def train(self):

        def save_fn(policy):
            torch.save(self.policy.state_dict(), os.path.join(self.log_path, 'policy.pth'))

        def stop_fn(x):
            return x >= self.args.reward_threshold

        def train_fn(x):
            self.policy.set_eps(self.args.eps_train)

        def test_fn(x):
            self.policy.set_eps(self.args.eps_test)

        # trainer
        result = offpolicy_trainer(
            self.policy, self.train_collector, self.test_collector, self.args.epoch,
            self.args.step_per_epoch, self.args.collect_per_step, self.args.test_num,
            self.args.batch_size, train_fn=train_fn, test_fn=test_fn,
            stop_fn=stop_fn, save_fn=save_fn, writer=None, verbose=True)

        self.dummy_env.render(0)

        self.train_collector.close()
        self.test_collector.close()

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        Batch = self.policy.forward(state)
        return Batch.act

        
    def save(self, path: str, **kwargs):
        mlflow.log_model(self.policy, path)

    def restore(self, path: str, **kwargs):
        self.policy = mlflow.pytorch.load_model()

    def train_old(self,
              n_steps: int = None,
              n_episodes: int = None,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        batch_size: int = kwargs.get('batch_size', 128)
        discount_factor: float = kwargs.get('discount_factor', 0.9999)
        learning_rate: float = kwargs.get('learning_rate', 0.0001)
        eps_start: float = kwargs.get('eps_start', 0.9)
        eps_end: float = kwargs.get('eps_end', 0.05)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', 200)
        update_target_every: int = kwargs.get('update_target_every', 1000)
        memory_capacity: int = kwargs.get('memory_capacity', 1000)
        render_interval: int = kwargs.get('render_interval', 50)  # in steps, None for episode end render only

        memory = ReplayMemory(memory_capacity, transition_type=DQNTransition)
        episode = 0
        total_reward = 0
        stop_training = False

        if n_steps and not n_episodes:
            n_episodes = np.iinfo(np.int32).max

        print('====      AGENT ID: {}      ===='.format(self.id))
        self.env.max_episodes = n_episodes
        self.env.max_steps = n_steps

        while episode < n_episodes and not stop_training:
            state = self.env.reset()
            done = False
            steps_done = 0

            while not done:
                threshold = eps_end + (eps_start - eps_end) * np.exp(-steps_done / eps_decay_steps)
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)

                memory.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps_done += 1

                if len(memory) < batch_size:
                    continue

                self._apply_gradient_descent(memory, batch_size, learning_rate, discount_factor)

                if n_steps and steps_done >= n_steps:
                    done = True

                if render_interval is not None and steps_done % render_interval == 0:
                    self.env.render(episode)

                if steps_done % update_target_every == 0:
                    self.target_network = tf.keras.models.clone_model(self.policy_network)
                    self.target_network.trainable = False

            is_checkpoint = save_every and episode % save_every == 0

            if save_path and (is_checkpoint or episode == n_episodes - 1):
                self.save(save_path, episode=episode)

            if not render_interval or steps_done < n_steps:
                self.env.render(episode)  # render final state at episode end if not rendered earlier

            self.env.save()

            episode += 1

        mean_reward = total_reward / steps_done

        return mean_reward
