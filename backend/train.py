import os
from copy import deepcopy
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

from env.negotiation import NegotiationEnv

def get_env(data=None):
    return PettingZooEnv(NegotiationEnv(data))

def get_agents(data=None):
    env = get_env(data)
    observation_space = env.observation_space

    optim = None

    agents = []
    for _ in range(env.n_agents):
        net = Net(observation_space.shape, env.action_space)
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)

        agent = DQNPolicy(
            model=net,
            optim=optim,
            action_space=env.action_space,
        )
        agents.append(agent)
    policy = MultiAgentPolicyManager(agents, env)
    
    return policy, optim, env.agents

if __name__ == "__main__":
    train_envs = DummyVectorEnv([get_env for _ in range(10)])
    test_envs = DummyVectorEnv([get_env for _ in range(10)])

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    policy, optim, agents = get_agents()

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(1000, 10),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect()

    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ttt", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "ttt", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 0.6

    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    )



