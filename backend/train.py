import argparse
import os
from typing import Tuple

import gymnasium
import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from env.negotiation import NegotiationEnv
from visualization import AnimatedGraph

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=2,
        help="the learned agent plays as the"
        " agent_id-th player. Choices are 1 and 2.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_agents(
    args: argparse.Namespace = get_args(),
    stakeholder_vals: np.ndarray | None = None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env(stakeholder_vals)
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n


    agents = []
    for _ in range(env.env.n_agents):
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)

        agents.append(DQNPolicy(net, optim, args.gamma, args.n_step, target_update_freq=args.target_update_freq))
    policy = MultiAgentPolicyManager(agents, env)
    return policy, env.agents


def get_env(data=None, render_mode=None):
    return PettingZooEnv(NegotiationEnv(stakeholder_matrix=data, render_mode=render_mode))


def train_agent(
    args: argparse.Namespace = get_args(),
    stakeholder_vals: np.ndarray | None = None
) -> Tuple[dict, BasePolicy]:
    # ======== environment setup =========
    train_envs = DummyVectorEnv([lambda: get_env(stakeholder_vals) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: get_env(stakeholder_vals) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, agents = get_agents(args, stakeholder_vals)

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "negotiate", "dqn", "policy.pth"
            )
        torch.save(
            policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path
        )

    def train_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, args.agent_id - 1]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        test_in_train=False,
        reward_metric=reward_metric
    )

    return result, policy.policies

def train(data):
    args = get_args()
    result, policies = train_agent(args, stakeholder_vals=data)
    return policies

def run(data, save=False):
    res = ""
    policies = train(data)
    env = get_env(data)
    obs, info = env.reset()
    done = False

    # track for gif
    adj_matrices = []
    actions = []

    while not done:
        agent = env.env.agent_selection
        policy = policies[agent]
        action = policy.forward(batch=Batch(obs=[obs], info=[info])).act[0]
        
        agent_idx = env.env.agent_to_idx[agent]
        recipient = f'agent_{action + 1}'

        # text output
        res += f'{agent} targeting {recipient}\n'
        res += str(env.env.observe(None)) + '\n'
        res += '\n'

        # gif frame
        adj_matrix = env.env.observe(None)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrices.append(adj_matrix)
        actions.append((agent_idx, action))
        
        obs, rew, done, truncated, info = env.step(action)

    res += 'Final state:\n'
    res += str(env.env.observe(None)) + '\n'

    final_state = env.env.observe(None)
    np.fill_diagonal(final_state, 0)
    adj_matrices.append(final_state)
    actions.append("Final State")

    env.close()

    if save:
        labels = {i: env.env.idx_to_agent[i] for i in range(env.env.n_agents)}
        AnimatedGraph(adj_matrices, actions, node_labels=labels).animate(save=True)
    return res

if __name__ == "__main__":
    data = pd.read_csv('data/test.csv', header=None).values
    print(run(data, save=True))
