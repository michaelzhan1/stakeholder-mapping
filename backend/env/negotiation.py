import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector

from pettingzoo.test import api_test

from enum import Enum

class Outcome(Enum):
    FAILURE = 0
    SUCCESS = 1
    EXISTING = 2
    SELF = 3

class NegotiationEnv(AECEnv):
    metadata = {
        "render_modes": ["ansi"],
        "name": "negotiation_v1",
        "is_parallelizable": False
    }
    
    def __init__(self, stakeholder_matrix=None, render_mode=None):
        # init agent info
        self.stakeholders = self._init_stakeholders(stakeholder_matrix)
        self.n_agents = len(self.stakeholders)
        self.agents = [f"agent_{idx + 1}" for idx in range(self.n_agents)]
        self.possible_agents = self.agents[:]

        # convenience mappings:
        self.agent_to_idx = {agent: idx for idx, agent in enumerate(self.agents)}
        self.idx_to_agent = {idx: agent for idx, agent in enumerate(self.agents)}

        # define key stakeholders
        self.primary = 'agent_1'
        self.target = f'agent_{self.n_agents}'

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._action_spaces = {agent: spaces.Discrete(self.n_agents) for agent in self.agents}
        self._observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(self.n_agents, self.n_agents), dtype=np.int8) for agent in self.agents}

        self.render_mode = render_mode

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
    
    def reset(self, seed=0, **kwargs):
        np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.agent_selection = self._agent_selector.reset()

        for i in range(self.n_agents):
            agent = self.idx_to_agent[i]
            self.stakeholders[agent]['relationships'] = np.zeros(self.n_agents, dtype=np.int8)
            self.stakeholders[agent]['relationships'][i] = 1
    
    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(None)
            return

        self._cumulative_rewards[agent] = 0
        recipient = self.idx_to_agent[action]

        outcome = self._engage(agent, recipient)
        self.rewards[agent] = self._calculate_reward(agent, recipient, outcome)
        self._accumulate_rewards()

        terminated = self._check_termination()
        if terminated:
            for i in range(self.n_agents):
                self.terminations[self.idx_to_agent[i]] = True
        
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        obs = np.zeros((0, self.n_agents), dtype=np.int8)
        for i in range(self.n_agents):
            obs = np.vstack((obs, self.stakeholders[self.idx_to_agent[i]]['relationships']))
        return obs

    def render(self):
        pass

    def close(self):
        pass

    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    # ================== #
    # END API METHODS    #
    # ================== #
    
    def _init_stakeholders(self, stakeholder_matrix):
        if stakeholder_matrix is None:
            # Default: Generate a simple set of stakeholders
            return {
                "agent_1": {
                    "position": 1, "power": 0, "knowledge": 1, "urgency": 1, "legitimacy": 1,
                    "relationships": np.array([1, 0, 0], dtype=np.int8)
                },
                "agent_2": {
                    "position": 0, "power": 2, "knowledge": 1, "urgency": 0, "legitimacy": 1,
                    "relationships": np.array([0, 1, 0], dtype=np.int8)
                },
                "agent_3": {
                    "position": -1, "power": 2, "knowledge": 2, "urgency": 1, "legitimacy": 1,
                    "relationships": np.array([0, 0, 1], dtype=np.int8)
                }
            }
        else:
            stakeholders = {}
            n_agents = len(stakeholder_matrix)
            for idx, row in enumerate(stakeholder_matrix):
                name = f"agent_{idx + 1}"
                stakeholders[name] = {
                    "position": row[0],
                    "power": row[1],
                    "knowledge": row[2],
                    "urgency": row[3],
                    "legitimacy": row[4],
                    "relationships": np.zeros(n_agents, dtype=int)
                }
                stakeholders[name]['relationships'][idx] = 1
            return stakeholders

    def _engage(self, agent, recipient):
        if agent == recipient:
            return Outcome.SELF
        if self.stakeholders[agent]['relationships'][self.agent_to_idx[recipient]] == 1:
            return Outcome.EXISTING
        
        if np.random.rand() < self._calculate_probability(agent, recipient):
            self.stakeholders[agent]['relationships'][self.agent_to_idx[recipient]] = 1
            self.stakeholders[recipient]['relationships'][self.agent_to_idx[agent]] = 1
            return Outcome.SUCCESS
        return Outcome.FAILURE

    def _calculate_probability(self, agent, recipient):
        agent_info = self.stakeholders[agent]
        recipient_info = self.stakeholders[recipient]

        attributes = ['position', 'power', 'knowledge', 'urgency', 'legitimacy']
        attr_diffs = [abs(agent_info[attr] - recipient_info[attr]) for attr in attributes]
        max_diffs = [2, 2, 2, 1, 1]
        return 1 - sum(attr_diffs) / sum(max_diffs)


    def _calculate_reward(self, agent, recipient, outcome):
        match outcome:
            case Outcome.SUCCESS:
                if agent == self.primary and recipient == self.target:
                    return 10
                else:
                    recipient_state = self.stakeholders[recipient]
                    return recipient_state['power'] + recipient_state['knowledge'] + recipient_state['urgency'] + recipient_state['legitimacy']
            case Outcome.EXISTING:
                return -3
            case Outcome.SELF:
                return -1
            case Outcome.FAILURE:
                return -0.1
    
    def _check_termination(self):
        return self.stakeholders[self.primary]['relationships'][self.agent_to_idx[self.target]] == 1


if __name__ == "__main__":
    env = NegotiationEnv()
    api_test(env, num_cycles=1000, verbose_progress=True)
