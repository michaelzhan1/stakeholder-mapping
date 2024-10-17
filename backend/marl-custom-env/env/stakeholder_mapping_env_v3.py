import functools

import numpy as np
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces


class NegotiationEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "negotiation_v1"}

    def __init__(self, stakeholder_matrix=None):
        super().__init__()

        # Parse stakeholder configuration
        self.stakeholders = self._initialize_stakeholders(stakeholder_matrix)

        # Define agents based on stakeholder names
        self.agents = list(self.stakeholders.keys())
        self.possible_agents = self.agents[:]
        self.primary_agent = self._find_primary_agent()
        self.negotiator = self._find_negotiator()

        if self.negotiator is None:
            raise ValueError("No negotiator found in the stakeholder configuration.")

        # Define action and observation spaces
        self._define_spaces()

        self.reset()

    def __init__(self, stakeholder_matrix=None, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        These attributes should not be changed after initialization.
        """
        super().__init__()

        # Parse stakeholder configuration
        self.stakeholders = self._initialize_stakeholders(stakeholder_matrix)

        # Define agents based on stakeholder names
        self.agents = list(self.stakeholders.keys())
        self.possible_agents = self.agents[:]
        self.primary_agent = self._find_primary_agent()
        self.negotiator = self._find_negotiator()

        # Mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        if self.negotiator is None:
            raise ValueError("No negotiator found in the stakeholder configuration.")

        # Define action and observation spaces
        self._action_spaces = {agent: self.action_space(agent) for agent in self.agents}      
        self._observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}

        self.render_mode = render_mode
        # self.reset() # EDIT

    def _initialize_stakeholders(self, stakeholder_matrix):
        if stakeholder_matrix is None:
            # Default: Generate a simple set of stakeholders
            return {
                "agent_1": {
                    "position": -1, "power": 2, "knowledge": 2, "urgency": 1, "legitimacy": 1,
                    "is_primary": True, "is_negotiator": False,
                    "relationships": [0, 0, 0]  # Relationship with self, agent_2, agent_3
                },
                "agent_2": {
                    "position": 0, "power": 2, "knowledge": 1, "urgency": 0, "legitimacy": 1,
                    "is_primary": False, "is_negotiator": True,
                    "relationships": [0, 0, 0]  # Relationship with agent_1, self, agent_3
                },
                "agent_3": {
                    "position": 1, "power": 0, "knowledge": 1, "urgency": 1, "legitimacy": 1,
                    "is_primary": False, "is_negotiator": False,
                    "relationships": [0, 0, 0]  # Relationship with agent_1, agent_2, self
                }
            }
        else:
            stakeholders = {}
            for idx, row in enumerate(stakeholder_matrix):
                name = f"agent_{idx + 1}"
                stakeholders[name] = {
                    "position": row[0],
                    "power": row[1],
                    "knowledge": row[2],
                    "urgency": row[3],
                    "legitimacy": row[4],
                    "is_primary": bool(row[5]),
                    "is_negotiator": bool(row[6]),
                    "relationships": np.zeros(len(stakeholder_matrix), dtype=int)
                }
            return stakeholders

    def _find_primary_agent(self):
        for agent, attrs in self.stakeholders.items():
            if attrs["is_primary"]:
                return agent
        raise ValueError("No primary agent found in the stakeholder configuration.")

    def _find_negotiator(self):
        for agent, attrs in self.stakeholders.items():
            if attrs["is_negotiator"]:
                return agent
        raise ValueError("No negotiator found in the stakeholder configuration.")

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Our observation space must include the agent's own characteristics and relationships,
        as well as the characteristics of other stakeholders
        """
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/

        # Observation space components
        self_info_space = spaces.Dict({
            "position": spaces.Discrete(3, start=-1),
            "power": spaces.Discrete(3),
            "knowledge": spaces.Discrete(3),
            "urgency": spaces.Discrete(2),
            "legitimacy": spaces.Discrete(2)
        })

        obs = spaces.Dict({
            'self_info': self_info_space,
            'self_relationships': spaces.MultiBinary(len(self.agents)),
            'other_agents_info': spaces.Dict({
                agent: self_info_space for agent in self.agents
            })
        })
        return obs

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(self.agents))


    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.agent_selection = self.agents[0]

        self.infos = {agent: {} for agent in self.agents}

        # Reset relationships
        for agent in self.agents:
            self.stakeholders[agent]["relationships"] = np.zeros(len(self.agents), dtype=int)

        return self._get_obs(self.agent_selection), self.infos
    
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.

        This includes the relationship array.
        """
        return self._get_obs(agent)
    
    def _get_self_info(self, agent):
        """Helper for _get_obs"""
        state = self.stakeholders[agent]
        self_info = {
            "position": state["position"],
            "power": state["power"],
            "knowledge": state["knowledge"],
            "urgency": state["urgency"],
            "legitimacy": state["legitimacy"]
        }
        return self_info

    def _get_obs(self, agent):
        state = self.stakeholders[agent]

        obs = {
            'self_info': self._get_self_info(agent),
            'self_relationships': state["relationships"],
            'other_agents_info': {
                other_agent: self._get_self_info(other_agent) for other_agent in self.agents
            }
        }
        return obs

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.terminations[self.agent_selection]:
            return self._get_obs(self.agent_selection), 0, True, False, {}

        agent = self.agent_selection

        if action == 0:
            # Do nothing
            reward = 0
        else:
            target_agent = self.agents[action - 1]
            reward = self._attempt_engagement(agent, target_agent)

        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward
        terminated = self._check_termination()

        self._next_agent()

        obs = self._get_obs(self.agent_selection)
        return obs, reward, terminated, False, self.infos

    def _attempt_engagement(self, agent, target_agent):
        if agent == target_agent:
            return 0  # Can't engage with self

        agent_state = self.stakeholders[agent]
        target_state = self.stakeholders[target_agent]

        # Calculate engagement probability
        p = self._calculate_engagement_probability(agent_state, target_state)

        if np.random.rand() < p:
            # Engagement successful
            agent_state["relationships"][self.agents.index(target_agent)] = 1
            target_state["relationships"][self.agents.index(agent)] = 1
            return self._calculate_reward(agent, target_agent)
        else:
            # Engagement failed
            return -1  # Negative reward for unsuccessful interaction

    def _calculate_engagement_probability(self, agent_state, target_state):
        # Implement logic for engagement probability based on stakeholder attributes
        # This is a simplified version and can be expanded
        p = 0.5  # Base probability

        if agent_state["position"] == target_state["position"]:
            p += 0.2
        elif agent_state["position"] * target_state["position"] < 0:  # Opposing positions
            p -= 0.2

        if agent_state["power"] > target_state["power"]:
            p += 0.1
        elif agent_state["power"] < target_state["power"]:
            p -= 0.1

        return max(0.1, min(0.9, p))  # Clamp probability between 0.1 and 0.9

    def _calculate_reward(self, agent, target_agent):
        agent_state = self.stakeholders[agent]
        target_state = self.stakeholders[target_agent]

        reward = 0

        # Reward for engaging with high power/knowledge stakeholder
        if target_state["power"] > 1 or target_state["knowledge"] > 1:
            reward += 1

        # Reward for engaging with primary stakeholder
        if target_agent == self.primary_agent:
            if sum(agent_state["relationships"]) >= 2:
                reward += 5
            else:
                reward += 1

        # Penalty for engaging with low power stakeholder
        if target_state["power"] == 0:
            reward -= 1

        return reward

    def _next_agent(self):
        """Agents are allowed to negotiate in order of index."""
        current_index = self.agents.index(self.agent_selection)
        self.agent_selection = self.agents[(current_index + 1) % len(self.agents)]

    def _check_termination(self):
        """Terminates when the negotiator engages with the primary negotiator"""
        if self.agent_selection == self.negotiator:
            engaged_with_primary = self.stakeholders[self.negotiator]["relationships"][self.agents.index(self.primary_agent)] == 1
            if engaged_with_primary:
                self.terminations = {a: True for a in self.agents}
            return engaged_with_primary
        return False

    def render(self, mode="human"):
        print("-" * 40)
        for agent, state in self.stakeholders.items():
            print(f"{agent}: Position = {state['position']}, Power = {state['power']}, "
                  f"Knowledge = {state['knowledge']}, Urgency = {state['urgency']}, "
                  f"Legitimacy = {state['legitimacy']}, Relationships = {state['relationships']}")

    def close(self):
        pass

# Wrapping the environment for Stable Baselines3 compatibility
def env(stakeholder_matrix=None):
    return wrappers.CaptureStdoutWrapper(NegotiationEnv(stakeholder_matrix))

if __name__ == "__main__":
    env = NegotiationEnv()
    obs, _ = env.reset()
    for _ in range(10):
        agent = env.agent_selection
        action = np.random.choice(env._action_spaces[agent].n)  
        obs, reward, terminated, truncated, _ = env.step(action)

        print(f"\nAgent: {agent}, Action: {action}, Reward: {reward}")
        print(f"Observation: {obs}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        env.render()

        if terminated or truncated:
            break