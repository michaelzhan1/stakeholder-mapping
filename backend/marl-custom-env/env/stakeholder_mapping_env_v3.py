import numpy as np
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv
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

    def _define_spaces(self):
        n_agents = len(self.agents)

        self.action_spaces = {
            agent: spaces.Discrete(n_agents) for agent in self.agents
        }

        self.observation_spaces = {
            agent: spaces.Dict({
                "position": spaces.Box(low=-1, high=1, shape=(1,), dtype=int),
                "power": spaces.Discrete(3),
                "knowledge": spaces.Discrete(3),
                "urgency": spaces.Discrete(2),
                "legitimacy": spaces.Discrete(2),
                "relationships": spaces.MultiBinary(n_agents)
            }) for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.agent_selection = self.agents[0]

        # Reset relationships
        for agent in self.agents:
            self.stakeholders[agent]["relationships"] = np.zeros(len(self.agents), dtype=int)

        return self._get_obs(self.agent_selection), {}
    
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return self._get_obs(agent)

    def _get_obs(self, agent):
        state = self.stakeholders[agent]
        return {
            "position": np.array([state["position"]]),
            "power": state["power"],
            "knowledge": state["knowledge"],
            "urgency": state["urgency"],
            "legitimacy": state["legitimacy"],
            "relationships": state["relationships"]
        }

    def step(self, action):
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
        terminated = self._check_termination()

        self._next_agent()

        obs = self._get_obs(self.agent_selection)
        return obs, reward, terminated, False, {}

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
        current_index = self.agents.index(self.agent_selection)
        self.agent_selection = self.agents[(current_index + 1) % len(self.agents)]

    def _check_termination(self):
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
        action = np.random.choice(env.action_spaces[agent].n)  
        obs, reward, terminated, truncated, _ = env.step(action)

        print(f"\nAgent: {agent}, Action: {action}, Reward: {reward}")
        print(f"Observation: {obs}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        env.render()

        if terminated or truncated:
            break