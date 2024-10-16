import numpy as np
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces

class NegotiationEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "negotiation_v1"}

    def __init__(self, stakeholder_matrix=None, p=0.5):
        """
        Args:
            stakeholder_matrix (np.ndarray): Optional input matrix where each row
                represents a stakeholder with columns for characteristics (e.g., power)
                and a one-hot indicator for primary stakeholder.
            p (float): Probability of successful engagement.
        """
        super().__init__()

        # Parse stakeholder configuration
        self.stakeholders = self._initialize_stakeholders(stakeholder_matrix)
        self.p = p  # Probability of successful engagement

        # Define agents based on stakeholder names
        self.agents = list(self.stakeholders.keys())
        self.possible_agents = self.agents[:]
        self.primary_agent = self._find_primary_agent()

        # Define action and observation spaces
        self._define_spaces()

        self.reset()

    def _initialize_stakeholders(self, stakeholder_matrix):
        if stakeholder_matrix is None:
            # Default: Generate a simple set of stakeholders
            return {
                "agent_1": {"position": -1, "power": 2, "knowledge": 2, "urgency": 1, "legitimacy": 1, "is_primary": True, "is_negotiator": False},
                "agent_2": {"position": 0, "power": 2, "knowledge": 1, "urgency": 0, "legitimacy": 1, "is_primary": False, "is_negotiator": True},
                "agent_3": {"position": 1, "power": 0, "knowledge": 1, "urgency": 1, "legitimacy": 1, "is_primary": False, "is_negotiator": False}
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
        """Identify the primary agent from the stakeholder data."""
        for agent, attrs in self.stakeholders.items():
            if attrs["is_primary"]:
                return agent
        return self.agents[0]  # Default to the first agent if none marked primary

    def _find_negotiator(self):
        for agent, attrs in self.stakeholders.items():
            if attrs["is_negotiator"]:
                return agent
        return None
    
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
        """Reset the environment to its initial state."""
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.agent_selection = self.agents[0]  # Start with the first agent

        # Clear all engagements
        for agent in self.agents:
            self.stakeholders[agent]["relationships"] = np.zeros(len(self.agents), dtype=int)

        return self._get_obs(self.agent_selection)

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
        """Perform one step in the environment."""
        agent = self.agent_selection

        if action == 0:
            # Do nothing
            reward = 0
        else:
            target_agent = self.agents[action - 1]
            reward = self._attempt_engagement(agent, target_agent)

        # Update rewards and move to the next agent
        self.rewards[agent] += reward
        self._next_agent()

        # Check termination conditions
        if self.agent_selection == self.negotiator:
            engaged_with_primary = self.stakeholders[self.negotiator]["relationships"][self.agents.index(self.primary_agent)] == 1
            self.terminations = {a: engaged_with_primary for a in self.agents}

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
        """Select the next agent in the sequence."""
        current_index = self.agents.index(self.agent_selection)
        self.agent_selection = self.agents[(current_index + 1) % len(self.agents)]
    
    # add the termination function
    def _check_termination(self):
        """Check if the negotiation should terminate."""
        negotiator = self.negotiator
        primary = self.primary_agent

        if primary in self.stakeholders[negotiator]["engagements"]:
            # Negotiation ends when negotiator engages with primary stakeholder
            self.terminations = {agent: True for agent in self.agents}

            # Calculate final reward based on negotiation outcome
            final_reward = self._calculate_final_reward()
            self.rewards[negotiator] += final_reward

    def render(self, mode="human"):
        print("-" * 40)
        for agent, state in self.stakeholders.items():
            print(f"{agent}: Position = {state['position']}, Power = {state['power']}, "
                  f"Knowledge = {state['knowledge']}, Urgency = {state['urgency']}, "
                  f"Legitimacy = {state['legitimacy']}, Relationships = {state['relationships']}")

    def close(self):
        """Optional cleanup."""
        pass

# Wrapping the environment for Stable Baselines3 compatibility
def env(stakeholder_matrix=None, p=0.5):
    return wrappers.CaptureStdoutWrapper(NegotiationEnv(stakeholder_matrix, p))


if __name__ == "__main__":
    env = NegotiationEnv(p=0.5)
    env.reset()
    for _ in range(10):
        agent = env.agent_selection
        action = np.random.choice(env.action_spaces[agent].n)  
        env.step(action)
        obs = env._get_obs(agent)
        reward = env.rewards[agent]

        print()
        print(f"Agent: {agent}, Action: {action}, Reward: {reward}")
        print(f"Observation: {obs}")
        env.render()