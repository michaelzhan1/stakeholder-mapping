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
        """Initialize stakeholders with their attributes."""
        if stakeholder_matrix is None:
            # Default: Randomly generate stakeholders if no matrix is provided
            num_agents = 3
            stakeholders = {
                f"agent_{i}": {
                    "power": np.random.randint(0, 3),  # Power: 0, 1, or 2
                    "engagements": set()  # Engagement tracking as a set of agents
                }
                for i in range(num_agents)
            }
            return stakeholders
        else:
            # Initialize stakeholders from input matrix
            stakeholders = {}
            for idx, row in enumerate(stakeholder_matrix):
                name = f"agent_{idx}"
                stakeholders[name] = {
                    "power": row[0],  # Assuming first column is power
                    "primary": bool(row[-1]),  # Assuming last column is primary indicator
                    "engagements": set()
                }
            return stakeholders

    def _find_primary_agent(self):
        """Identify the primary agent from the stakeholder data."""
        for agent, attrs in self.stakeholders.items():
            if attrs.get("primary", False):
                return agent
        return self.agents[0]  # Default to the first agent if none marked primary

    def _define_spaces(self):
        """Define action and observation spaces."""
        n_agents = len(self.agents)

        # Action: 0 = Do nothing, 1 to n_agents = Speak with a stakeholder
        self.action_spaces = {
            agent: spaces.Discrete(n_agents) for agent in self.agents
        }

        # Observation: Power level and engagement status
        self.observation_spaces = {
            agent: spaces.Dict({
                "power": spaces.Discrete(3),
                "engagements": spaces.MultiBinary(n_agents)
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
            self.stakeholders[agent]["engagements"] = set()

        return self._get_obs(self.agent_selection)

    def _get_obs(self, agent):
        """Generate the observation for a given agent."""
        state = self.stakeholders[agent]
        engagements = np.zeros(len(self.agents), dtype=int)
        for engaged_agent in state["engagements"]:
            engagements[self.agents.index(engaged_agent)] = 1

        return {
            "power": state["power"],
            "engagements": engagements
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

    def _attempt_engagement(self, agent, target_agent):
        """Attempt to engage with another agent."""
        if np.random.rand() < self.p:
            # Engagement successful
            self.stakeholders[agent]["engagements"].add(target_agent)
            self.stakeholders[target_agent]["engagements"].add(agent)

            # Calculate reward
            return self._calculate_reward(agent, target_agent)
        else:
            # Engagement failed
            return 0

    def _calculate_reward(self, agent, target_agent):
        """Calculate the reward for engaging with a specific target."""
        reward = 1 if self.stakeholders[target_agent]["power"] > 0 else 0
        if target_agent == self.primary_agent:
            reward += 2
        elif len(self.stakeholders[agent]["engagements"]) > 1:
            reward += 5
        return reward

    def _next_agent(self):
        """Select the next agent in the sequence."""
        current_index = self.agents.index(self.agent_selection)
        self.agent_selection = self.agents[(current_index + 1) % len(self.agents)]

    def render(self, mode="human"):
        """Render the current state of the environment."""
        print("-"*40)
        for agent, state in self.stakeholders.items():
            print(f"{agent}: Power = {state['power']}, Engagements = {state['engagements']}")

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