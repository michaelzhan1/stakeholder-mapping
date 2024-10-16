import numpy as np
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces

class NegotiationEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "negotiation_v0"}

    def __init__(self, n_agents=3, p=0.5):
        super().__init__()
        self.n_agents = n_agents  # Number of agents (1 primary + (n_agents-1) common)
        self.p = p  # Probability of successful engagement

        # Define agents
        self.agents = ["primary"]+[f"agent_{i}" for i in range(n_agents-1)]
        self.possible_agents = self.agents[:]
        self.primary_agent = self.agents[0]  # First agent is the primary stakeholder

        # Initialize spaces (power: 0, 1, or 2; engagements: a set of engaged agents)
        self.action_spaces = {agent: spaces.Discrete(n_agents) for agent in self.agents} # agent can always engage with any stakeholder
        self.observation_spaces = {
            agent: spaces.Dict({
                "power": spaces.Discrete(3),  # Power levels: 0, 1, or 2
                "engagements": spaces.MultiBinary(n_agents)  # Binary vector for engagement status
            })
            for agent in self.agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        # EDIT
        self.agent_selection = self.agents[0]  # Start with the primary agent

        # Initialize state: random power levels and empty engagement sets
        self.state = {
            agent: {
                "power": np.random.randint(0, 3),  # Random power 0, 1, or 2
                "engagements": np.zeros(self.n_agents, dtype=int)
            }
            for agent in self.agents
        }
        return self._get_obs(self.agent_selection)

    def _get_obs(self, agent):
        """Return the observation for a given agent."""
        return self.state[agent]

    def step(self, action):
        """Perform one step in the environment."""
        agent = self.agent_selection
        if action == 0:
            # Do nothing
            reward = 0
        else:
            target_agent = self.agents[action - 1]
            if np.random.rand() < self.p:
                # Engagement successful, update states
                self.state[agent]["engagements"][action - 1] = 1
                self.state[target_agent]["engagements"][self.agents.index(agent)] = 1

                # Calculate reward
                reward = 1 if self.state[target_agent]["power"] > 0 else 0
                if target_agent == self.primary_agent:
                    reward += 2
                elif (target_agent == self.primary_agent) & (np.sum(self.state[agent]["engagements"]) > 1):
                    reward += 5
            else:
                # Engagement failed
                reward = 0

        # Update rewards and prepare for next agent's turn
        self.rewards[agent] += reward
        self._next_agent()

    def _next_agent(self):
        """Select the next agent in the sequence."""
        current_index = self.agents.index(self.agent_selection)
        self.agent_selection = self.agents[(current_index + 1) % self.n_agents]

    def render(self, mode="human"):
        """Render the current state of the environment."""
        print("-"*40)
        for agent, state in self.state.items():
            print(f"{agent}: Power = {state['power']}, Engagements = {state['engagements']}")

    def close(self):
        """Close the environment (optional cleanup)."""
        pass

# Wrapping the environment to ensure compatibility with Stable Baselines3
def env(n_agents=3, p=0.5):
    env = NegotiationEnv(n_agents=n_agents, p=p)
    return wrappers.CaptureStdoutWrapper(env)


if __name__ == "__main__":
    env = NegotiationEnv(n_agents=3, p=0.5)
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