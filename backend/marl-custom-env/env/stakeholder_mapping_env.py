# Custom MARL Negotiation Environment: Simplified Case

import numpy as np
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces

class NegotiationEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "negotiation_v1"}

    def __init__(self, stakeholder_matrix=None, render_mode=None):
        """Initialize the negotiation environment."""

        # Parse stakeholder configuration
        self.stakeholders = self._initialize_stakeholders(stakeholder_matrix)

        # Define agents based on stakeholder names
        self.possible_agents = list(self.stakeholders.keys())
        self.agents = self.possible_agents[:]
        self.primary_agent = self._find_primary_agent()
        self.negotiator = self._find_negotiator()
        self.n_agents = len(self.agents)

        # Mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        if self.negotiator is None:
            raise ValueError("No negotiator found in the stakeholder configuration.")

        # Pre-compute action and observation spaces
        self._action_spaces = {}
        self._observation_spaces = {}

        for agent in self.possible_agents:
            # Create action space
            self._action_spaces[agent] = spaces.Discrete(self.n_agents)

            # Create observation space
            low = np.repeat(np.array([[-1,0,0,0,0]]), repeats=self.n_agents, axis=0) # lowest bound for characteristics

            self._observation_spaces[agent] = spaces.Dict({
                "observation": spaces.MultiDiscrete(np.array([[3,3,3,2,2]]*self.n_agents), start=low),
                "action_mask": spaces.Box(low=0, high=1, shape=(self.n_agents,), dtype=np.int8)
            })

        self.render_mode = render_mode

    def _initialize_stakeholders(self, stakeholder_matrix):
        """
        Initializes a test case if user does not provide stakeholders
        """
        if stakeholder_matrix is None:
            # Default: Generate a simple set of stakeholders
            return {
                "agent_1": {
                    "position": -1, "power": 2, "knowledge": 2, "urgency": 1, "legitimacy": 1,
                    "is_primary": True, "is_negotiator": False,
                    "relationships": np.zeros(3, dtype=int)
                },
                "agent_2": {
                    "position": 0, "power": 2, "knowledge": 1, "urgency": 0, "legitimacy": 1,
                    "is_primary": False, "is_negotiator": True,
                    "relationships": np.zeros(3, dtype=int)
                },
                "agent_3": {
                    "position": 1, "power": 0, "knowledge": 1, "urgency": 1, "legitimacy": 1,
                    "is_primary": False, "is_negotiator": False,
                    "relationships": np.zeros(3, dtype=int)
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
                    "is_primary": bool(row[5]),
                    "is_negotiator": bool(row[6]),
                    "relationships": np.zeros(n_agents, dtype=int)
                }
            return stakeholders


    def observation_space(self, agent):
        """Return the cached observation space for the agent."""
        return self._observation_spaces[agent]

    def action_space(self, agent):
        """Return the cached action space for the agent."""
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        # Reset RNG state
        if seed is not None:
            np.random.seed(seed)

        # Reset agents and state
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Initialize agent selection
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Reset relationships
        for agent in self.agents:
            self.stakeholders[agent]["relationships"] = np.zeros(self.n_agents, dtype=int)

    def step(self, action):
        """Execute one step in the environment
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return 

        agent = self.agent_selection

        self._cumulative_rewards[agent] = 0
        # Process action and get reward
        reward = self._attempt_engagement(agent, self.agents[action])

        # Update rewards
        self.rewards[agent] = reward

        # Check termination conditions
        terminated = self._check_termination()
        if terminated:
            self.terminations = {agent: True for agent in self.agents}

        self._accumulate_rewards()

        # Move to next agent
        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        """Return observation for the specified agent."""
        # format stakeholder attributes into a matrix
        stakeholder_attributes = []
        characteristics = ['position', 'power', 'knowledge', 'urgency', 'legitimacy']
        for agent in self.possible_agents:
            stakeholder_attributes.append([self.stakeholders[agent][char] for char in characteristics])
        # create the observation
        observation = {
            "observation": np.array(stakeholder_attributes), 
            "action_mask" : np.ones(self.n_agents, dtype=np.int8) # action_mask does nothing right now
        }
        return observation

    def _find_primary_agent(self):
        """Find the primary agent in stakeholders."""
        for agent, attrs in self.stakeholders.items():
            if attrs["is_primary"]:
                return agent
        raise ValueError("No primary agent found in the stakeholder configuration.")

    def _find_negotiator(self):
        """Find the negotiator in stakeholders."""
        for agent, attrs in self.stakeholders.items():
            if attrs["is_negotiator"]:
                return agent
        return None

    def _attempt_engagement(self, agent, target_agent):
        """Attempt engagement between agents."""
        if agent == target_agent:
            return -1  # Penalty for trying to engage with self

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
            return -1

    def _calculate_engagement_probability(self, agent_state, target_state):
        """Calculate probability of successful engagement."""
        p = 0.5  # Base probability

        # Adjust based on positions
        if agent_state["position"] == target_state["position"]:
            p += 0.2
        elif agent_state["position"] * target_state["position"] < 0:  # Opposing positions
            p -= 0.2

        # Adjust based on power dynamics
        if agent_state["power"] > target_state["power"]:
            p += 0.1
        elif agent_state["power"] < target_state["power"]:
            p -= 0.1

        # Bound probability
        return max(0.1, min(0.9, p))

    def _calculate_reward(self, agent, target_agent):
        """Calculate reward for successful engagement."""
        agent_state = self.stakeholders[agent]
        target_state = self.stakeholders[target_agent]

        reward = 0

        # Reward for engaging with influential stakeholders
        if target_state["power"] > 1 or target_state["knowledge"] > 1:
            reward += 1

        # Special reward for engaging with primary agent
        if target_agent == self.primary_agent:
            if sum(agent_state["relationships"]) >= 2:
                reward += 5  # Bonus for having multiple relationships
            else:
                reward += 1

        # Penalty for engaging with low-power stakeholders
        if target_state["power"] == 0:
            reward -= 1

        return reward

    def _check_termination(self):
        """Check if environment should terminate."""
        if self.agent_selection == self.negotiator:
            return self.stakeholders[self.negotiator]["relationships"][
                self.agents.index(self.primary_agent)] == 1
        return False

    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human":
            print("\n" + "="*50)
            print("Current Environment State:")
            print("-"*50)
            print(f"Current Agent: {self.agent_selection}")
            for agent, state in self.stakeholders.items():
                print(f"\n{agent}:")
                print(f"  Position: {state['position']}")
                print(f"  Power: {state['power']}")
                print(f"  Knowledge: {state['knowledge']}")
                print(f"  Urgency: {state['urgency']}")
                print(f"  Legitimacy: {state['legitimacy']}")
                print(f"  Relationships: {state['relationships']}")
            print("="*50)
    def close(self):
        pass

if __name__ == "__main__":
    # Test the environment
    env = NegotiationEnv(render_mode="human")
    env.reset()

    print("\nInitial State:")
    env.render()

    for step in range(5):
        print(f"\nStep {step + 1}")
        agent = env.agent_selection
        obs = env.observe(agent)
        action_mask = obs["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]
        action = np.random.choice(valid_actions)

        env.step(action)
        obs, reward, terminated, truncated, info = env.last()

        print(f"Agent: {agent}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")

        if terminated or truncated:
            print("\nEnvironment terminated!")
            break

    env.close()