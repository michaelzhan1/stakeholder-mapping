import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
import networkx as nx
import math
from pettingzoo.test import api_test

from enum import Enum

# # Define global weights
# w_position = 0.5
# w_power = 5
# w_knowledge = 0.5
# w_urgency = 6
# w_legitimacy = 8
# w_distance = 14

# # The probability of success given you have max(power, urgency, legitimacy) but no relationships
# prob_success_alone = 0.6
# standardisation_factor = (2 * w_power + w_urgency + w_legitimacy) / prob_success_alone


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
    
    def __init__(self, stakeholder_matrix=None, render_mode=None, weights=None):
        # init agent info
        self.stakeholders = self._init_stakeholders(stakeholder_matrix)
        self.n_agents = len(self.stakeholders)
        self.agents = [f"agent_{idx + 1}" for idx in range(self.n_agents)]
        self.possible_agents = self.agents[:]

        # Define global weights
        self.w_position = 0.5
        self.w_power = 5
        self.w_knowledge = 0.5
        self.w_urgency = 6
        self.w_legitimacy = 8
        self.w_distance = 14
        self.prob_success_alone = 0.6
        self.direct = 0.4
        self.indirect = 0.1
        

        # parse weights ## FOR EVALUATION
        if weights:
            if 'w_position' in weights.keys():
                self.w_position = weights['w_position']
            if 'w_power' in weights.keys():
                self.w_power = weights['w_power']
            if 'w_knowledge' in weights.keys():
                self.w_knowledge = weights['w_knowledge']
            if 'w_urgency' in weights.keys():
                self.w_urgency = weights['w_urgency']
            if 'w_legitimacy' in weights.keys():
                self.w_legitimacy = weights['w_legitimacy']
            if 'w_distance' in weights.keys():
                self.w_distance = weights['w_distance']
            if 'prob_success_alone' in weights.keys():
                self.prob_success_alone = weights['prob_success_alone']
            if 'direct' in weights.keys():
                self.direct = weights['direct']
            if 'indirect' in weights.keys():
                self.indirect = weights['indirect']

        self.standardisation_factor = (2 * self.w_power + self.w_urgency 
            + self.w_legitimacy) / self.prob_success_alone


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

        # Initialize network graph for coalition tracking
        # self.graph = nx.Graph()
        # self.graph.add_nodes_from(self.agents)


    
    def reset(self, seed=0, **kwargs):
        np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.agent_selection = self._agent_selector.reset()

        # track metrics
        self.primary_steps = 0
        self.final_prob_success = 0.05
        self.metrics = None

        # track agent actions
        self.agent_actions = {agent: { 'actions':[] } for agent in self.agents} 

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

        # # track agent's actions:
        self.agent_actions[agent]['actions'].append(action)

        # track primary agent's steps
        if agent == self.primary:
            self.primary_steps += 1

        terminated = self._check_termination()
        if terminated:
            for i in range(self.n_agents):
                self.terminations[self.idx_to_agent[i]] = True

            # evaluation metric values
            self.metrics = [self.primary_steps, self.final_prob_success]
            self.infos[agent]['metrics'] = self.metrics
        
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
        # always make target actions fail
        if agent == self.target:
            return Outcome.FAILURE

        if agent == recipient:
            return Outcome.SELF
        if self.stakeholders[agent]['relationships'][self.agent_to_idx[recipient]] == 1:
            return Outcome.EXISTING
        
        if np.random.rand() < self._calculate_probability(agent, recipient):
            self.stakeholders[agent]['relationships'][self.agent_to_idx[recipient]] = 1
            self.stakeholders[recipient]['relationships'][self.agent_to_idx[agent]] = 1
            return Outcome.SUCCESS
        return Outcome.FAILURE
    
    def _calculate_prob_helper(self, recipient, stakeholder):  
        """Calculate the reward of the recipient got from accepting a relationship from a stakeholder"""
        stakeholder_info = self.stakeholders[stakeholder]
        recipient_info = self.stakeholders[recipient]

        reward = (- self.w_position * abs(stakeholder_info['position'] - recipient_info['position']) +      
            self.w_power * stakeholder_info['power'] +
            self.w_knowledge * stakeholder_info['knowledge'] +
            self.w_urgency * stakeholder_info['urgency'] +
            self.w_legitimacy * stakeholder_info['legitimacy'] )
        
        if recipient == self.primary:
            # d = nx.shortest_path_length(self.graph, agent, self.target)
            d = self._calculate_distance_to_target(stakeholder)
            reward += self.w_distance * (1/ (1 + d)) # d is distance between individual and target

        return reward

    def _calculate_probability(self, agent, recipient):

        agent_idx = self.agent_to_idx[agent]
        coalition_reward = 0
        coalition_prob = 0

        # Stakeholders directly connected to the agent (degree = 1)
        direct_neighbors = [
            self.idx_to_agent[i]
            for i, relationship in enumerate(self.stakeholders[agent]['relationships'])
            if relationship == 1 and i != agent_idx
        ]

        # Stakeholders indirectly connected (degree = 2)
        indirect_neighbors = set()
        for neighbor in direct_neighbors:
            neighbor_idx = self.agent_to_idx[neighbor]
            for i, relationship in enumerate(self.stakeholders[neighbor]['relationships']):
                if relationship == 1 and i != agent_idx and self.idx_to_agent[i] != neighbor:
                    indirect_neighbors.add(self.idx_to_agent[i])

        # Combine direct and indirect neighbors
        all_neighbors = set(direct_neighbors).union(indirect_neighbors)

        # Calculate the weighted reward for the recipient
        coalition_reward += self._calculate_prob_helper(recipient, agent)  # e^0 = 1

        # Calculate the weighted reward for direct and indirect neighbors
        for stakeholder in all_neighbors:
            degree_of_separation = 1 if stakeholder in direct_neighbors else 2
            # weight = math.exp(-degree_of_separation)
            weight = self.direct if degree_of_separation == 1 else self.indirect
            individual_reward = self._calculate_prob_helper(recipient, stakeholder)
            coalition_reward += weight * individual_reward

        # Convert coalition reward into a probability
        standardized_reward = (coalition_reward / self.standardisation_factor) ** 1.2
        if standardized_reward <= 0.05:
            standardized_reward = 0.05
        elif standardized_reward > 1:
            standardized_reward = 1 

        # updates success probability for evaluation metric
        if (agent == self.primary) and (recipient == self.target):
            self.final_prob_success = standardized_reward

        return standardized_reward

    def _calculate_reward_individual(self, agent, stakeholder):
        """Calculate the base value of a stakeholder based on their attributes."""

        stakeholder_info = self.stakeholders[stakeholder]

        reward = ( self.w_power * stakeholder_info['power'] +
            self.w_knowledge * stakeholder_info['knowledge'] +
            self.w_urgency * stakeholder_info['urgency'] + 
            self.w_legitimacy * stakeholder_info['legitimacy'] )
        
        if agent == self.primary:
            # d = nx.shortest_path_length(self.graph, agent, self.target)
            d = self._calculate_distance_to_target(stakeholder)
            reward += self.w_distance * (1/ (1 + d)) # d is distance between individual and target
        
        return reward
    
    def _calculate_distance_to_target(self, recipient): 
        """ Computes the degrees of separation between the recipient and the target. If the recipient is the target, the degree of separation is 0. """ 
        # Get the target agent index 
        target_idx = self.agent_to_idx[self.target] 
        recipient_idx = self.agent_to_idx[recipient] 
        
        if recipient_idx == target_idx: 
            return 0 # The recipient is the target 
            
        # Initialize BFS 
        visited = set() 
        queue = [(recipient_idx, 0)] 
        
        # Start with the recipient, at degree 0
        while queue: 
            current_idx, degree = queue.pop(0) 
            
            # Mark the current agent as visited 
            visited.add(current_idx) 
            # Check all relationships of the current agent 
            for neighbor_idx, relationship in enumerate(self.stakeholders[self.idx_to_agent[current_idx]]['relationships']): 
                if relationship == 1 and neighbor_idx not in visited: 
                    if neighbor_idx == target_idx: 
                        return degree + 1 # Found the target 
                    queue.append((neighbor_idx, degree + 1)) 
        
        return float('inf') # Target is not reachable
    
    def _calculate_reward_coalition(self, agent, recipient):
        """
        Computes the coalition reward for the recipient's coalition.
        
        Focuses only on stakeholders within 1 or 2 degrees of separation.
        """
        recipient_idx = self.agent_to_idx[recipient]
        coalition_reward = 0

        # Stakeholders directly connected to the recipient (degree = 1)
        direct_neighbors = [
            self.idx_to_agent[i]
            for i, relationship in enumerate(self.stakeholders[recipient]['relationships'])
            if relationship == 1 and i != recipient_idx
        ]

        # Stakeholders indirectly connected (degree = 2)
        indirect_neighbors = set()
        for neighbor in direct_neighbors:
            neighbor_idx = self.agent_to_idx[neighbor]
            for i, relationship in enumerate(self.stakeholders[neighbor]['relationships']):
                if relationship == 1 and i != recipient_idx and self.idx_to_agent[i] != neighbor:
                    indirect_neighbors.add(self.idx_to_agent[i])

        # Combine direct and indirect neighbors
        all_neighbors = set(direct_neighbors).union(indirect_neighbors)

        # Calculate the weighted reward for the recipient
        coalition_reward += self._calculate_reward_individual(agent, recipient)  # e^0 = 1

        # Calculate the weighted reward for direct and indirect neighbors
        for stakeholder in all_neighbors:
            degree_of_separation = 1 if stakeholder in direct_neighbors else 2
            # weight = math.exp(-degree_of_separation)
            weight = self.direct if degree_of_separation == 1 else self.indirect
            individual_reward = self._calculate_reward_individual(agent, stakeholder)
            coalition_reward += weight * individual_reward

        return coalition_reward

    def _calculate_reward(self, agent, recipient, outcome):
        match outcome:
            case Outcome.SUCCESS:
                return self._calculate_reward_coalition(agent, recipient)
            case Outcome.EXISTING:
                return -3
            case Outcome.SELF:
                return -1
            case Outcome.FAILURE:
                return -0.1
    
    def _check_termination(self):
        # truncate if primary takes more than 100 actions
        if self.primary_steps > 100:
            return True
        # end if primary engages target
        else:
            return self.stakeholders[self.primary]['relationships'][self.agent_to_idx[self.target]] == 1


if __name__ == "__main__":
    env = NegotiationEnv()
    api_test(env, num_cycles=1000, verbose_progress=True)
