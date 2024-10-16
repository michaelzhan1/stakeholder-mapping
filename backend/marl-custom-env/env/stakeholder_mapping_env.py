# MARL Custom Environment for Stakeholder Mapping
import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "stakeholder_mapping_v0",
    }

    def __init__(self):
        """The init method takes in environment arguments."""
        self.possible_agents = ["stakeholder1", "stakeholder2", "primary_stakeholder"]
        self.timestep = None
        self.coalitions = None

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - environment variables
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.coalitions = []

        # observations will be existing coalitions - no initial coalitions
        # agent observes its own coalition
        observations = {a: {} for a in self.agents}

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - existing coalitions
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        # Execute actions
        stakeholder1_action = actions["stakeholder1"]
        stakeholder2_action = actions["stakeholder2"]
        primarystakeholder_action = actions["primary_stakeholder"]

        if stakeholder1_action == 1:
            for coalition in self.coalitions:
                added = False
                if "stakeholder1" in coalition:
                    coalition.add("stakeholder2")
                    added = True
            if added == False:
                coalition.add("stakeholder1")
                coalition.add("stakeholder2")
            self.coalitions += ["stakeholder1", "stakeholder2"]
        elif stakeholder1_action == 2:
            self.coalitions += ["stakeholder1", "primary_stakeholder"]

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if ["stakeholder1", "primary_stakeholder"] in self.coalitions:
            rewards = {"stakeholder1": 1, "primary_stakeholder": 1, "stakeholder2":-1}
            terminations = {a: True for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"stakeholder1": 0, "primary_stakeholder": 0, "stakeholder2":0}
            truncations = {"stakeholder1": True, "primary_stakeholder": True, "stakeholder2":True}
        self.timestep += 1

        # Get observations
        observations = {
            a: (
                self.coalitions
            )
            for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        print(self.coalitions)
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([7 * 7] * 3)
        #return self.observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)
        # return self.action_spaces[agent]

    def add_to_coalitions(self, stakeholer1, stakeholder2):
        for coalition in self.coalitions:
            added = False
            if "stakeholder1" in coalition:
                coalition.add("stakeholder2")
                added = True
        if added == False:
            coalition.add("stakeholder1")
            coalition.add("stakeholder2")

if __name__ == "__main__":
    env = CustomEnvironment()
    env.reset()
    for _ in range(10):
        obs, rewards, term, trunc, infos = env.step({"stakeholder1":1, 
                                                     "stakeholder2":0, 
                                                     "primary_stakeholder":0})
        print(obs)
        # env.render()
