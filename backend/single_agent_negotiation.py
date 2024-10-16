import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np
import random

class NegotationEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    # init the environment
    def __init__(self, render_mode='ansi'):
        # define stakeholders and powers. let 0 be the agent, and n-1 be the goal
        self.n = 3
        self.powers = [1, 1, 5]

        # actions: talk to each person (including self)
        self.action_space = spaces.Discrete(self.n)

        # observations: true if we have convinced them, false otherwise
        self.observation_space = spaces.MultiBinary(self.n)

        # define a render mode (otherwise sb3 is unhappy)
        self.render_mode = render_mode
    
    # reset the environment to the start
    def reset(self, seed=0):
        super().reset(seed=seed)

        # reset the state to all zeros except for the agent
        self.state = np.zeros(self.n)
        self.state[0] = 1

        # reset done state 
        self.done = False

        # track iteration number for info logging
        self.n_iter = 0

        return self.state, self.get_info()

    # take a step
    def step(self, action):
        # set base reward
        self.reward = -1

        # try to convince the person
        self_power = self.powers[0]
        other_power = self.powers[action]

        # on successful convince, update state and reward
        prob = self_power / (self_power + other_power)
        if (random.random() < prob ** 2):
            self.state[action] = 1
            self.reward += self.powers[action]

        # check if we have convinced the target
        if self.state[self.n - 1]:
            self.done = True
        
        self.n_iter += 1

        return self.state, self.reward, self.done, False, self.get_info()

    # render the environment
    def render(self):
        if self.render_mode == "ansi":
            print(self.state)
        elif self.render_mode == 'human':
            pass
        elif self.render_mode == 'rgb_array':
            pass

    def close(self):
        cv2.destroyAllWindows()

    # Helper function to get info
    def get_info(self):
        return {
            "state": self.state,
            "iter": self.n_iter,
        }