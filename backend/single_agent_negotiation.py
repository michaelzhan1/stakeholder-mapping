import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np
import random

class NegotationEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    # init the environment
    def __init__(self, power_array=None, render_mode='ansi'):
        # define stakeholders and powers. let 0 be the agent, and n-1 be the goal
        if power_array is None:
            self.powers = np.array([1, 5, 3, 10])
        else:
            self.powers = np.array(power_array)
        self.n = len(self.powers)

        # actions: talk to each person (including self)
        self.action_space = spaces.Discrete(self.n)

        # observations: true if we have convinced them, false otherwise
        self.observation_space = spaces.MultiBinary(self.n)

        # define a render mode (otherwise sb3 is unhappy)
        self.render_mode = render_mode
    
    # reset the environment to the start
    def reset(self, seed=0):
        super().reset(seed=seed)
        random.seed(seed)

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

        # check if we've already convinced the target
        if self.state[action] == 0:
            # on successful convince, update state and reward
            total_power = self.powers[self.state == 1].sum()
            prob = total_power / (total_power + self.powers[action])
            if (random.random() < prob ** 1.5):  # let the probability scale slower than the reward to incentivize taking intermediate steps
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

    # close a CV2 render environment
    def close(self):
        cv2.destroyAllWindows()

    # Helper function to get info
    def get_info(self):
        return {
            "state": self.state,
            "iter": self.n_iter,
        }
