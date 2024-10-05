import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

MAPS = {
    "4x4": ["0000", "1101", "0000", "1000"],
    "8x8": [
        "00000000",
        "00000000",
        "00010000",
        "00000100",
        "00010000",
        "01100010",
        "01001010",
        "00010000",
    ],
}

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 8}
    FREE = 0
    OBSTACLE = 1
    MOVES = {
        0: (-1, 0),  # UP
        1: (1, 0),  # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),  # RIGHT
    }

    def __init__(self):
        self.obstacles = [
            [0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
        ]

        self.nrow = len(self.obstacles)
        self.ncol = len(self.obstacles[0])

        self.action_space = spaces.Discrete(4) # define the number of possible actions/moves
        self.observation_space = spaces.Discrete(self.nrow * self.ncol) # define the number of possible states, where the agent is. this means each observation is a single int

        self.fig = None
        self.render_mode = 'ansi'
        self.fps = self.metadata["render_fps"]
    
    def reset(self, seed=0):
        super().reset(seed=seed)

        self.start_xy = (0, 0)
        self.goal_xy = (self.nrow - 1, self.ncol - 1)

        self.agent_xy = self.start_xy
        self.reward = self.calc_reward(*self.agent_xy)
        self.done = False
        self.agent_action = None
        self.n_iter = 0

        # self.render()

        return self.get_obs(), self.get_info()

    def step(self, action):
        self.agent_action = action
        row, col = self.agent_xy

        dx, dy = self.MOVES[action]

        new_row, new_col = row + dx, col + dy

        self.reward = self.calc_reward(new_row, new_col)

        if 0 <= new_row < self.nrow and 0 <= new_col < self.ncol and self.obstacles[new_row][new_col] != self.OBSTACLE:
            self.agent_xy = (new_row, new_col)

            if self.agent_xy == self.goal_xy:
                self.done = True
        
        self.n_iter += 1

        # self.render()
        return self.get_obs(), self.reward, self.done, False, self.get_info()

    def render(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                if (row, col) == self.agent_xy:
                    print("A", end="")
                elif (row, col) == self.goal_xy:
                    print("G", end="")
                elif self.obstacles[row][col] == self.OBSTACLE:
                    print("X", end="")
                else:
                    print(".", end="")
            print()
        print()
    
    ##### HELPER FUNCTIONS
    def calc_reward(self, x, y):
        if not (0 <= x < self.nrow and 0 <= y < self.ncol):
            return -1
        if self.obstacles[x][y] == self.OBSTACLE:
            return -1
        if (x, y) == self.goal_xy:
            return 1
        return 0

    def get_obs(self):
        x, y = self.agent_xy
        return x * self.ncol + y
    
    def get_info(self):
        return {
            "agent_xy": self.agent_xy,
            "goal_xy": self.goal_xy,
            "iter": self.n_iter,
        }
    
    def get_pos(self):
        return self.agent_xy