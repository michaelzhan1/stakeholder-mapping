import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np
from stable_baselines3.common.env_checker import check_env

PIXEL_SIZE = 50

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

    def __init__(self, render_mode='ansi'):
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
        self.observation_space = spaces.Discrete(self.nrow * self.ncol) # define the number of possible states, where the agent is. this case means each observation is a single int

        # define a render mode (otherwise sb3 is unhappy)
        self.render_mode = render_mode
    
    def reset(self, seed=0, **kwargs):
        super().reset(seed=seed)
        np.random.seed(seed)

        self.start_xy = (0, 0)
        self.goal_xy = (self.nrow - 1, self.ncol - 1)

        self.agent_xy = self.start_xy
        self.reward = self.calc_reward(*self.agent_xy)
        self.done = False
        self.agent_action = None
        self.n_iter = 0

        return self.get_obs(), self.get_info()

    def step(self, action):
        self.agent_action = action
        row, col = self.agent_xy

        dx, dy = self.MOVES[action]

        if (np.random.rand() < 0.5):
            self.n_iter += 1
            self.reward = -0.1
            return self.get_obs(), self.reward, self.done, False, self.get_info()

        new_row, new_col = row + dx, col + dy

        self.reward = self.calc_reward(new_row, new_col)

        if 0 <= new_row < self.nrow and 0 <= new_col < self.ncol and self.obstacles[new_row][new_col] != self.OBSTACLE:
            self.agent_xy = (new_row, new_col)

            if self.agent_xy == self.goal_xy:
                self.done = True
        
        self.n_iter += 1

        return self.get_obs(), self.reward, self.done, False, self.get_info()

    def render(self):
        if self.render_mode == "ansi":
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
        elif self.render_mode == 'human':
            # use cv2 to render the grid
            # make a 3-channel image where each cell is PIXEL_SIZE x PIXEL_SIZE
            img = np.zeros((self.nrow * PIXEL_SIZE, self.ncol * PIXEL_SIZE, 3), dtype=np.uint8)
            for row in range(self.nrow):
                for col in range(self.ncol):
                    if (row, col) == self.agent_xy:
                        color = (0, 0, 255)
                    elif (row, col) == self.goal_xy:
                        color = (0, 255, 0)
                    elif self.obstacles[row][col] == self.OBSTACLE:
                        color = (0, 0, 0)
                    else:
                        color = (255, 255, 255)
                    img[row * PIXEL_SIZE:(row + 1) * PIXEL_SIZE, col * PIXEL_SIZE:(col + 1) * PIXEL_SIZE] = color
            img = cv2.resize(img, (self.ncol * PIXEL_SIZE * 2, self.nrow * PIXEL_SIZE * 2), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("Grid", img)
            cv2.waitKey(100)
        elif self.render_mode == 'rgb_array':
            # use cv2 to render the grid
            # make a 3-channel image where each cell is PIXEL_SIZE x PIXEL_SIZE
            img = np.zeros((self.nrow * PIXEL_SIZE, self.ncol * PIXEL_SIZE, 3), dtype=np.uint8)
            for row in range(self.nrow):
                for col in range(self.ncol):
                    if (row, col) == self.agent_xy:
                        color = (0, 0, 255)
                    elif (row, col) == self.goal_xy:
                        color = (0, 255, 0)
                    elif self.obstacles[row][col] == self.OBSTACLE:
                        color = (0, 0, 0)
                    else:
                        color = (255, 255, 255)
                    img[row * PIXEL_SIZE:(row + 1) * PIXEL_SIZE, col * PIXEL_SIZE:(col + 1) * PIXEL_SIZE] = color
            img = cv2.resize(img, (self.ncol * PIXEL_SIZE * 2, self.nrow * PIXEL_SIZE * 2), interpolation=cv2.INTER_NEAREST)
            return img

    def close(self):
        cv2.destroyAllWindows()

    ##### HELPER FUNCTIONS
    def calc_reward(self, x, y):
        if not (0 <= x < self.nrow and 0 <= y < self.ncol):
            return -1
        if self.obstacles[x][y] == self.OBSTACLE:
            return -1
        if (x, y) == self.goal_xy:
            return 1
        return -0.1

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
    
    def get_action_mask(self):
        mask = np.ones(4)
        if self.agent_xy[0] == 0:
            mask[0] = 0
        if self.agent_xy[0] == self.nrow - 1:
            mask[1] = 0
        if self.agent_xy[1] == 0:
            mask[2] = 0
        if self.agent_xy[1] == self.ncol - 1:
            mask[3] = 0
        return mask
    
if __name__ == "__main__":
    env = GridEnv()
    check_env(env)
