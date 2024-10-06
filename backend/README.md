# Backend
This folder will contain (for now) code related to the RL backend of the project. Right now, the included files are:
- `testing.ipynb`: A notebook for various RL testing purposes
- `custom_env.py`: A custom Gym environment of a grid, where an agent must walk around obstacles to reach the end (maze solver)
- `train.py`: The code used to actually learn and solve the environment
  - This file also allows for rendered videos to be saved of the agent in the environment, which are saved under a created `videos/` subfolder.