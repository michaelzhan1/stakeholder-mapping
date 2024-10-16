# Backend
This folder will contain (for now) code related to the RL backend of the project. Right now, the included files are:
- `testing.ipynb`: A notebook for various RL testing purposes
- `grid_env.py`: A custom Gym environment of a grid, where an agent must walk around obstacles to reach the end (maze solver)
- `single_agent_negotiation.py`: A custom environment of a negotiation environment. Here, only one agent can act, and each agent only has one attribute (a power attribute).
- `train.py`: The code used to actually learn and solve the environment (currently configured with the negotiation env)
- `train_with_video.py`: Same training code, but allows videos to be saved of rendered states.