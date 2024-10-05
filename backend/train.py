import gymnasium as gym

from stable_baselines3 import PPO
from custom_env import GridEnv
from stable_baselines3.common.env_util import make_vec_env

# wrap in DummyVecEnv to make it compatible with stable_baselines3
env = make_vec_env(GridEnv, n_envs=1)

# define, learn, and save model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("ppo_model")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_model")

# render final output of model
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        print("Note: SB3 resets agent position on finish.")
    env.render()
    if dones:
        break