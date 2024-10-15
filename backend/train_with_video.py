import gymnasium as gym
import moviepy

from stable_baselines3 import PPO
from grid_env import GridEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor

# wrap in DummyVecEnv to make it compatible with stable_baselines3
# then wrap in a VecMonitor to record statistics
# then wrap in a VecVideoRecorder to record video
env = make_vec_env(GridEnv, n_envs=1, env_kwargs={'render_mode': 'rgb_array'})
env = VecMonitor(env)
env = VecVideoRecorder(env, video_folder="./videos", record_video_trigger=lambda x: x % 2000 == 0, video_length=200, name_prefix="grid")

# define, learn, and save model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("ppo_model")  # saves model to ppo_model.zip

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_model")  # load model from zip

# render final output of model
obs = env.reset()
while True:
    # run the final strategy
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    
    env.render()
    if dones:
        env.close() # make sure to close, otherwise errors occur with gymnasium
        break