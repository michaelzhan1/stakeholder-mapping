from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from single_agent_negotiation import NegotationEnv
import numpy as np
import pandas as pd

# read csv for data first
fname = "test.csv"
df = pd.read_csv(fname, header=None, names=["Stakeholder", "Power", "Urgency", "Knowledge", "Legitimacy", "Position"])
df['Total'] = df[['Power', 'Urgency', 'Knowledge', 'Legitimacy', 'Position']].sum(axis=1)

total_powers = df['Total'].to_list()

set_random_seed(0)

# wrap in DummyVecEnv to make it compatible with stable_baselines3
# then wrap in a VecMonitor to record statistics
# then wrap in a VecVideoRecorder to record video
env = make_vec_env(NegotationEnv, n_envs=1, env_kwargs={'render_mode': 'ansi', 'power_array': total_powers})

# # define, learn, and save model
model = PPO("MlpPolicy", env, verbose=1, gamma=1)
model.learn(total_timesteps=20000)
model.save("ppo_model")  # saves model to ppo_model.zip

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_model")  # load model from zip

# render final output of model
obs = env.reset()
last_obs = np.copy(env.get_attr('state')[0])
env.render()
while True:
    # run the final strategy
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    
    if done:
        print("Final state: ")
        print(info[0]['terminal_observation'])
        env.close() # make sure to close, otherwise errors occur with gymnasium
        break
    else:
        # only print successful actions
        if (last_obs != obs).any():
            env.render()
            last_obs = obs
