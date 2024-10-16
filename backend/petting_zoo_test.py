import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor

from pettingzoo.mpe import simple_adversary_v3

# wrapper function to help train the parallel environment
def train_parallel_env(env_fn, steps=10000, seed=0, render_mode=None, **env_kwargs):
    # initialize parallel environment
    env = env_fn.parallel_env(render_mode=render_mode, **env_kwargs)
    env.reset(seed=seed)
    
    print(f"Training {env.num_agents} agents in parallel")

    # in this case only (likely not for our env), different agents have diff observation spaces, so we should pad them to be equal
    env = ss.multiagent_wrappers.pad_observations_v0(env)

    # convert pettingzoo env to vec env
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # convert vec env to stable baselines vec env
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')

    # try to record video
    # env = VecMonitor(env)  # <- we may have needed this in the single agent case, but including this here breaks the code
    env = VecVideoRecorder(env, video_folder="./videos", record_video_trigger=lambda x: x % 2000 == 0, video_length=200, name_prefix="grid")

    # run the model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)
    model.save("ppo_multiagent")

    print(f"Finished training")

    env.close()

def evaluate(env_fn, render_mode, **env_kwargs):
    # set up environtment
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env.reset(seed=0)
    
    model = PPO.load("ppo_multiagent")

    # run the model's predictions
    for agent in env.agent_iter():
        obs, reward, done, truncated, info = env.last()
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        if done or truncated:
            break
        env.step(action)
    env.close()

    # for final recording, see: https://github.com/Farama-Foundation/PettingZoo/issues/698



def main():
    env_fn = simple_adversary_v3
    train_parallel_env(env_fn, steps=20000, seed=0, render_mode="rgb_array")
    evaluate(env_fn, render_mode="human")



if __name__ == "__main__":
    main()