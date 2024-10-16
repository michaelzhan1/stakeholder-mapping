import supersuit as ss

from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor

from pettingzoo.mpe import simple_adversary_v3

def train_parallel_env(env_fn, steps=10000, seed=0, **env_kwargs):
    env = env_fn.parallel_env(**env_kwargs)

    env.reset(seed=seed)
    
    print(f"Training {env.num_agents} agents in parallel")

    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')

    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=steps)

    model.save("ppo_multiagent")

    print(f"Finished training")

    env.close()

def main():
    env_fn = simple_adversary_v3
    train_parallel_env(env_fn, steps=10000, seed=0)



if __name__ == "__main__":
    main()