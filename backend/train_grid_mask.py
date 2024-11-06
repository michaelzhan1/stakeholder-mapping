from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from grid_env import GridEnv

def mask_fn(env):
    return env.get_action_mask()

env = GridEnv()
env = ActionMasker(env, mask_fn)

# define, learn, and save model
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
model.learn(total_timesteps=20000)
model.save("ppo_model_masked")  # saves model to ppo_model.zip

del model # remove to demonstrate saving and loading

model = MaskablePPO.load("ppo_model_masked")  # load model from zip

# render final output of model
obs = env.reset()[0]
print("Starting final run")
env.render()
while True:
    # run the final strategy
    action, _states = model.predict(obs, deterministic=True, action_masks=env.get_action_mask())
    obs, rewards, dones, truncated, info = env.step(action.item())
    
    env.render()
    if dones:
        env.close() # make sure to close, otherwise errors occur with gymnasium
        break