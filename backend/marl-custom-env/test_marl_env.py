from env.stakeholder_mapping_env import NegotiationEnv

from pettingzoo.test import parallel_api_test, api_test

if __name__ == "__main__":
    env = NegotiationEnv()
    api_test(env, num_cycles=1_000_000, verbose_progress=True)