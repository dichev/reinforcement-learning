import gymnasium as gym
from gymnasium.envs.registration import register
from envs.wrappers import StepPenaltyWrapper, DiscreteOneHotWrapper, DiscountedRewardWrapper
from lib.playground import play_episode


def make__FrozenLake_OneHot_DiscountedReward(render_mode=None, is_slippery=False, gamma=0.99):
    env = gym.make("FrozenLake-v1", render_mode=render_mode, is_slippery=is_slippery)
    env = DiscountedRewardWrapper(env, gamma=gamma)
    env = DiscreteOneHotWrapper(env)
    return env


env_id = 'custom/FrozenLake-OneHot-StepPenalty'
if env_id not in gym.envs.registry:
    register(id=env_id, entry_point='src.envs.custom_games:make__FrozenLake_OneHot_DiscountedReward')



# Testing only
if __name__ == '__main__':
    env = gym.make('custom/FrozenLake-OneHot-DiscountedReward', render_mode='human')
    episode = play_episode(env, policy=lambda obs: env.action_space.sample())
    print(episode)
    env.close()

