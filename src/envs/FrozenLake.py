import gymnasium as gym
from envs.wrappers import DiscreteOneHotWrapper, DiscountedRewardWrapper


def make__FrozenLake_OneHot_DiscountedReward(render_mode=None, is_slippery=False, gamma=0.99):
    env = gym.make("FrozenLake-v1", render_mode=render_mode, is_slippery=is_slippery)
    env = DiscountedRewardWrapper(env, gamma=gamma)
    env = DiscreteOneHotWrapper(env)
    return env



# Testing only
if __name__ == '__main__':
    import custom_gyms
    from lib.playground import play_episode

    env = gym.make('custom/FrozenLake_OneHot_DiscountedReward', render_mode='human')
    episode = play_episode(env, policy=lambda obs: env.action_space.sample())
    print(episode)
    env.close()

