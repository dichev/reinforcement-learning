import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from lib.playground import play_episode



class FilterObservations(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        space = env.observation_space
        self.observation_space = gym.spaces.Box(0, 255, (space.shape[0], space.shape[1]), space.dtype)

    def observation(self, observation):
        objects, colors, state = observation[:, :, 0], observation[:, :, 1], observation[:, :, 2]
        return objects



def make__MiniGrid(render_mode=None):
    env = gym.make("MiniGrid-LavaCrossingS9N3-v0", render_mode=render_mode)
    # env = gym.make("MiniGrid-Empty-Random-5x5-v0", render_mode=render_mode)
    # env = gym.make("MiniGrid-Empty-5x5-v0", render_mode=render_mode)
    env = FullyObsWrapper(env)     # full observability
    env = ImgObsWrapper(env)       # Get rid of the 'mission' field
    env = FilterObservations(env)  # Filter only objects data

    # todo: normalize the state? / add small noise? / reward on each step?
    return env



# Testing only
if __name__ == '__main__':
    import custom_gyms
    env = gym.make('custom/MiniGrid', render_mode='human')
    episode = play_episode(env, policy=lambda obs: env.action_space.sample())
    print(episode)
    print(episode.observations[-1])
    env.close()

