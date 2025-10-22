import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace


# Using the old gym!
import gym
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

class RescaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def observation(self, observation): # destroys the LazyFrames
        return np.array(observation).astype(np.float32) / 255.

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        assert skip > 0, 'Frame skip must be greater than 0 if used'
        self.skip = skip

    def step(self, action):
        total_reward = 0.
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_mario__old_gym(id='SuperMarioBros-1-1-v0', render_mode=None, max_episode_steps=None, ignore_deprecation_warnings=True):
    if ignore_deprecation_warnings:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="gym.envs.registration")
        warnings.filterwarnings("ignore", category=UserWarning, module="gym.utils.passive_env_checker")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")

    env = gym_super_mario_bros.make(id, max_episode_steps=max_episode_steps, render_mode=render_mode, apply_api_compatibility=True)
    # env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, 4)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)
    env = RescaleObservation(env)

    return env


# Testing only
if __name__ == '__main__':
    from lib.playground import play_episode
    from matplotlib import pyplot as plt

    env = make_mario__old_gym(render_mode='human', max_episode_steps=1000)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(obs.shape, info)

    # plot what the agent will see
    unstacked_obs = np.array(obs).reshape(-1, obs.shape[-1])
    plt.matshow(unstacked_obs, cmap='gray')
    plt.axis('off')
    plt.show()

    # collect an episode
    episode = play_episode(env, policy=lambda obs: env.action_space.sample())
    print(episode)
    env.close()

