import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, TransformObservation
from vizdoom.gymnasium_wrapper.gymnasium_env_defns import VizdoomScenarioEnv
from vizdoom import wads, scenarios_path
from os import path

# modified "VizdoomMyWayHome-v0" config: https://github.com/Farama-Foundation/ViZDoom/blob/master/scenarios/my_way_home.cfg
CONFIG_PATH = path.join(path.dirname(path.abspath(__file__)), 'configs/doom__my_way_home_custom.cfg')
WAD_PATH    = path.join(scenarios_path, 'my_way_home.wad')


def make_doom(render_mode=None, episode_timeout=None):
    env = VizdoomScenarioEnv(CONFIG_PATH, frame_skip=1, max_buttons_pressed=1, render_mode=render_mode)
    env.game.set_doom_scenario_path(WAD_PATH)

    # override the scenario config
    if episode_timeout:
        env.game.set_episode_timeout(episode_timeout)

    # pick only the screen observation and resize/rescale
    env = gym.wrappers.TransformObservation(env, lambda obs: obs['screen'], env.observation_space['screen'])
  # env = GrayscaleObservation(env) # gray mode was set in the config file
    env = ResizeObservation(env, (84, 84))
    env = TransformObservation(env, lambda obs: obs.astype(np.float32).reshape(1, 84, 84) / 255., gym.spaces.Box(0., 1., (1, 84, 84)) )

    return env


# Testing only
if __name__ == '__main__':
    import custom_gyms
    from lib.playground import play_episode
    from matplotlib import pyplot as plt

    env = gym.make('custom/DoomMaze', render_mode='human', episode_timeout=200)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(obs, info)

    # plot what the agent will see
    c, h, w = obs.shape
    plt.figure(figsize=(w/100, h/100), dpi=100)
    plt.imshow(obs.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()

    # collect an episode
    episode = play_episode(env, policy=lambda obs: env.action_space.sample())
    print(episode)
    env.close()
