import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, ClipReward
from envs.wrappers import FireResetEnv


def make_atari(id, scale_obs=True, clip_rewards=True, fire_reset=True, **kwargs):
    env = gym.make(id, frameskip=1, repeat_action_probability=0, **kwargs)
    env = AtariPreprocessing(env,
        noop_max = 30,
        frame_skip = 4,
        screen_size = 84,
        terminal_on_life_loss = False,
        grayscale_obs = True, grayscale_newaxis = False,
        scale_obs = scale_obs,  # it limits the memory optimization benefits of FrameStack Wrapper.
    )
    if fire_reset:
        env = FireResetEnv(env)

    if clip_rewards:
        env = ClipReward(env, -1, 1)

    env = FrameStackObservation(env, stack_size=4)
    return env



if __name__ == '__main__':
    from lib.playground import play_episode
    import envs.custom_gyms
    env = gym.make('custom/Pong', render_mode='human')
    obs, _ = env.reset()
    print('Obs:', obs.shape)
    episode = play_episode(env, policy=lambda obs: env.action_space.sample())
    print(episode)
    # env.close()
