import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper
from gymnasium.wrappers import TransformObservation
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from minigrid.core.constants import IDX_TO_OBJECT
from minigrid.core.actions import Actions



def make__MiniGrid(render_mode=None, fully_observable=False):
    env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode=render_mode)

    env = FullyObsWrapper(env) if fully_observable else env
    env = ImgObsWrapper(env)       # Get rid of the 'mission' field
    env = FilterObservations(env)  # Filter only objects data
    env = RemapActions(env)        # Remove unused actions

    # Normalize to [0,1] and reshape to (1, H, W)
    ob_shape = (1, *env.observation_space.shape)
    env = TransformObservation(env, lambda obs: obs.astype(np.float32).reshape(ob_shape) / max(IDX_TO_OBJECT), gym.spaces.Box(0., 1., ob_shape))

    # Default: A reward of "1 - 0.9 * (step_count / max_steps)" is given for success, and ‘0’ for failure.
    # Custom: env = gym.wrappersTransformReward(env, lambda r: -0.01 if r == 0 else 1)

    return env


class FilterObservations(gym.ObservationWrapper):
    """
    Filter obs state: [H, W, (objects, colors, state)] -> [H, W, objects]
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        space = env.observation_space
        self.observation_space = gym.spaces.Box(0, 255, (space.shape[0], space.shape[1]), space.dtype)

    def observation(self, observation):
        objects, colors, state = observation[:, :, 0], observation[:, :, 1], observation[:, :, 2]
        return objects

class RemapActions(ActionWrapper):
    """
    Get rid of "drop" and "done" actions
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            0: Actions.left,
            1: Actions.right,
            2: Actions.forward,
            3: Actions.pickup,
            4: Actions.toggle,
        }
        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def action(self, action):
        return int(self.action_map[action])



# Testing only
if __name__ == '__main__':
    import custom_gyms
    from lib.playground import play_episode

    env = gym.make('custom/MiniGrid', render_mode='human')
    obs, info = env.reset()
    episode = play_episode(env, policy=lambda obs: env.action_space.sample())
    print(episode)
    env.close()

