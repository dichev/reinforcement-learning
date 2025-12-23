from enum import IntEnum
import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.actions import Actions
from envs.wrappers import OneHotWrapper

class ActionRemap(IntEnum):
    left    = 0
    right   = 1
    forward = 2
    pickup  = 3
    toggle  = 4

class ObservationRemap(IntEnum): # encoded as channels
    wall  = 0
    door  = 1
    key   = 2
    goal  = 3
    agent = 4 # not visible in partial observations

USED_OBJECT_IDX = tuple(OBJECT_TO_IDX[remap.name] for remap in ObservationRemap)


def make__MiniGrid(render_mode=None, fully_observable=False, max_steps=None):
    env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode=render_mode, max_steps=max_steps)

    env = FullyObsWrapper(env) if fully_observable else env
    env = ImgObsWrapper(env)                                     # (X, Y, (objects, colors, state)) - Get rid of the 'mission' field
    env = FilterObservations(env)                                # (X, Y) - Filter only objects data
    env = TransposeWrapper(env)                                  # (H, W)
    env = OneHotWrapper(env, channels=USED_OBJECT_IDX)           # (obj_type, H, W) - Split each object into a separate channel (one-hot)
    env = RemapActions(env)                                      # Remove unused actions

    # Default: A reward of "1 - 0.9 * (step_count / max_steps)" is given for success, and ‘0’ for failure.
    # Custom: env = gym.wrappers.TransformReward(env, lambda r: -0.01 if r == 0 else 1)

    return env



class FilterObservations(gym.ObservationWrapper):
    """
    Filter obs state: [X, Y, (objects, colors, state)] -> [X, Y] for objects only
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        X, Y, C = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, max(USED_OBJECT_IDX), (X, Y), env.observation_space.dtype)

    def observation(self, observation):
        objects, colors, state = observation[:, :, 0], observation[:, :, 1], observation[:, :, 2]
        return objects


class TransposeWrapper(gym.ObservationWrapper):
    """
    Converts (X, Y) -> (H, W).
    """
    def __init__(self, env):
        super().__init__(env)
        w, h = self.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (h, w), dtype=np.uint8)

    def observation(self, obs):
        return obs.T


class RemapActions(ActionWrapper):
    """
    Get rid of "drop" and "done" actions
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            ActionRemap.left: Actions.left,
            ActionRemap.right: Actions.right,
            ActionRemap.forward: Actions.forward,
            ActionRemap.pickup: Actions.pickup,
            ActionRemap.toggle: Actions.toggle,
        }
        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def action(self, action):
        return int(self.action_map[action])



# Testing only
if __name__ == '__main__':
    import custom_gyms
    from lib.playground import play_episode

    env = gym.make('custom/MiniGrid', render_mode='human', fully_observable=False, max_steps=2)
    ob, info = env.reset()
    episode = play_episode(env, policy=lambda obs: env.action_space.sample())
    print(episode)
    env.close()

