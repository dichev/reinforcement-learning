import numpy as np
import gymnasium as gym


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    """
    Converts discrete observation to one-hot vector.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        onehot = np.zeros(self.observation_space.shape, dtype=np.float32)
        onehot[observation] = 1.0
        return onehot


class DiscountedRewardWrapper(gym.RewardWrapper):
    """
    Applies a discount factor to the reward at each step based on
    """

    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        discounted_reward = (self.gamma ** self._steps) * reward
        self._steps += 1
        return obs, discounted_reward, terminated, truncated, info



class StepPenaltyWrapper(gym.Wrapper):
    """
    Subtracts a small penalty at each step.
    """

    def __init__(self, env, step_penalty=-0.01):
        super().__init__(env)
        self.step_penalty = step_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            reward = self.step_penalty

        return obs, reward, terminated, truncated, info



class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    """
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Converts image observation shape from (H, W, C) format to (C, H, W) for PyTorch compatibility
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Box), "Image observation space must be Box"
        prev_space = self.observation_space
        H, W, C = prev_space.shape
        self.observation_space = gym.spaces.Box(low=prev_space.low.min(), high=prev_space.high.max(), shape=(C, H, W), dtype=prev_space.dtype)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class OneHotWrapper(gym.ObservationWrapper):
    """
    Converts integer observation Box (height, width) into one-hot encoding (channels, height, width).
    """
    def __init__(self, env, channels:tuple=None):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), "Observation space must be Box"
        assert len(env.observation_space.shape) == 2, "Observation must be 2D (height, width)"
        height, width = self.observation_space.shape

        if channels is None:
            selected_channels = np.arange(int(self.observation_space.high.max()) + 1)
        else:
            selected_channels = np.array(channels)

        self.observation_space = gym.spaces.Box(low=0.0,high=1.0, shape=(len(selected_channels), height, width), dtype=np.float32)
        self._channels = selected_channels.astype(np.float32).reshape(-1, 1, 1)  # to be broadcasted with (H, W)

    def observation(self, observation):
        onehot = (observation == self._channels).astype(np.float32) # (C, 1, 1) == (H, W)  ->  (C, H, W)
        return onehot
