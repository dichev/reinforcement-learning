import numpy as np
import gymnasium as gym


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    """
    Converts discrete state to one-hot vector.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, state):
        onehot = np.zeros(self.observation_space.shape, dtype=np.float32)
        onehot[state] = 1.0
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
        state, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            reward = self.step_penalty

        return state, reward, terminated, truncated, info


