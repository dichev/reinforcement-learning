import numpy as np


class MultiArmBandit:
    def __init__(self, k_arms, max_reward, stationary=False):
        self.probs = np.random.uniform(0, 1, size=k_arms)
        self.noise = np.zeros(k_arms)
        self.stationary = stationary
        self.max_reward = max_reward

    def step(self, action):
        if self.stationary:
            p = self.probs[action]
        else:
            p = (self.probs[action] + self.noise[action]).clip(0, 1)
            self.noise[action] += np.random.normal(0, 0.1)

        reward = np.random.binomial(self.max_reward, p)
        return reward

    def reset(self):
        self.noise *= 0

