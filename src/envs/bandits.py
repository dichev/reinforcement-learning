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


class ContextualBandit:
    def __init__(self, n_bandits, k_arms, max_reward, stationary=False):
        self.bandits = [MultiArmBandit(k_arms, max_reward, stationary) for _ in range(n_bandits)]
        self.n_bandits = n_bandits
        self.selected = None

    def step(self, action):
        if self.selected is None: raise Exception('Cannot call env.step() before calling env.reset()')

        state = self.selected
        reward = self.bandits[self.selected].step(action)
        self._update()
        return state, reward

    def reset(self):
        for bandit in self.bandits:
            bandit.reset()
        self._update()
        return self.selected

    def _update(self):
        self.selected = np.random.randint(self.n_bandits)


