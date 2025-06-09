import numpy as np

class MultiArmBandit:
    def __init__(self, k_arms, stationary=False):
        self.probs = np.random.normal(0, 1, size=k_arms)
        self.noise = np.zeros(k_arms)
        self.stationary = stationary

    def step(self, action):
        if self.stationary:
            p = self.probs[action]
        else:
            p = (self.probs[action] + self.noise[action]).clip(0, 1)
            self.noise[action] += np.random.normal(0, 0.1)

        reward = np.random.normal(p, 0.1)
        return reward

    def reset(self):
        self.noise *= 0


class ContextualBandit:
    def __init__(self, n_bandits, k_arms, stationary=False):
        self.bandits = [MultiArmBandit(k_arms, stationary) for _ in range(n_bandits)]
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



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = MultiArmBandit(10, True)
    for action in range(10):
        r = [env.step(action) for _ in range(1000)]
        plt.hist(r, bins=100, label=f'arm {action}', alpha=.5, density=True)
    plt.ylabel('rewards')
    plt.title('Multi-arm bandit')
    plt.legend()
    plt.show()
