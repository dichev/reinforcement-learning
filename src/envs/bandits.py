import numpy as np

class MultiArmBandit:
    def __init__(self, k_arms, stationary=False, rewards_boost=5):
        self.probs = np.random.normal(0, 1, size=k_arms)
        self.probs[np.random.randint(k_arms)] += rewards_boost  # one of the arms pays far more than others (simplifies the analysis)
        self.noise = np.zeros(k_arms)
        self.stationary = stationary
        self.optimal_reward = np.max(self.probs)

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
    def __init__(self, n_bandits, k_arms, stationary=False, rewards_boost=5):
        self.bandits = [MultiArmBandit(k_arms, stationary, rewards_boost) for _ in range(n_bandits)]
        self.n_bandits = n_bandits
        self.selected = None
        self.optimal_reward = sum(bandit.optimal_reward for bandit in self.bandits) / n_bandits

    def step(self, action):
        if self.selected is None: raise Exception('Cannot call env.step() before calling env.reset()')

        reward = self.bandits[self.selected].step(action)
        state_next = self._switch_bandit()
        return state_next, reward

    def reset(self):
        for bandit in self.bandits:
            bandit.reset()
        state = self._switch_bandit()
        return state

    def _switch_bandit(self):
        self.selected = np.random.randint(self.n_bandits)
        return self.selected



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
