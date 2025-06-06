import numpy as np
import matplotlib.pyplot as plt


N_BANDITS = 10
MAX_REWARD = 10
EPS_GREEDY = 0.01
LOG_EVERY = 10


class Bandit:
    def __init__(self, prob, max_reward):
        self.prob = prob
        self.max_reward = max_reward

    def step(self, n=1):
        return np.random.binomial(self.max_reward, self.prob, size=n if n > 1 else None)

    def __repr__(self):
        return f'Bandit(p={self.prob:.2f})'

class MultiBandit:
    def __init__(self, n, max_reward):
        self.bandits = [Bandit(prob, max_reward) for prob in np.random.uniform(0, 1, size=n)]

    def step(self, action):
        return self.bandits[action].step()


class Agent:
    def __init__(self, n_actions, eps_greedy):
        self.n_actions = n_actions
        self.eps_greedy = eps_greedy
        self.memory = np.zeros((n_actions, 2)) # action -> total rewards
        self.steps = 0
        self.total_reward = 0

    def update(self, action, reward):
        k, mu = self.memory[action]
        self.memory[action][0] = k + 1
        self.memory[action][1] = ((mu * k) + reward) / (k + 1)  # accumulative mean
        self.steps += 1
        self.total_reward += reward

    def policy(self):
        if self.steps < self.n_actions: # initially play each action once
            return self.steps

        if np.random.random() < self.eps_greedy:
            return np.random.randint(self.n_actions)

        Q = self.memory[:, 1]
        return np.argmax(Q).item()




agent = Agent(N_BANDITS, EPS_GREEDY)
game = MultiBandit(N_BANDITS, MAX_REWARD)
rewards = []
for i in range(500):
    action = agent.policy()
    reward = game.step(action)
    agent.update(action, reward)
    rewards.append(reward)
    if i % LOG_EVERY == 0:
        print(f'{i}. Avg reward {np.mean(rewards)}')


# Plots
avg_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
plt.scatter(range(len(rewards)), rewards, s=.2); plt.title('Rewards over time');
plt.scatter(range(len(rewards)), avg_rewards, s=.5); plt.title('Rewards over time');
plt.show()

