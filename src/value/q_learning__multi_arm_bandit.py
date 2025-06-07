import numpy as np
import matplotlib.pyplot as plt


N_BANDITS = 10
MAX_REWARD = 10


class MultiBandit:
    def __init__(self, n, max_reward):
        self.probs = np.random.uniform(0, 1, size=n)
        self.max_reward = max_reward

    def step(self, action):
        return np.random.binomial(self.max_reward, self.probs[action])


class Agent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.memory = np.zeros((n_actions, 2)) # action -> total rewards
        self.steps = 0

    def update(self, action, reward):
        n, q = self.memory[action]
        self.memory[action][0] = n + 1
        self.memory[action][1] = q + (reward - q) / (n + 1)  # accumulative mean
        self.steps += 1

    def reset(self):
        self.memory *= 0
        self.steps = 0

    def policy(self):
        raise NotImplementedError


class EpsilonAgent(Agent):
    def __init__(self, n_actions, eps_greedy):
        super().__init__(n_actions)
        self.eps_greedy = eps_greedy

    def policy(self):
        # if self.steps < self.n_actions: return self.steps  # warm-up: initially play each action once

        if np.random.random() < self.eps_greedy:
            return np.random.randint(self.n_actions)

        Q = self.memory[:, 1]
        return np.argmax(Q).item()

    def __repr__(self):
        return f'EpsilonAgent(eps={self.eps_greedy:.2f})'


class SoftmaxAgent(Agent):
    def __init__(self, n_actions, temp):
        super().__init__(n_actions)
        self.temp = temp

    def policy(self):
        # if self.steps < self.n_actions: return self.steps  # warm-up: initially play each action once

        Q = self.memory[:, 1]
        e = np.exp(Q / self.temp)
        p = e / e.sum()
        return np.random.multinomial(1, p).argmax()

    def __repr__(self):
        return f'SoftmaxAgent(temp={self.temp:.2f})'



# Test two agents
agent1 = EpsilonAgent(N_BANDITS, 0.10)
agent2 = SoftmaxAgent(N_BANDITS, 1.)
game = MultiBandit(N_BANDITS, MAX_REWARD)
for agent in [agent1, agent2]:
    rewards = []
    for i in range(500):
        action = agent.policy()
        reward = game.step(action)
        agent.update(action, reward)
        rewards.append(reward)
        if i % 100 == 0:
            print(f'{i}. Avg reward {np.mean(rewards)}')

    avg_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    plt.scatter(range(len(rewards)), rewards, s=.2, label=f'{agent}')
    plt.plot(avg_rewards, label=f'{agent}')

plt.title('Rewards over time (single run)')
plt.legend()
plt.show()




# Benchmark multiple agents over multiple runs
RUNS = 500
STEPS = 1000

agents = (
    EpsilonAgent(N_BANDITS, 0.),
    EpsilonAgent(N_BANDITS, 0.01),
    EpsilonAgent(N_BANDITS, 0.10),
    SoftmaxAgent(N_BANDITS, 1),
)
games = [MultiBandit(N_BANDITS, MAX_REWARD) for _ in range(RUNS)]

for i, agent in enumerate(agents):
    rewards = np.zeros((RUNS, STEPS))
    for run, game in enumerate(games):
        agent.reset()
        for t in range(STEPS):
            action = agent.policy()
            reward = game.step(action)
            agent.update(action, reward)
            rewards[run, t] = reward

        print(f'Run {run + 1}/{RUNS} {agent} | Avg reward: {rewards[run].mean():.2f}')
    plt.plot(rewards.mean(axis=0), label=f'{agent}', linewidth=.7)

plt.title(f'Rewards over time ({RUNS} runs)')
plt.legend()
plt.tight_layout()
plt.show()


