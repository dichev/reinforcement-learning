import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N_BANDITS = 10
MAX_REWARD = 10


class MultiBandit:
    def __init__(self, n, max_reward, stationary=False):
        self.probs = np.random.uniform(0, 1, size=n)
        self.noise = np.zeros(n)
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


class Agent:
    def __init__(self, n_actions, step_size: str|float = 'sample-average'):
        self.n_actions = n_actions
        self.memory = np.zeros((n_actions, 2)) # action -> total rewards
        self.step_size = step_size
        self.steps = 0

    def update(self, action, reward):
        n, q = self.memory[action]
        step_size = 1 / (n + 1) if self.step_size == 'sample-average' else self.step_size
        self.memory[action][0] = n + 1
        self.memory[action][1] = q + step_size * (reward - q) # accumulative mean
        self.steps += 1

    def reset(self):
        self.memory *= 0
        self.steps = 0

    def policy(self):
        raise NotImplementedError


class EpsilonAgent(Agent):
    def __init__(self, n_actions, step_size, eps_greedy):
        super().__init__(n_actions, step_size)
        self.eps_greedy = eps_greedy

    def policy(self):
        if np.random.random() < self.eps_greedy:
            return np.random.randint(self.n_actions)

        Q = self.memory[:, 1]
        return np.argmax(Q).item()

    def __repr__(self):
        return f'EpsilonAgent(eps={self.eps_greedy:.2f}, step_size={self.step_size})'


class SoftmaxAgent(Agent):
    def __init__(self, n_actions, step_size, temp):
        super().__init__(n_actions, step_size)
        self.temp = temp

    def policy(self):
        Q = self.memory[:, 1]
        e = np.exp(Q / self.temp)
        p = e / e.sum()
        return np.random.multinomial(1, p).argmax()

    def __repr__(self):
        return f'SoftmaxAgent(temp={self.temp:.2f}, step_size={self.step_size})'



# Test two agents
agent1 = EpsilonAgent(N_BANDITS, 'sample-average',0.10)
agent2 = SoftmaxAgent(N_BANDITS, 'sample-average',1.)
game = MultiBandit(N_BANDITS, MAX_REWARD, True)
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




# Benchmark
def benchmark(agents, games, steps=10000):
    RUNS = len(games)
    for i, agent in enumerate(agents):
        rewards = np.zeros((RUNS, steps))
        for run, game in enumerate(tqdm(games, desc=str(agent))):
            agent.reset()
            game.reset()
            for t in range(steps):
                action = agent.policy()
                reward = game.step(action)
                agent.update(action, reward)
                rewards[run, t] = reward
        plt.plot(rewards.mean(axis=0), label=f'{agent}', linewidth=.7)

    plt.title(f'Rewards over time ({RUNS} runs) - stationary bandits')
    plt.legend()
    plt.tight_layout()
    plt.show()



# Benchmark multiple agents over multiple runs
print('Benchmark multiple agents over multiple runs')
RUNS = 500
STEPS = 1_000
agents = (
    EpsilonAgent(N_BANDITS, 'sample-average', 0.),
    EpsilonAgent(N_BANDITS, 'sample-average', 0.01),
    EpsilonAgent(N_BANDITS, 'sample-average', 0.10),
    SoftmaxAgent(N_BANDITS, 'sample-average', 1),
)
games = [MultiBandit(N_BANDITS, MAX_REWARD, True) for _ in range(RUNS)]
benchmark(agents, games, STEPS)



# Benchmark fixed vs sample-average step size on non-stationary agents
print('Benchmark fixed vs sample-average step size on non-stationary agents')
RUNS = 500
STEPS = 10_000
agents = (
    EpsilonAgent(N_BANDITS, 'sample-average', 0.01),
    EpsilonAgent(N_BANDITS, 0.1, 0.01),
)
games = [MultiBandit(N_BANDITS, MAX_REWARD, False) for _ in range(RUNS)]
benchmark(agents, games, STEPS)
