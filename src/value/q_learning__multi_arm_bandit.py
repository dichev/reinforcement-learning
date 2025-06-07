import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N_BANDITS = 10
MAX_REWARD = 10


class MultiBandit:
    def __init__(self, k, max_reward, stationary=False):
        self.probs = np.random.uniform(0, 1, size=k)
        self.noise = np.zeros(k)
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
    def __init__(self, n_actions, step_size: str|float = 'sample-average', initial_q=0.):
        self.n_actions = n_actions
        self.initial_q = initial_q
        self.memory = np.zeros((n_actions, 2))
        self.memory[:, 1].fill(initial_q)
        self.step_size = step_size
        self.steps = 0

    def update(self, action, reward):
        n, q = self.memory[action]
        step_size = 1 / (n + 1) if self.step_size == 'sample-average' else self.step_size
        self.memory[action][0] = n + 1
        self.memory[action][1] = q + step_size * (reward - q) # accumulative mean
        self.steps += 1

    def reset(self):
        self.memory[:, 0].fill(0)
        self.memory[:, 1].fill(self.initial_q)
        self.steps = 0

    def policy(self):
        raise NotImplementedError


class EpsilonAgent(Agent):
    def __init__(self, n_actions, eps_greedy, step_size='sample-average', initial_q=0.):
        super().__init__(n_actions, step_size, initial_q)
        self.eps_greedy = eps_greedy

    def policy(self):
        if np.random.random() < self.eps_greedy:
            return np.random.randint(self.n_actions)

        Q = self.memory[:, 1]
        return np.argmax(Q).item()

    def __repr__(self):
        return f'EpsilonAgent(eps={self.eps_greedy:.2f}, step={self.step_size}, init_q={self.initial_q})'


class SoftmaxAgent(Agent):
    def __init__(self, n_actions, temp, step_size='sample-average', initial_q=0.):
        super().__init__(n_actions, step_size, initial_q)
        self.temp = temp

    def policy(self):
        Q = self.memory[:, 1]
        e = np.exp(Q / self.temp)
        p = e / e.sum()
        return np.random.multinomial(1, p).argmax()

    def __repr__(self):
        return f'SoftmaxAgent(temp={self.temp:.2f}, step={self.step_size}, init_q={self.initial_q})'



# Test two agents
agent1 = EpsilonAgent(N_BANDITS, 0.10)
agent2 = SoftmaxAgent(N_BANDITS, 1.)
game = MultiBandit(N_BANDITS, MAX_REWARD, True)
for agent in [agent1, agent2]:
    rewards = []
    for i in range(1000):
        action = agent.policy()
        reward = game.step(action)
        agent.update(action, reward)
        rewards.append(reward)
    avg_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    plt.scatter(range(len(rewards)), rewards, s=.2, label=f'{agent}')
    plt.plot(avg_rewards, label=f'{agent}')
plt.title('Rewards over time (single run)')
plt.legend()
plt.show()



# Benchmark
def benchmark(agents, games, steps=10000, subtitle=''):
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

    plt.title(f'Rewards over time ({RUNS} runs) - {subtitle}')
    plt.legend()
    plt.tight_layout()
    plt.show()



# Benchmark multiple agents over multiple runs
print('Benchmark multiple agents over multiple runs')
RUNS = 500
STEPS = 1_000
agents = (
    EpsilonAgent(N_BANDITS, 0.),
    EpsilonAgent(N_BANDITS,  0.01),
    EpsilonAgent(N_BANDITS,  0.10),
    SoftmaxAgent(N_BANDITS,  1.00),
)
games = [MultiBandit(N_BANDITS, MAX_REWARD, True) for _ in range(RUNS)]
benchmark(agents, games, STEPS, 'stationary bandits')



# Benchmark fixed vs sample-average step size on non-stationary agents
print('Benchmark fixed vs sample-average step size on non-stationary agents')
RUNS = 500
STEPS = 10_000
agents = (
    EpsilonAgent(N_BANDITS, 0.01, 'sample-average'),
    EpsilonAgent(N_BANDITS, 0.01, 0.1),
)
games = [MultiBandit(N_BANDITS, MAX_REWARD, False) for _ in range(RUNS)]
benchmark(agents, games, STEPS, 'NON-stationary bandits')



# Benchmark against initial Q values
print('Benchmark against initial Q values')
RUNS = 500
STEPS = 500
agents = (
    EpsilonAgent(N_BANDITS, 0., initial_q=MAX_REWARD),
    EpsilonAgent(N_BANDITS, 0.1, initial_q=0.),
)
games = [MultiBandit(N_BANDITS, MAX_REWARD, False) for _ in range(RUNS)]
benchmark(agents, games, STEPS, 'initial Q values, NON-stationary')
