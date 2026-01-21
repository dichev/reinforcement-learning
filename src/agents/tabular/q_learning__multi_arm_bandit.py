import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs.bandits import MultiArmBandit


K_ARMS = 10


class Agent:
    def __init__(self, k_actions, step_size: str|float = 'sample-average', initial_q=0.):
        self.k_actions = k_actions
        self.initial_q = initial_q
        self.memory = np.zeros((k_actions, 2))  # [actions, (trials, exp_reward)]
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
    def __init__(self, k_actions, eps_greedy, step_size='sample-average', initial_q=0.):
        super().__init__(k_actions, step_size, initial_q)
        self.eps_greedy = eps_greedy

    def policy(self):
        if np.random.random() < self.eps_greedy:
            return np.random.randint(self.k_actions)

        Q = self.memory[:, 1]
        return np.argmax(Q).item()

    def __repr__(self):
        return f'EpsilonAgent(eps={self.eps_greedy:.2f}, step={self.step_size}, init_q={self.initial_q})'


class SoftmaxAgent(Agent):
    def __init__(self, k_actions, temp, step_size='sample-average', initial_q=0.):
        super().__init__(k_actions, step_size, initial_q)
        self.temp = temp

    def policy(self):
        Q = self.memory[:, 1]
        e = np.exp(Q / self.temp)
        p = e / e.sum()
        return np.random.multinomial(1, p).argmax()

    def __repr__(self):
        return f'SoftmaxAgent(temp={self.temp:.2f}, step={self.step_size}, init_q={self.initial_q})'


# Test two agents
agent1 = EpsilonAgent(K_ARMS, 0.10)
agent2 = SoftmaxAgent(K_ARMS, 0.5)
env = MultiArmBandit(K_ARMS, True)
for agent in [agent1, agent2]:
    rewards = []
    for i in range(1000):
        action = agent.policy()
        reward = env.step(action)
        agent.update(action, reward)
        rewards.append(reward)
    avg_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    plt.scatter(range(len(rewards)), rewards, s=.2, label=f'{agent}')
    plt.plot(avg_rewards, label=f'{agent}')
plt.title('Rewards over time (single run)')
plt.legend()
plt.show()



# Benchmark
def benchmark(agents, envs, steps=10000, subtitle=''):
    RUNS = len(envs)
    for i, agent in enumerate(agents):
        rewards = np.zeros((RUNS, steps))
        for run, env in enumerate(tqdm(envs, desc=str(agent))):
            agent.reset()
            env.reset()
            for t in range(steps):
                action = agent.policy()
                reward = env.step(action)
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
    EpsilonAgent(K_ARMS, 0.),
    EpsilonAgent(K_ARMS, 0.01),
    EpsilonAgent(K_ARMS, 0.10),
    SoftmaxAgent(K_ARMS, 0.5),
)
envs = [MultiArmBandit(K_ARMS, True) for _ in range(RUNS)]
benchmark(agents, envs, STEPS, 'stationary bandits')



# Benchmark fixed vs sample-average step size on non-stationary agents
print('Benchmark fixed vs sample-average step size on non-stationary agents')
RUNS = 500
STEPS = 10_000
agents = (
    EpsilonAgent(K_ARMS, 0.01, 'sample-average'),
    EpsilonAgent(K_ARMS, 0.01, 0.1),
)
envs = [MultiArmBandit(K_ARMS, False) for _ in range(RUNS)]
benchmark(agents, envs, STEPS, 'NON-stationary bandits')



# Benchmark against initial Q values
print('Benchmark against initial Q values')
RUNS = 500
STEPS = 500
agents = (
    EpsilonAgent(K_ARMS, 0., initial_q=5),
    EpsilonAgent(K_ARMS, 0.1, initial_q=0.),
)
envs = [MultiArmBandit(K_ARMS, False) for _ in range(RUNS)]
benchmark(agents, envs, STEPS, 'initial Q values, NON-stationary')

