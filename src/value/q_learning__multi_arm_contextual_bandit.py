import numpy as np
import matplotlib.pyplot as plt
from envs.bandits import ContextualBandit


N_BANDITS   = 12    # states
K_ARMS      = 10   # actions
MAX_REWARD  = 10
STEPS       = 1000 * N_BANDITS


class EpsilonAgent:
    def __init__(self, k_actions, n_states, eps_greedy, step_size: str|float = 'sample-average', initial_q=0.):
        self.eps_greedy = eps_greedy
        self.k_actions = k_actions
        self.n_states = n_states
        self.initial_q = initial_q
        self.memory = np.zeros((k_actions, n_states, 2))  # for [actions, states, (trials, exp_reward)]
        self.memory[:, :, 1].fill(initial_q)
        self.step_size = step_size

    def update(self, action, state, reward):
        n, q = self.memory[action, state]
        step_size = 1 / (n + 1) if self.step_size == 'sample-average' else self.step_size
        self.memory[action, state][0] = n + 1
        self.memory[action, state][1] = q + step_size * (reward - q) # accumulative mean

    def reset(self):
        self.memory[:, :, 0].fill(0)
        self.memory[:, :, 1].fill(self.initial_q)

    def policy(self, state):
        if np.random.random() < self.eps_greedy:
            return np.random.randint(self.k_actions)

        Q = self.memory[:, state, 1]
        return np.argmax(Q).item()

    def __repr__(self):
        return f'EpsilonAgent(eps={self.eps_greedy:.2f}, step={self.step_size}, init_q={self.initial_q})'



class RandomAgent:
    def __init__(self, k_actions, n_states):
        self.k_actions = k_actions
        self.n_states = n_states

    def update(self, action, state, reward):
        pass

    def reset(self):
        pass

    def policy(self, state):
        return np.random.randint(self.k_actions)

    def __repr__(self):
        return f'RandomAgent()'




# Test two agents
env = ContextualBandit(N_BANDITS, K_ARMS, MAX_REWARD, True)
agents = (
    RandomAgent(K_ARMS, N_BANDITS),
    EpsilonAgent(K_ARMS, N_BANDITS,  0.1),
)
for agent in agents:
    agent.reset()
    state = env.reset()
    rewards = []
    mov_rewards = []
    mov_reward = 0

    for t in range(1, STEPS + 1):
        action = agent.policy(state)
        state_next, reward = env.step(action)
        agent.update(action, state, reward)
        state = state_next
        rewards.append(reward)
        mov_reward = (.99 * mov_reward + .01 * reward) if t > 0 else reward
        mov_rewards.append(mov_reward)
        if t % 100 == 0:
            print(f'{t:>5}/{STEPS}) {agent} | avg_rewards={np.mean(rewards):.2f}, mov_rewards={np.mean(mov_rewards[-1]):.2f}')

    avg_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    plt.scatter(range(len(rewards)), rewards, s=.01, label=f'{agent}')
    plt.plot(mov_rewards, label=f'{agent}')
plt.title('Rewards over time (single run)')
plt.legend()
plt.show()

