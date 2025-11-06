import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from envs.bandits import ContextualBandit

N_BANDITS   = 8    # states
K_ARMS      = 10   # actions
STEPS       = 10_000
HIDDEN_SIZE = 32


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



class NeuralAgent:
    def __init__(self, k_actions, n_states, hidden_size):
        self.k_actions = k_actions
        self.n_states = n_states
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, k_actions),
        )
        self.hidden_size = hidden_size

    def forward(self, state):
        state = F.one_hot(torch.tensor(state), num_classes=self.n_states).float()
        return self.net(state)

    @torch.no_grad()
    def reset(self):
        self.net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    @torch.no_grad()
    def policy(self, state):
        Q = agent.forward(state)
        p = F.softmax(Q, dim=-1)
        action = torch.multinomial(p, num_samples=1)
        return action.item()

    def __repr__(self):
        return f'NeuralAgent(hidden_size={self.hidden_size},)'


class RandomAgent:
    def __init__(self, k_actions, n_states):
        self.k_actions = k_actions
        self.n_states = n_states

    def update(self, *args, **kwargs): pass
    def reset(self): pass

    def policy(self, state):
        return np.random.randint(self.k_actions)

    def __repr__(self):
        return f'RandomAgent()'



# Test two agents
env = ContextualBandit(N_BANDITS, K_ARMS, True)
agents = (
    RandomAgent(K_ARMS, N_BANDITS),
    EpsilonAgent(K_ARMS, N_BANDITS,  0.10, step_size=0.1),
    NeuralAgent(K_ARMS, N_BANDITS, HIDDEN_SIZE),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=agents[-1].net.parameters(), lr=.001)

for agent in agents:
    agent.reset()
    ob = env.reset()
    rewards = []
    mov_rewards = []
    mov_reward = 0
    mov_loss = 0

    for t in range(1, STEPS + 1):
        action = agent.policy(ob)
        ob_next, reward = env.step(action)

        if isinstance(agent, NeuralAgent):
            optimizer.zero_grad()
            Q = agent.forward(ob)
            R = F.one_hot(torch.tensor(action), num_classes=K_ARMS).float() * reward
            loss = loss_fn(Q, R)
            loss.backward()
            optimizer.step()
            mov_loss = (.9 * mov_loss + .1 * loss.item()) if t > 0 else loss.item()
        else:
            agent.update(action, ob, reward)

        ob = ob_next
        rewards.append(reward)
        mov_reward = (.99 * mov_reward + .01 * reward) if t > 0 else reward
        mov_rewards.append(mov_reward)
        if t % 1000 == 0:
            print(f'{t:>5}/{STEPS}) {agent} | avg_rewards={np.mean(rewards):.2f}, mov_rewards={np.mean(mov_rewards[-1]):.2f}' + (f', {mov_loss=:.4f}' if isinstance(agent, NeuralAgent) else '') )

    avg_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    plt.scatter(range(len(rewards)), rewards, s=.005)
    plt.plot(mov_rewards, label=f'{agent}')
plt.title('Rewards over time (single run)')
optimal_reward = env.optimal_reward
plt.hlines(optimal_reward, 0, STEPS, color="green", linestyles='dashed')
plt.text(0, optimal_reward+.1, f'optimal reward', horizontalalignment='left')
plt.legend()
plt.tight_layout()
plt.show()


# For analysis
"""
agent = agents[1]
analyze = {
    'E[R]': np.array([b.probs for b in env.bandits]).T,
    'Q': agent.memory[:, :, 1],
    'N': agent.memory[:, :, 0],
}
fig = plt.figure(figsize=np.array([N_BANDITS * 3, K_ARMS]).clip(6, 20))
for i, (k, v) in enumerate(analyze.items()):
    plt.subplot(1, 3, i + 1)
    plt.title(k)
    plt.imshow(v, cmap='Blues')
    plt.xlabel('states')
    plt.ylabel('actions')
plt.tight_layout()
plt.show()
"""
