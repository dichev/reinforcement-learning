import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from lib.playground import play_episode, play_steps

ENV = 'FrozenLake-v1'  # slippery
MAX_ITERATIONS = 200
LEARNING_RANDOM_STEPS = 100  # exploration
TEST_EPISODES = 20
GAMMA = .9


class AgentV:
    def __init__(self, k_actions, n_states, gamma):
        self.k_actions = k_actions
        self.n_states = n_states
        self.gamma = gamma
        self.memory = {
            'values':      np.zeros((n_states,)),
            'rewards':     np.zeros((n_states, k_actions, n_states)),
            'transitions': np.zeros((n_states, k_actions, n_states)),  # number of transitions
        }

    def collect(self, state, action, reward, state_next):
        self.memory['rewards'][state, action, state_next] = reward  # note: average it for stochastic rewards
        self.memory['transitions'][state, action, state_next] += 1

    def calc_actions_values(self, state):
        Q = np.zeros(self.k_actions)
        for action in range(self.k_actions):
            counts = self.memory['transitions'][state, action]
            total = counts.sum()

            # skip states with no transitions
            filtered_states = np.where(counts > 0)[0]

            # average over future states
            for i, state_next in enumerate(filtered_states):
                p = counts[state_next] / total
                r = self.memory['rewards'][state, action, state_next]
                v_next = self.get_state_value(state_next)
                Q[action] += p * (r + self.gamma * v_next)

        return Q

    def get_state_value(self, state):
        v = self.memory['values'][state]
        return v

    def update_values(self):
        Q = np.array([self.calc_actions_values(state) for state in range(self.n_states)])

        # synchronously update all the q values
        for state in range(self.n_states):
            self.memory['values'][state] = Q[state].max()

    def policy(self, state):
        Q = self.calc_actions_values(state) # one step look ahead
        return np.argmax(Q).item()

    def __repr__(self):
        return f'AgentV(gamma={self.gamma})'


class AgentQ(AgentV): # for comparison train only the q-values

    def __init__(self, k_actions, n_states, gamma):
        super().__init__(k_actions, n_states, gamma)
        self.memory['q_values'] = np.zeros((n_states, k_actions))
        del self.memory['values']

    def get_state_value(self, state):
        a = self.policy(state)
        q = self.memory['q_values'][state, a]
        return q

    def update_values(self):
        Q = np.array([self.calc_actions_values(state) for state in range(self.n_states)])

        # synchronously update all the q values
        self.memory['q_values'][:] = Q

    def policy(self, state):
        Q = self.memory['q_values'][state]
        return np.argmax(Q).item()

    def __repr__(self):
        return f'AgentQ(gamma={self.gamma})'


# Test agents
env = gym.make(ENV, render_mode=None)
agents = (
    AgentV(env.action_space.n, env.observation_space.n, gamma=GAMMA),
    AgentQ(env.action_space.n, env.observation_space.n, gamma=GAMMA),
)
for agent in agents:
    env.reset()
    avg_rewards = []
    for i in range(1, MAX_ITERATIONS + 1):
        # learning
        for obs, action, reward, obs_next, terminated, truncated in play_steps(env, max_steps=LEARNING_RANDOM_STEPS):
            agent.collect(obs, action, reward, obs_next)
        agent.update_values()

        # testing
        episodes = [play_episode(env, agent.policy) for _ in range(TEST_EPISODES)]
        for e in episodes:
            for obs, action, reward, obs_next, done in e.as_trajectory():
                agent.collect(obs, action, reward, obs_next)

        avg_reward = sum([ep.total_rewards for ep in episodes]) / TEST_EPISODES
        avg_rewards.append(avg_reward)
        if i % 10 == 0:
            print(f'{i:>5}/{MAX_ITERATIONS}) {agent} | avg_rewards={avg_reward:.2f}')

    plt.plot(avg_rewards, label=f'{agent}')
plt.title('Avg rewards over time')
optimal_reward = 0.8
plt.hlines(optimal_reward, 0, MAX_ITERATIONS, color="green", linestyles='dashed')
plt.text(0, optimal_reward+.1, f'optimal reward', horizontalalignment='left')
plt.legend()
plt.tight_layout()
plt.show()


# Play one episode with visualization
print(f"Playing one episode with the trained agent")
env = gym.make(ENV, render_mode='human')
episode = play_episode(env, agents[0].policy)
print(f"Episode finished with reward {episode.total_rewards}")
env.close()


