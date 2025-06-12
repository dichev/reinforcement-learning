import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from lib.playground import play_episode


ENV = 'FrozenLake-v1'  # slippery
MAX_ITERATIONS = 200
LEARNING_RANDOM_STEPS = 100  # exploration
TEST_EPISODES = 20
GAMMA = .9


class Agent:
    def __init__(self, k_actions, n_states, gamma):
        self.k_actions = k_actions
        self.n_states = n_states
        self.memory = {
            'values':      np.zeros((n_states,)),
            'q_values':    np.zeros((n_states, k_actions)),
            'rewards':     np.zeros((n_states, k_actions, n_states)),
            'transitions': np.zeros((n_states, k_actions, n_states)),  # number of transitions
        }
        self.gamma = gamma

    def collect(self, state, action, reward, state_next):
        self.memory['rewards'][state, action, state_next] = reward  # note: average it for stochastic rewards
        self.memory['transitions'][state, action, state_next] += 1


    def update_values(self):
        Q = np.zeros_like(self.memory['q_values'])
        for action in range(self.k_actions):
            for state in range(self.n_states):
                counts = self.memory['transitions'][state, action]
                total = counts.sum()

                # skip states with no transitions
                filtered_states = np.where(counts > 0)[0]

                # average over future states
                q = 0.
                for i, state_next in enumerate(filtered_states):
                    p = counts[state_next] / total
                    r = self.memory['rewards'][state, action, state_next]
                    a = self.policy(state_next)
                    q_next = self.memory['q_values'][state_next, a]
                    q += p * (r + self.gamma * q_next)

                Q[state, action] = q

        # synchronously update all the q values
        self.memory['q_values'][:] = Q

    def policy(self, state):
        Q = self.memory['q_values'][state]
        return np.argmax(Q).item()

    def __repr__(self):
        return f'Agent(gamma={self.gamma})'



# Test two agents
env = gym.make(ENV, render_mode=None)
agent = Agent(env.action_space.n, env.observation_space.n, gamma=GAMMA)

avg_rewards = []
for i in range(1, MAX_ITERATIONS + 1):

    # learning
    obs, info = env.reset()
    for t in range(LEARNING_RANDOM_STEPS):
        action = env.action_space.sample() # NO policy, random exploration
        obs_next, reward, terminated, truncated, info = env.step(action)
        agent.collect(obs, action, reward, obs_next)
        if terminated or truncated:
            obs_next, info = env.reset()
        obs = obs_next
    agent.update_values()

    # testing
    episodes = [play_episode(env, agent.policy, False) for _ in range(TEST_EPISODES)]  # todo: can collect the transitions and rewards too
    avg_reward = sum([ep.total_rewards for ep in episodes]) / TEST_EPISODES
    avg_rewards.append(avg_reward)
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
episode = play_episode(env, agent.policy)
print(f"Episode finished with reward {episode.total_rewards}")
env.close()


# For analysis
analyze = {
    'Q': agent.memory['q_values'].T,
    'N': agent.memory['transitions'].sum(axis=0),
}
fig = plt.figure(figsize=np.array([env.observation_space.n * 2, env.action_space.n]).clip(6, 20))
for i, (k, v) in enumerate(analyze.items()):
    plt.subplot(1, 2, i + 1)
    plt.title(k)
    plt.imshow(v, cmap='Blues')
    plt.xlabel('states')
    plt.ylabel('actions')
plt.tight_layout()
plt.show()

