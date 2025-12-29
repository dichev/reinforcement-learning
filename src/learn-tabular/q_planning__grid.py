"""
Reinforcement Learning, Sutton & Barto, 2018
[ref 8.1: Dyna Maze] Dyna-Q Planning
"""

from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

import envs.custom_gyms
from lib.rng import random_argmax

EPISODES           = 50
MAX_EPISODE_STEPS  = 2000
ALPHA              = 0.10  # learn rate / step size
GAMMA              = 0.95
EPSILON            = 0.10


class DynaQ:
    def __init__(self, planning_steps, k_actions, n_states, eps):
        self.Q = np.zeros((n_states, k_actions), dtype=np.float32)
        self.model = EnvModel()
        self.planning_steps = planning_steps
        self.k_actions = k_actions
        self.n_states = n_states
        self.eps = eps

    def policy(self, s):
        action = random_argmax(self.Q[s, :])  # arg max with tie breaking (since all Q were initialized to 0)
        return action

    def behavior(self, s):
        if np.random.rand() < self.eps:
            return np.random.randint(self.k_actions)
        return self.policy(s)

    def update(self, s, a, r, s_, bootstrap):
        target = r + GAMMA * self.Q[s_,:].max() if bootstrap else r
        self.Q[s,a] += ALPHA * (target - self.Q[s,a])


class EnvModel:
    def __init__(self):
        self.transitions = {} # {(s,a): (r, s_, term)}
        self.pool = []        # pool of state-action pairs visited at least once

    def update(self, s, a, r, s_, term):
        if (s,a) not in self.transitions:
            self.pool.append((s, a))

        self.transitions[(s,a)] = (r, s_, term)  # the env is deterministic, so just keep the last seen reward

    def simulated_exp(self):
        s, a = random.choice(self.pool)
        r, s_, term = self.transitions[(s,a)]
        return s, a, r, s_, term



# Prepare the env and agents
env = gym.make('custom/TinyGrid', template='dyna_maze_9x6', fully_observable=False, max_steps=MAX_EPISODE_STEPS)
env = gym.wrappers.TransformReward(env, lambda r: 1. if r > 0. else 0.)
agents = {
    'DynaQ(0)':  DynaQ( 0, env.action_space.n, env.observation_space.n, EPSILON),
    'DynaQ(5)':  DynaQ( 5, env.action_space.n, env.observation_space.n, EPSILON),
    'DynaQ(50)': DynaQ(50, env.action_space.n, env.observation_space.n, EPSILON),
}


# Learn policies
history = defaultdict(list)
for name, agent in agents.items():
    print(f'\n# {name}')
    for episode in range(EPISODES):
        s, _ = env.reset()
        steps =  0
        done = False
        while not done:
            a = agent.behavior(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            agent.update(s, a, r, s_next, bootstrap=not terminated)
            agent.model.update(s, a, r, s_next, terminated)
            done = terminated or truncated
            s = s_next
            steps += 1

            for i in range(agent.planning_steps):
                ss, sa, sr, ss_next, term = agent.model.simulated_exp()
                agent.update(ss, sa, sr, ss_next, bootstrap=not term)


        print(f'Episode #{episode+1} | {steps=}, last_reward={r}')
        history[name].append(steps)



# Plot results
for name, steps in history.items():
    plt.plot(range(EPISODES), steps, label=name)
plt.legend()
plt.ylim(0, 500)
plt.xlabel('Episodes')
plt.ylabel(f'Steps (per episode)')
plt.title(f'DynaQ with n planning steps ($\epsilon$-greedy={EPSILON})')
plt.legend()
plt.show()


