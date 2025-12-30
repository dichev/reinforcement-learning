"""
Reinforcement Learning, Sutton & Barto, 2018
[ref 8.1: Dyna Maze] Dyna-Q Planning
"""

import numpy as np
import random
from lib.rng import random_argmax


class DynaQ:
    def __init__(self, planning_steps, n_states, n_actions, eps, alpha, gamma, env_model):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.model = env_model
        self.planning_steps = planning_steps
        self.n_actions = n_actions
        self.n_states = n_states
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma

    def policy(self, s):
        action = random_argmax(self.Q[s, :])  # arg max with tie breaking (since all Q were initialized to 0)
        return action

    def behavior(self, s):
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        return self.policy(s)

    def update(self, s, a, r, s_, bootstrap):
        target = r + self.gamma * self.Q[s_,:].max() if bootstrap else r
        self.Q[s,a] += self.alpha * (target - self.Q[s,a])


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

