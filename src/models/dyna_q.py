"""
Reinforcement Learning, Sutton & Barto, 2018
[ref 8.1: Dyna Maze] Dyna-Q Planning
[ref 8.3: Shortcut Maze] Dyna-Q+ Planning with Exploration Bonus
"""

import numpy as np
import random
from math import sqrt
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
    def __init__(self, on_policy_sampling=False):
        self.transitions = {} # {(s,a): (r, s_, term)}
        self.pool = []        # pool of state-action pairs visited at least once
        self.on_policy = on_policy_sampling # just a flag

    def update(self, s, a, r, s_, term):
        if (s,a) not in self.transitions:
            self.pool.append((s, a))

        self.transitions[(s,a)] = (r, s_, term)  # the env is deterministic, so just keep the last seen reward

    def sample(self): # simulated exp
        s, a = random.choice(self.pool)
        r, s_, term = self.transitions[(s,a)]
        return s, a, r, s_, term

    def predict(self, s, a):
        if (s, a) not in self.transitions:
            return 0.0, s, False
        r, s_, term = self.transitions[(s, a)]
        return r, s_, term



class EnvModelPlus: # DynaQ+ with exploration bonus
    def __init__(self, n_states, n_actions, k=0.001):
        self.transitions = {} # {(s,a): (r, s_, term, last_visit)}
        self.n_states = n_states
        self.n_actions = n_actions
        self.steps = 0
        self.k = k

    def update(self, s, a, r, s_, term):
        self.steps += 1
        self.transitions[(s,a)] = (r, s_, term, self.steps)  # the env is deterministic, so just keep the last seen reward

    def sample(self): # simulated exp
        s = random.randrange(self.n_states)
        a = random.randrange(self.n_actions)

        if (s,a) not in self.transitions: # never visited states are still used in DynaQ+, but they transition to themselves
            r, s_, term, last_visit = 0.0, s, False, 0
        else:
            r, s_, term, last_visit = self.transitions[(s,a)]

        # add exploration bonus
        t = self.steps - last_visit
        r_exp = r + self.k * sqrt(t)

        return s, a, r_exp, s_, term

