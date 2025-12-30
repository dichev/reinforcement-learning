"""
Reinforcement Learning, Sutton & Barto, 2018
[ref 8.1: Dyna Maze] Dyna-Q Planning
"""

from collections import defaultdict
import gymnasium as gym
import matplotlib.pyplot as plt

import envs.custom_gyms
from lib.plots import draw_policy_grid
from envs.TinyGrid import GRID_MAP, ACTION_ARROWS_MAP
from models.dyna_q import DynaQ, EnvModel



EPISODES               = 50
MAX_STEPS_PER_EPISODE  = 2000
ALPHA                  = 0.10  # learn rate / step size
GAMMA                  = 0.95
EPSILON                = 0.10



# Prepare the env and agents
env = gym.make('custom/TinyGrid', template='dyna_maze_9x6', fully_observable=False, max_steps=MAX_STEPS_PER_EPISODE)
env = gym.wrappers.TransformReward(env, lambda r: 1. if r > 0. else 0.)
n_states, n_actions = env.observation_space.n, env.action_space.n,
agents = {
    'DynaQ(0)':   DynaQ( 0, n_states, n_actions, EPSILON, ALPHA, GAMMA, env_model=EnvModel()),
    'DynaQ(5)':   DynaQ( 5, n_states, n_actions, EPSILON, ALPHA, GAMMA, env_model=EnvModel()),
    'DynaQ(50)':  DynaQ(50, n_states, n_actions, EPSILON, ALPHA, GAMMA, env_model=EnvModel()),
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
            for i in range(agent.planning_steps):
                ss, sa, sr, ss_next, term = agent.model.simulated_exp()
                agent.update(ss, sa, sr, ss_next, bootstrap=not term)

            done = terminated or truncated
            s = s_next
            steps += 1

        print(f'Episode #{episode + 1} | {steps=}, last_reward={r}')
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



# Draw the learned policies:
for name, agent in agents.items():
    draw_policy_grid(agent.Q, env.unwrapped.grid, ACTION_ARROWS_MAP, GRID_MAP, title=f'\nPolicy of {name}')
