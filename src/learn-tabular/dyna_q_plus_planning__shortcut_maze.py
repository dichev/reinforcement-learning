"""
Reinforcement Learning, Sutton & Barto, 2018
[ref 8.3: Shortcut Maze] Dyna-Q+ Planning with Exploration Bonus
"""

from collections import defaultdict
import gymnasium as gym
import matplotlib.pyplot as plt

import envs.custom_gyms
from lib.plots import draw_policy_grid
from envs.TinyGrid import GRID_MAP, ACTION_ARROWS_MAP
from models.dyna_q import DynaQ, EnvModel, EnvModelPlus


TRAINING_STEPS         = 6000
MAX_STEPS_PER_EPISODE  = 2000
SWITCH_ENV_AFTER       = 3000  # steps
ALPHA                  = 0.10  # learn rate / step size
GAMMA                  = 0.95
EPSILON                = 0.10
PLANNING_EXP_BONUS     = 0.001  # k factor for DynaQ+ bonus exploration during planning


# Prepare the env and agents
env = gym.make('custom/TinyGrid', template='shortcut_maze_9x6_A', fully_observable=False, max_steps=MAX_STEPS_PER_EPISODE)
env = gym.wrappers.TransformReward(env, lambda r: 1. if r > 0. else 0.)
n_states, n_actions = env.observation_space.n, env.action_space.n,
agents = {
    'DynaQ(50)':  DynaQ(50, n_states, n_actions, EPSILON, ALPHA, GAMMA, env_model=EnvModel()),
    'DynaQ(50)+': DynaQ(50, n_states, n_actions, EPSILON, ALPHA, GAMMA, env_model=EnvModelPlus(n_states, n_actions, k=PLANNING_EXP_BONUS)),
}


# Learn policies
history = defaultdict(list)
for name, agent in agents.items():
    print(f'\n# {name}')
    total_steps = 0
    total_rewards = 0
    episode = 0
    while total_steps < TRAINING_STEPS:
        s, _ = env.reset(options={'template': 'shortcut_maze_9x6_A' if total_steps < SWITCH_ENV_AFTER else 'shortcut_maze_9x6_B'})
        steps = rewards = 0
        done = False
        while not done: # always finish each episode
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
            total_rewards += r
            history[name].append(total_rewards)

        episode += 1
        total_steps += steps
        if episode < 5 or episode % 100 == 0 or total_steps >= TRAINING_STEPS:
            print(f'Episode #{episode} | {steps=}, last_reward={r} | {total_steps=}, {total_rewards=}')



# Plot results
for name, cum_rewards in history.items():
    plt.plot(range(len(cum_rewards)), cum_rewards, label=name)
plt.axvline(x=SWITCH_ENV_AFTER, color='red', linestyle='--', label='Env Change')
plt.xlabel('Time steps')
plt.ylabel(f'Cumulative reward')
plt.title('Adaptation to Shortcut Maze')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# Draw the learned policies:
for name, agent in agents.items():
    draw_policy_grid(agent.Q, env.unwrapped.grid, ACTION_ARROWS_MAP, GRID_MAP, title=f'\nPolicy of {name}')
