from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym

from lib.rng import random_argmax

ENV_NAME   = "CliffWalking-v1"
GOAL       = -13    # optimal score
EPISODES   = 500
MAX_STEPS  = 5000   # truncate limit
EPSILON    = 0.05    # epsilon-greedy
ALPHA      = 0.4    # step size / learn rate
GAMMA      = 1.0    # decay


class Agent:
    def __init__(self, k_actions, n_states, eps=EPSILON):
        self.k_actions = k_actions
        self.n_states = n_states
        self.eps = eps
        self.Q = np.zeros((n_states, k_actions))

    def policy(self, s):
        if np.random.rand() < self.eps:
            return np.random.randint(self.k_actions)
        action = random_argmax(self.Q[s,:]) # arg max with tie breaking (since all Q were initialized to 0)
        return action

    def get_policy_probs(self, s):
        best_action = random_argmax(self.Q[s,:])
        p = np.full(self.k_actions, self.eps / self.k_actions)
        p[best_action] += 1 - self.eps
        assert np.isclose(p.sum(), 1.)
        return p


class SarsaAgent(Agent):
    def update(self, s, a, r, s_, a_, bootstrap):
        target = r + GAMMA * self.Q[s_, a_] if bootstrap else r
        self.Q[s,a] += ALPHA * (target - self.Q[s,a])


class ExpectedSarsaAgent(Agent):
    def update(self, s, a, r, s_, bootstrap):
        target = r + GAMMA * (self.get_policy_probs(s_) @ self.Q[s_, :]) if bootstrap else r
        self.Q[s,a] += ALPHA * (target - self.Q[s,a])


class QAgent(Agent):
    def update(self, s, a, r, s_, bootstrap):
        target = r + GAMMA * self.Q[s_,:].max() if bootstrap else r
        self.Q[s,a] += ALPHA * (target - self.Q[s,a])



# Prepare the env and agents
env = gym.make(ENV_NAME, max_episode_steps=MAX_STEPS)
k_actions, n_states = env.action_space.n, env.observation_space.n
agents = {
    'Sarsa': SarsaAgent(k_actions, n_states),
    'Q-learning': QAgent(k_actions, n_states),
    'Expected Sarsa': ExpectedSarsaAgent(k_actions, n_states),
}


# Learn policies
history = defaultdict(list)
for method, agent in agents.items():
    print(f'\n# {method}')
    avg_score = 0
    for episode in range(EPISODES):
        ob, info = env.reset()
        done = False
        action_next = None
        score = steps = 0
        while not done:
            if isinstance(agent, SarsaAgent): # Sarsa must select the next action before update (and then use the same action in the env step)
                action = agent.policy(ob) if action_next is None else action_next
                ob_next, reward, terminated, truncated, info = env.step(action)
                action_next = agent.policy(ob_next)
                agent.update(ob, action, reward, ob_next, action_next, bootstrap=not terminated)
            else:
                action = agent.policy(ob)
                ob_next, reward, terminated, truncated, info = env.step(action)
                agent.update(ob, action, reward, ob_next, bootstrap=not terminated)

            ob = ob_next
            done = terminated or truncated
            score += reward
            steps += 1

        avg_score = .9*avg_score + .1*score if episode > 0 else score
        history[method].append(avg_score)
        if episode % 50 == 0 or episode == EPISODES - 1:
            print(f'Episode #{episode+1} | {steps=}, {score=} | {avg_score=:.4f}')



# Plot results
for method, avg_scores in history.items():
    plt.plot(range(EPISODES), avg_scores, label=method)
plt.ylim(-100, 0)
plt.hlines(GOAL, 0, EPISODES, color="black", linestyles='dashed', label='Optimal')
plt.legend()
plt.xlabel('Episodes')
plt.xlabel('Episode')
plt.ylabel(f'Score (mov avg)')
plt.title(f'TD(0) learning over action values (with $\epsilon$-greedy={EPSILON})')
plt.legend()
plt.show()



# Preview one of the agents:
env = gym.make(ENV_NAME, render_mode="human")
ob, info = env.reset()
done = False
score = 0
agent = agents['Expected Sarsa']
while not done:
    action = agent.policy(ob)
    ob, reward, terminated, truncated, info = env.step(action)
    row, col = divmod(ob, ncols:=12)
    score += reward
    print(f'State: {ob:>2} ({row},{col}) | {reward=} | {score=}')
    done = terminated or truncated
env.close()

