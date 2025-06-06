import torch
import numpy as np

class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.total_rewards = 0
        self.steps = 0
        self.done = False

    def step(self, obs, action, reward):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_rewards += reward
        self.steps += 1

    def finish(self, as_tensors=True):
        self.done = True
        if as_tensors:
            self.observations = torch.tensor(np.vstack(self.observations))
            self.actions = torch.tensor(self.actions)
            self.rewards = torch.tensor(self.rewards)


    def __repr__(self):
        return f'Episode(steps={self.steps}, total_rewards={self.total_rewards}, done={self.done})'



@torch.no_grad()
def batched_episodes(env, policy, num_episodes):
    while True:
        batch = [play_episode(env, policy, as_tensors=True) for _ in range(num_episodes)]
        yield batch

@torch.no_grad()
def play_episode(env, policy, as_tensors=True):
    episode = Episode()
    obs, _ = env.reset()
    while True:
        action = policy(obs)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        episode.step(obs, action, reward)
        if terminated or truncated:
            episode.finish(as_tensors)
            return episode

        obs = obs_next
