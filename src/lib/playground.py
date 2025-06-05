import torch

class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.total_rewards = 0
        self.steps = 0

    def step(self, obs, action, reward):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_rewards += reward
        self.steps += 1

    def __repr__(self):
        return f'Episode(steps={self.steps}, total_rewards={self.total_rewards})'



@torch.no_grad()
def batched_episodes(env, policy, num_episodes):
    while True:
        batch = [play_episode(env, policy) for _ in range(num_episodes)]
        yield batch

@torch.no_grad()
def play_episode(env, policy):
    episode = Episode()
    obs, _ = env.reset()
    while True:
        action = policy(obs)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        episode.step(obs, action, reward)
        if terminated or truncated:
            return episode

        obs = obs_next
