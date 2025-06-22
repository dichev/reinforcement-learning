import torch
import numpy as np

class Episode:
    def __init__(self):
        self.actions = []
        self.rewards = []
        self.observations = []
        self.observations_next = []
        self.total_rewards = 0
        self.steps = 0
        self.done = False

    def step(self, obs, action, reward, obs_next, done):
        self.observations.append(obs)
        self.observations_next.append(obs_next)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_rewards += reward
        self.steps += 1
        self.done = done

    def as_tensors(self, device=None):
        obs = torch.tensor(np.stack(self.observations), device=device)
        actions = torch.tensor(self.actions, device=device)
        rewards = torch.tensor(self.rewards, device=device)
        obs_next = torch.tensor(np.stack(self.observations), device=device)
        done = torch.tensor(self.done, device=device, dtype=torch.long)
        return obs, actions, rewards, obs_next, done

    def as_trajectory(self):
        for i in range(self.steps):
            obs = self.observations[i]
            action = self.actions[i]
            reward = self.rewards[i]
            obs_next = self.observations_next[i]
            yield obs, action, reward, obs_next, self.done


    def __repr__(self):
        return f'Episode(steps={self.steps}, total_rewards={self.total_rewards}, done={self.done})'



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
        episode.step(obs, action, reward, obs_next, terminated)
        if terminated or truncated:
            return episode

        obs = obs_next

def play_random(env, steps=1):
    obs, _ = env.reset()
    for t in range(steps):
        action = env.action_space.sample()
        obs_next, reward, terminated, truncated, _ = env.step(action)
        yield obs, action, reward, obs_next
        if terminated or truncated:
            obs_next, _ = env.reset()
        obs = obs_next


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="human")
    episode = play_episode(env, lambda obs: env.action_space.sample())
    print(f"Episode finished with reward {episode.total_rewards}")
    trajectory = list(episode.as_trajectory())
    print(trajectory)
    env.close()
