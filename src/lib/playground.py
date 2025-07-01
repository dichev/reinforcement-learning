import torch
import numpy as np
from collections import deque
import random, math

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
        T = self.steps
        obs = torch.tensor(np.stack(self.observations), device=device)         # T, S
        actions = torch.tensor(self.actions, device=device).view(T, 1)         # T, 1
        rewards = torch.tensor(self.rewards, device=device).view(T, 1)         # T, 1
        obs_next = torch.tensor(np.stack(self.observations), device=device)    # T, S
        done = torch.zeros((T, 1), device=device, dtype=torch.long)            # T, 1
        done[-1] = self.done                                                   # where only the last step can be terminal
        return obs, actions, rewards, obs_next, done

    def as_trajectory(self):
        for i in range(self.steps):
            obs = self.observations[i]
            action = self.actions[i]
            reward = self.rewards[i]
            obs_next = self.observations_next[i]
            done = self.done if (i == self.steps - 1) else False
            yield obs, action, reward, obs_next, done


    def __repr__(self):
        return f'Episode(steps={self.steps}, total_rewards={self.total_rewards}, done={self.done})'



class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.steps = deque(maxlen=capacity)

    def add(self, episode: Episode):
        for obs, action, reward, obs_next, done in episode.as_trajectory():
            self.add_step(obs, action, reward, obs_next, done, False)

    def add_step(self, obs, action, reward, obs_next, terminated, truncated):
        self.steps.append((obs, action, reward, obs_next, terminated))


    def sample(self, batch_size, device=None):
        assert len(self.steps) >= batch_size, f"Replay buffer has {len(self.steps)} steps, but batch_size={batch_size}"
        batch = random.sample(self.steps, batch_size)
        obs, actions, rewards, obs_next, done = zip(*batch)

        obs = torch.tensor(np.stack(obs), dtype=torch.float, device=device).view(batch_size, -1)
        actions = torch.tensor(actions, dtype=torch.long, device=device).view(batch_size, -1)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device).view(batch_size, -1)
        obs_next = torch.tensor(np.stack(obs_next), dtype=torch.float, device=device).view(batch_size, -1)
        done = torch.tensor(done, dtype=torch.long, device=device).view(batch_size, -1)

        return obs, actions, rewards, obs_next, done

    @property
    def size(self):
        return len(self.steps)

    def __repr__(self):
        return f'ReplayBuffer(size={self.size}, capacity={self.capacity})'



@torch.no_grad()
def batched_episodes(env, policy, num_episodes):
    while True:
        batch = [play_episode(env, policy) for _ in range(num_episodes)]
        yield batch

@torch.no_grad()
def play_episode(env, policy, max_steps=math.inf):
    episode = Episode()
    obs, _ = env.reset()
    while True:
        action = policy(obs)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        episode.step(obs, action, reward, obs_next, terminated)
        if terminated or truncated or episode.steps >= max_steps:
            return episode
        obs = obs_next


def play_steps(env, max_steps=None, policy=None):
    steps = 0
    obs, _ = env.reset()
    while True:
        action = env.action_space.sample() if policy is None else policy(obs)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        yield obs, action, reward, obs_next, terminated, truncated
        if terminated or truncated:
            obs_next, _ = env.reset()
        obs = obs_next
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="human")
    episode = play_episode(env, lambda obs: env.action_space.sample())
    print(f"Episode finished with reward {episode.total_rewards}")
    trajectory = list(episode.as_trajectory())
    print(trajectory)
    replay = ReplayBuffer(capacity=1000)
    replay.add(episode)
    print(replay)
    batch = replay.sample(batch_size=10)
    env.close()
