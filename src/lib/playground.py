import torch
import numpy as np
from collections import namedtuple
import math


Exp = namedtuple('Exp', field_names=['ob', 'action', 'reward', 'ob_next', 'done'])

class Episode:
    def __init__(self):
        self.experiences = []
        self.total_rewards = 0
        self.done = False

    def step(self, obs, action, reward, obs_next, done):
        exp = Exp(obs, action, reward, obs_next, done)
        self.experiences.append(exp)
        self.total_rewards += reward
        self.done = done

    def as_tensors(self, device=None):
        obs, actions, rewards, obs_next, done = to_tensors(self.experiences)
        return obs, actions, rewards, obs_next, done

    def __len__(self):
        return len(self.experiences)

    def __repr__(self):
        return f'Episode(steps={len(self)}, total_rewards={self.total_rewards}, done={self.done})'


def to_tensors(experiences: list[Exp], device=None): # unified tensor formats
    obs, actions, rewards, obs_next, done = zip(*experiences)
    T = len(experiences)
    obs      = torch.tensor(np.stack(obs),      dtype=torch.float, device=device)              # T, *S
    actions  = torch.tensor(actions,            dtype=torch.long,  device=device).view(T, 1)   # T, 1
    rewards  = torch.tensor(rewards,            dtype=torch.float, device=device).view(T, 1)   # T, 1
    obs_next = torch.tensor(np.stack(obs_next), dtype=torch.float, device=device)              # T, *S
    done     = torch.tensor(np.stack(done),     dtype=torch.long,  device=device).view(T, 1)   # T, 1
    return obs, actions, rewards, obs_next, done


@torch.no_grad()
def batched_episodes(env, policy, num_episodes):
    while True:
        batch = [play_episode(env, policy) for _ in range(num_episodes)]
        yield batch

@torch.no_grad()
def play_episode(env, policy, max_steps=math.inf):
    episode = Episode()
    ob, _ = env.reset()
    while True:
        action = policy(ob)
        ob_next, reward, terminated, truncated, _ = env.step(action)
        episode.step(ob, action, reward, ob_next, terminated)
        if terminated or truncated or len(episode) >= max_steps:
            return episode
        ob = ob_next


def play_steps(env, max_steps=None, policy=None):
    steps = 0
    ob, _ = env.reset()
    while True:
        action = env.action_space.sample() if policy is None else policy(ob)
        ob_next, reward, terminated, truncated, _ = env.step(action)
        yield ob, action, reward, ob_next, terminated, truncated
        if terminated or truncated:
            ob_next, _ = env.reset()
        ob = ob_next
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break


@torch.no_grad()
def evaluate_policy_agent(env, agent, n_episodes, device=None):
    values = scores = episode_length = 0
    for i in range(n_episodes):
        episode = play_episode(env, agent.policy)
        scores += episode.total_rewards
        episode_length += len(episode)
        obs = torch.tensor(np.stack([exp[0] for exp in episode.experiences]), device=device, dtype=torch.float)
        Q_max = agent.q_values(obs).max(dim=-1)[0].mean().item()  # max Q values averaged over each episode
        values += Q_max
    score, episode_length, value = [s/n_episodes for s in (scores, episode_length, values)]
    return score, episode_length, value


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="human")
    episode = play_episode(env, lambda ob: env.action_space.sample())
    print(f"Episode finished with reward {episode.total_rewards}")
    env.close()
