import torch
import numpy as np
from collections import deque, namedtuple
import random, math

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
        T = len(self.experiences)
        obs, actions, rewards, obs_next, done = zip(*self.experiences)
        obs = torch.tensor(np.stack(obs), device=device)                  # T, *S
        actions = torch.tensor(actions, device=device).view(T, 1)         # T, 1
        rewards = torch.tensor(rewards, device=device).view(T, 1)         # T, 1
        obs_next = torch.tensor(np.stack(obs_next), device=device)        # T, *S
        done = torch.tensor(np.stack(done), device=device).view(T, 1)     # T, 1
        return obs, actions, rewards, obs_next, done

    def __len__(self):
        return len(self.experiences)

    def __repr__(self):
        return f'Episode(steps={len(self)}, total_rewards={self.total_rewards}, done={self.done})'



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.experiences = deque(maxlen=capacity)
        self.stats = {
            'avg_score': 0.,
            'best_score': 0.,
            'avg_episode_length': 0,
        }
        self._last_rewards = []
        self._episodes = 0

    def add(self, ob, action, reward, ob_next, terminated, truncated=None):
        exp = Exp(ob, action, reward, ob_next, terminated) # note: we ignore truncated states, since the agent shouldn't treat them as done state
        self.experiences.append(exp)
        self._last_rewards.append(reward)
        if terminated or truncated:
            self._update_stats()

    def sample(self, batch_size, device=None):
        assert len(self.experiences) >= batch_size, f"Replay buffer has {len(self.experiences)} steps, but batch_size={batch_size}"
        batch = random.sample(self.experiences, batch_size)
        obs, actions, rewards, obs_next, done = zip(*batch)

        obs = torch.tensor(np.stack(obs), dtype=torch.float, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).view(batch_size, -1)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device).view(batch_size, -1)
        obs_next = torch.tensor(np.stack(obs_next), dtype=torch.float, device=device)
        done = torch.tensor(done, dtype=torch.long, device=device).view(batch_size, -1)

        return obs, actions, rewards, obs_next, done

    def _update_stats(self):
        rewards = self._last_rewards
        score, length = sum(rewards), len(rewards)
        self.stats['best_score'] = max(score, self.stats['best_score'])
        self.stats['avg_score'] = (.9 * score + .1 * self.stats['avg_score']) if self._episodes else score
        self.stats['avg_episode_length'] = (.9 * self.stats['avg_episode_length'] + .1 * length) if self._episodes else length
        self._episodes += 1
        self._last_rewards.clear()

    def __len__(self):
        return len(self.experiences)

    def __repr__(self):
        return f'ReplayBuffer(size={len(self)}, capacity={self.capacity})'



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
        obs = torch.tensor(np.stack([exp[0] for exp in episode.experiences]), device=device)
        Q_max = agent(obs).max(dim=-1)[0].mean().item()  # max Q values averaged over each episode
        values += Q_max
    score, episode_length, value = [s/n_episodes for s in (scores, episode_length, values)]
    return score, episode_length, value


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="human")
    episode = play_episode(env, lambda ob: env.action_space.sample())
    print(f"Episode finished with reward {episode.total_rewards}")
    replay = ReplayBuffer(capacity=1000)
    replay.add(episode)
    print(replay)
    batch = replay.sample(batch_size=10)
    env.close()
