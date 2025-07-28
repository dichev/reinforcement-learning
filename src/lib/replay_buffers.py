import random
import torch
from math import inf
from lib.data_structures import CircularBuffer, CircularTensor
from lib.playground import Exp, to_tensors, play_steps


class Stats:
    def __init__(self):
        self.avg_score = 0.
        self.best_score = -inf
        self.avg_episode_length = 0
        self._episodes = 0
        self._last_rewards = []

    def update(self, reward, terminated, truncated):
        self._last_rewards.append(reward)
        if terminated or truncated:
            score, length = sum(self._last_rewards), len(self._last_rewards)
            self.best_score = max(score, self.best_score)
            self.avg_score = (.9 * score + .1 * self.avg_score) if self._episodes else score
            self.avg_episode_length = (.9 * self.avg_episode_length + .1 * length) if self._episodes else length
            self._episodes += 1
            self._last_rewards.clear()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.experiences = CircularBuffer(capacity)  # O(1) random access, note deque has O(n) random access
        self.stats = Stats()

    def add(self, ob, action, reward, ob_next, terminated, truncated=None):
        exp = Exp(ob, action, reward, ob_next, terminated) # note: we ignore truncated states, since the agent shouldn't treat them as done state
        self.experiences.append(exp)
        self.stats.update(reward, terminated, truncated)


    def sample(self, batch_size, device=None):
        assert len(self.experiences) >= batch_size, f"Replay buffer has {len(self.experiences)} steps, but batch_size={batch_size}"
        batch = random.sample(self.experiences, batch_size)
        obs, actions, rewards, obs_next, dones = to_tensors(batch, device=device)
        return obs, actions, rewards, obs_next, dones

    def __len__(self):
        return len(self.experiences)

    def __repr__(self):
        return f'ReplayBuffer(size={len(self)}, capacity={self.capacity})'


class PrioritizedReplayBuffer: # with proportional prioritization
    """
    Paper: Prioritized Replay Buffer
    https://arxiv.org/pdf/1511.05952
    """

    def __init__(self, capacity, alpha, eps=1e-6):
        """
        :param int capacity: The maximum number of experiences the buffer can store
        :param float alpha: Determines how much prioritization is used, with α = 0 corresponding to the uniform case
        :param float eps: Prevents zero (loss) value of priorities
        """
        assert 0 <= alpha <= 1, f"Alpha must be in the range [0, 1], but got alpha={alpha}"

        self.capacity = capacity
        self.experiences = CircularBuffer(capacity)  # O(1) random access, note deque has O(n) random access
        self.priorities = CircularTensor(capacity, dtype=torch.float)
        self.stats = Stats()
        self.alpha = alpha
        self.eps = eps
        self._last_sampled = None  # store internally the sampled indices to update their priorities later
        self._max_seen_priority = 1.

    def add(self, ob, action, reward, ob_next, terminated, truncated=None):
        assert self._last_sampled is None, "Unexpected behavior"
        exp = Exp(ob, action, reward, ob_next, terminated) # note: we ignore truncated states, since the agent shouldn't treat them as done state
        self.experiences.append(exp)
        self.priorities.append(self._max_seen_priority)
        self.stats.update(reward, terminated, truncated)

    def sample(self, batch_size, replacement=True, device=None):
        assert len(self.experiences) >= batch_size, f"Replay buffer has {len(self.experiences)} steps, but batch_size={batch_size}"
        assert self._last_sampled is None, "You must call update() before calling sample()"

        # O(n) but can be optimized with sum-tree to O(log n)
        p = self.priorities.get_data()
        probs = p / p.sum()
        indices = torch.multinomial(probs, batch_size, replacement).tolist()
        self._last_sampled = indices

        batch = [self.experiences[i] for i in indices]
        obs, actions, rewards, obs_next, dones = to_tensors(batch, device=device)
        return obs, actions, rewards, obs_next, dones  # todo: weights for important sampling

    def update(self, priorities):
        assert self._last_sampled is not None, "You must call sample() before calling update()"
        assert len(priorities) == len(self._last_sampled), f"The number of priorities: {len(priorities)}, must match the number of last sampled indices {len(self._last_sampled)}"

        priorities = (priorities.abs() + self.eps) ** self.alpha
        for idx, priority in zip(self._last_sampled, priorities.tolist()):
            self.priorities[idx] = priority
            self._max_seen_priority = max(self._max_seen_priority, priority)   # note that ignores the evicted experiences, which will cause the max priority to stale

        self._last_sampled = None


    def __len__(self):
        return len(self.experiences)

    def __repr__(self):
        return f'PrioritizedReplayBuffer(size={len(self)}, capacity={self.capacity}, alpha={self.alpha})'




if __name__ == '__main__':
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    replay = PrioritizedReplayBuffer(capacity=20, alpha=.6)
    print(replay)
    for ob, action, reward, ob_next, terminated, truncated in play_steps(env, 10, lambda ob: env.action_space.sample()):
        replay.add(ob, action, reward, ob_next, terminated, truncated)
    batch_size = 10
    batch = replay.sample(batch_size)
    replay.update(torch.rand(batch_size))
    env.close()
