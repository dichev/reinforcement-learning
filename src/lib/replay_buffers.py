import random
import torch
from math import inf
from lib.data_structures import CircularBuffer, SumTree
from lib.playground import Exp, to_tensors, play_steps
from lib.rng import stratified_draws


class Stats:
    def __init__(self):
        self.avg_score = 0.
        self.best_score = -inf
        self.avg_episode_length = 0
        self.episodes = 0
        self._last_rewards = []

    def update(self, reward, terminated, truncated):
        self._last_rewards.append(reward)
        if terminated or truncated:
            score, length = sum(self._last_rewards), len(self._last_rewards)
            self.best_score = max(score, self.best_score)
            self.avg_score = (.9 * self.avg_score + .1 * score) if self.episodes else score
            self.avg_episode_length = (.9 * self.avg_episode_length + .1 * length) if self.episodes else length
            self.episodes += 1
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
        indices = torch.randint(len(self), (batch_size,)).tolist()  # with replacement
        batch = [self.experiences[i] for i in indices]
        obs, actions, rewards, obs_next, dones = to_tensors(batch, device=device)
        return obs, actions, rewards, obs_next, dones

    def __iter__(self):
        for i in range(len(self)):
            yield self.experiences[i]

    def __len__(self):
        return len(self.experiences)

    def __repr__(self):
        return f'ReplayBuffer(size={len(self)}, capacity={self.capacity})'


class PrioritizedReplayBuffer: # with proportional prioritization
    """
    Paper: Prioritized Replay Buffer
    https://arxiv.org/pdf/1511.05952
    """

    def __init__(self, capacity, alpha, beta, eps=1e-6, priorities_storage=None):
        """
        :param int capacity: The maximum number of experiences the buffer can store
        :param float alpha: Controls how much prioritization is used (α = 0 is uniform sampling)
        :param float beta: Controls how much importance sampling weights affect learning (β = 0 is no correction)
        :param float eps: Prevents zero (loss) value of priorities
        :param priorities_storage: Provide custom priorities storage (e.g. SumTree), otherwise torch tensor will be used
        """
        assert 0 <= alpha <= 1, f"Alpha must be in the range [0, 1], but got alpha={alpha}"

        self.capacity = capacity
        self.experiences = [None] * capacity
        self.priorities = torch.empty(capacity, dtype=torch.float) if priorities_storage is None else priorities_storage
        self.stats = Stats()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self._max_seen_priority = 1.

        self.pos  = 0  # pos is the writing head used for circular buffering
        self.size = 0  # of the experiences and priorities


    def add(self, ob, action, reward, ob_next, terminated, truncated=None):
        exp = Exp(ob, action, reward, ob_next, terminated) # note: we ignore truncated states, since the agent shouldn't treat them as done state

        self.experiences[self.pos] = exp
        self.priorities[self.pos] = self._max_seen_priority
        self.stats.update(reward, terminated, truncated)

        self.pos = (self.pos + 1) % self.capacity    # circular list
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device=None):
        assert self.size >= batch_size, f"Replay buffer has {self.size} steps, but batch_size={batch_size}"

        # O(n) but can be optimized with sum-tree to O(log n)
        p = self.priorities[:self.size] if self.size < self.capacity else self.priorities
        probs = p / p.sum()
        indices = torch.multinomial(probs, batch_size, replacement=True).tolist()

        # (weighted) importance sampling w = (1/N 1/P)^β
        weights = (self.size * probs[indices]) ** -self.beta
        weights /= weights.max()  # normalize so that they only scale the update downwards

        batch = [self.experiences[i] for i in indices]
        weights = weights.view(batch_size, 1)  # important: reshape to (B,1) to avoid unexpected broadcasting
        return to_tensors(batch, device=device), indices, weights.to(device)

    def update(self, indices, priorities):
        priorities = (priorities.abs() + self.eps) ** self.alpha
        for idx, priority in zip(indices, priorities.tolist()):  # O(k x log n)
            self.priorities[idx] = priority
            self._max_seen_priority = max(self._max_seen_priority, priority)   # note that ignores the evicted experiences, which will cause the max priority to stale

    def get_priorities(self):
        return self.priorities

    def __iter__(self):
        for i in range(self.size):
            idx = (self.pos + i) % self.capacity if self.size == self.capacity else i
            yield self.priorities[idx], self.experiences[idx]

    def __len__(self):
        return self.size

    def __repr__(self):
        return f'{self.__class__.__name__}(size={len(self)}, capacity={self.capacity}, alpha={self.alpha})'



class PrioritizedReplayBufferTree(PrioritizedReplayBuffer):
    """
        k x O(log n) sampling and updating with SumTree data structure
    """
    def __init__(self, capacity, alpha, beta, eps=1e-6):
        super().__init__(capacity, alpha, beta, eps, priorities_storage=SumTree(capacity))

    def sample(self, batch_size, device=None):
        assert self.size >= batch_size, f"Replay buffer has {self.size} steps, but batch_size={batch_size}"

        # Efficient sampling: k x O(log n), (but note updating a priority is also O(log n))
        draws = stratified_draws(self.priorities.total_sum, batch_size)
        indices = [self.priorities.query(r) for r in draws.tolist()]

        # (weighted) importance sampling: w = (1/N 1/P)^β
        priorities = torch.tensor([self.priorities[i] for i in indices], device=device)
        probs = priorities / self.priorities.total_sum
        weights = (self.size * probs) ** -self.beta
        weights /= weights.max()  # normalize so that they only scale the update downwards

        batch = [self.experiences[i] for i in indices]
        weights = weights.view(batch_size, 1)  # important: reshaping to (B,1) to avoid unexpected broadcasting later
        return to_tensors(batch, device=device), indices, weights.to(device)

    def get_priorities(self):
        return torch.tensor(self.priorities.get_data())


if __name__ == '__main__':
    import random
    from lib.utils import measure
    import numpy as np

    capacity = 2**17 # > 100_000
    A = ReplayBuffer(capacity)
    B = PrioritizedReplayBuffer(capacity, alpha=.6, beta=.4)
    C = PrioritizedReplayBufferTree(capacity, alpha=.6, beta=.4)
    for i in range(capacity):  # Fill capacity
        exp = dict(
            ob = i,
            action = np.random.randint(0, 5),
            reward = np.random.randint(0, 2),
            ob_next = i + 1,
            terminated = np.random.choice([True, False]),
            truncated = np.random.choice([True, False]),
        )
        A.add(**exp)
        B.add(**exp)
        C.add(**exp)

    batch_size = 32
    p = torch.rand(batch_size)
    indices = random.sample(range(capacity), batch_size)
    measure('ReplayBuffer                | add & sample         ', number=1000, fn=lambda : [A.add(**exp), a := A.sample(batch_size)] )                     # O(1)     + k x O(n)
    measure('PrioritizedReplayBuffer     | add & sample & update', number=1000, fn=lambda : [B.add(**exp), b := B.sample(batch_size), B.update(b[1], p)] )  # O(1)     + k x O(n)     + k x O(1)
    measure('PrioritizedReplayBufferTree | add & sample & update', number=1000, fn=lambda : [C.add(**exp), c := C.sample(batch_size), C.update(c[1], p)] )  # O(log n) + k x O(log n) + k x O(log n)

