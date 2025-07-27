import random
from math import inf

from lib.data_structures import CircularBuffer
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
        obs, actions, rewards, obs_next, done = to_tensors(batch, device=device)
        return obs, actions, rewards, obs_next, done

    def __len__(self):
        return len(self.experiences)

    def __repr__(self):
        return f'ReplayBuffer(size={len(self)}, capacity={self.capacity})'




if __name__ == '__main__':
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="human")
    replay = ReplayBuffer(capacity=1000)
    print(replay)
    for ob, action, reward, ob_next, terminated, truncated in play_steps(env, 100, lambda ob: env.action_space.sample()):
        replay.add(ob, action, reward, ob_next, terminated, truncated)
    batch = replay.sample(batch_size=10)
    env.close()
