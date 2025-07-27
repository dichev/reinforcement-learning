import random
import numpy as np

from lib.data_structures import CircularBuffer
from lib.playground import Exp, to_tensors, play_steps


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.experiences = CircularBuffer(capacity)  # O(1) random access, note deque has O(n) random access
        self.stats = {
            'avg_score': 0.,
            'best_score': -np.inf,
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
        obs, actions, rewards, obs_next, done = to_tensors(batch, device=device)
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




if __name__ == '__main__':
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="human")
    replay = ReplayBuffer(capacity=1000)
    print(replay)
    for ob, action, reward, ob_next, terminated, truncated in play_steps(env, 100, lambda ob: env.action_space.sample()):
        replay.add(ob, action, reward, ob_next, terminated, truncated)
    batch = replay.sample(batch_size=10)
    env.close()
