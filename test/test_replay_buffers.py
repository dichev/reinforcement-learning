import pytest
import torch
from lib.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
import numpy as np


def generate_exp():
    ob = 0
    while True:
        action = np.random.randint(0, 5)
        reward = np.random.randint(0, 2)
        ob_next = ob + 1
        terminated = np.random.choice([True, False])
        truncated = np.random.choice([True, False])
        yield ob, action, reward, ob_next, terminated, truncated
        ob = ob_next


@pytest.mark.parametrize("capacity", [10, 100, 1_000])
@pytest.mark.parametrize("batch_size", [1, 10])
def test_disabled_priorities(capacity, batch_size):
    A = ReplayBuffer(capacity)
    B = PrioritizedReplayBuffer(capacity, alpha=0.)

    exp_gen = generate_exp()
    for _ in range(capacity + capacity//2):  # Fill beyond capacity
        ob, action, reward, ob_next, terminated, truncated = next(exp_gen)
        A.add(ob, action, reward, ob_next, terminated, truncated)
        B.add(ob, action, reward, ob_next, terminated, truncated)
    assert len(A) == len(B) == capacity

    # Update priorities in the prioritized buffer (should have no effect with alpha=0)
    B.sample(batch_size, replacement=True)
    B.update(priorities := torch.randn(batch_size) * 5)
    for exp_A, (p, exp_B) in zip(A.experiences, B.experiences):  # Test multiple samples
        print(exp_A, exp_B)
        assert exp_A == exp_B
        assert p == B._max_seen_priority == 1.



@pytest.mark.parametrize("capacity", [10, 100, 1_000])
@pytest.mark.parametrize("alpha", [0, .1, .9, .1])
@pytest.mark.parametrize("batch_size", [1, 10, 99])
def test_priorities_are_updated(capacity, alpha, batch_size):
    buffer = PrioritizedReplayBuffer(capacity, alpha)

    # Fill up the buffer (beyond its capacity)
    exp_gen = generate_exp()
    for i in range(capacity + capacity//2):
        ob, action, reward, ob_next, terminated, truncated = next(exp_gen)
        buffer.add(ob, action, reward, ob_next, terminated, truncated)
    assert len(buffer) == capacity
    obs = torch.tensor([exp.ob for p, exp in buffer.experiences])
    torch.testing.assert_close(obs, torch.arange(capacity) + capacity//2)

    # Sample and update priorities
    batch_size = 3
    buffer.sample(batch_size, replacement=False)
    indices = buffer._last_sampled
    priorities = torch.randn(batch_size) * 5
    buffer.update(priorities)
    torch.testing.assert_close(torch.tensor(buffer._max_seen_priority), torch.tensor(max(priorities.abs().max() ** alpha, 1.)))
    for idx, p_new in zip(indices, priorities):
        p, exp = buffer.experiences[idx]
        p_expected = (p_new.abs() + buffer.eps) ** alpha
        torch.testing.assert_close(torch.tensor(p), p_expected)



