import pytest
import torch

from lib.data_structures import SumTree
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
    for exp_A, exp_B in zip(A.experiences, B.experiences):  # Test multiple samples
        print(exp_A, exp_B)
        assert exp_A == exp_B
    torch.testing.assert_close(B.priorities.get_data(), torch.ones(capacity))



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
    obs = torch.tensor([exp.ob for exp in buffer.experiences])
    torch.testing.assert_close(obs, torch.arange(capacity) + capacity//2)

    # Sample and update priorities
    batch_size = 3
    buffer.sample(batch_size, replacement=False)
    indices = buffer._last_sampled
    priorities = torch.randn(batch_size) * 5
    buffer.update(priorities)
    torch.testing.assert_close(torch.tensor(buffer._max_seen_priority), torch.tensor(max(priorities.abs().max() ** alpha, 1.)))
    for idx, p_new in zip(indices, priorities):
        p = buffer.priorities[idx]
        p_expected = (p_new.abs() + buffer.eps) ** alpha
        torch.testing.assert_close(torch.tensor(p), p_expected)



@pytest.mark.parametrize("capacity",    [2, 2**3, 2**10, 2**16])
@pytest.mark.parametrize("values_mplr", [10., 0.001, 3.33])
def test_full_sumtree(capacity, values_mplr):
    tree = SumTree(capacity=capacity)

    # Fill up the tree
    values = (np.arange(capacity) * values_mplr).tolist()
    for i, val in enumerate(values):
        tree.update(i, val)
    assert values == tree.get_data()
    assert abs(tree.total_sum - sum(values)) < 1e-6
    assert abs(tree.total_sum - sum(tree.get_data())) < 1e-6

    # Verify tree sums
    for i in reversed(range(tree.first_leaf)):
        left  = 2*i + 1
        right = 2*i + 2
        expected_sum = tree.nodes[left] + tree.nodes[right]
        assert abs(tree.nodes[i] - expected_sum) < 1e-6

    # Test update functionality
    prev_sum = tree.total_sum
    extra_value = values_mplr * 10 #np.random.randn() * values_mplr
    tree.update(0, extra_value)
    assert abs(tree.total_sum - (prev_sum + extra_value)) < 1e-6


@pytest.mark.parametrize("capacity", [2, 2**3, 2**10, 2**16])
@pytest.mark.parametrize("values_mplr", [10., 0.001, 3.33])
def test_sumtree_select(capacity, values_mplr):
    tree = SumTree(capacity=capacity)

    # Fill up the tree
    values = (np.arange(capacity) * values_mplr).tolist()
    for i, val in enumerate(values):
        tree.update(i, val)

    # Test selecting by priority mass
    s = 0.
    priorities = tree.get_data()
    for i, p in enumerate(priorities):
        s += p

        delta = values_mplr / 2
        # less than p mass
        if i > 0:
            idx = tree.query(s - delta)
            p_selected = tree.get(idx)
            assert i == idx and p == p_selected, f"Expected index {i} and priority {p}, but got index {idx} and {p_selected}"

        # more than p mass, select next p
        if i < len(priorities)-1:
            idx = tree.query(s + delta)
            p_selected = tree.get(idx)
            assert i+1 == idx and priorities[i+1] == p_selected, f"Expected index {i+1} and priority {priorities[i+1]}, but got index {idx} and {p_selected}"

