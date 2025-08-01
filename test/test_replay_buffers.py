import pytest
import torch

from lib.data_structures import SumTree
from lib.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, PrioritizedReplayBufferTree
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


@pytest.mark.parametrize("capacity", [16, 1024, 2**14])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("use_sumtree", [False, True])
def test_disabled_priorities(capacity, batch_size, use_sumtree):
    A = ReplayBuffer(capacity)
    if use_sumtree:
        B = PrioritizedReplayBufferTree(capacity, alpha=0.)
    else:
        B = PrioritizedReplayBuffer(capacity, alpha=0.)

    exp_gen = generate_exp()
    for _ in range(capacity + capacity//3):  # Fill beyond capacity
        ob, action, reward, ob_next, terminated, truncated = next(exp_gen)
        A.add(ob, action, reward, ob_next, terminated, truncated)
        B.add(ob, action, reward, ob_next, terminated, truncated)
    assert len(A) == len(B) == capacity

    # Update priorities in the prioritized buffer (should have no effect with alpha=0)
    B.sample(batch_size)
    B.update(priorities := torch.randn(batch_size) * 5)
    for exp_A, (p, exp_B) in zip(A, B):  # Test multiple samples
        assert exp_A == exp_B
        assert p == 1.



@pytest.mark.parametrize("capacity", [16, 1024, 2**14])
@pytest.mark.parametrize("alpha", [0, .1, .9, .1])
@pytest.mark.parametrize("batch_size", [1, 10, 99])
@pytest.mark.parametrize("use_sumtree", [False, True])
def test_priorities_are_updated(capacity, alpha, batch_size, use_sumtree):
    if use_sumtree:
        replay = PrioritizedReplayBufferTree(capacity, alpha)
    else:
        replay = PrioritizedReplayBuffer(capacity, alpha)

    # Fill up the buffer (beyond its capacity)
    exp_gen = generate_exp()
    for i in range(capacity + capacity//3):
        ob, action, reward, ob_next, terminated, truncated = next(exp_gen)
        replay.add(ob, action, reward, ob_next, terminated, truncated)
    assert len(replay) == capacity
    obs = torch.tensor([exp.ob for p, exp in replay])
    torch.testing.assert_close(obs, torch.arange(capacity) + capacity//3)

    # Sample and update priorities
    batch_size = 3
    replay.sample(batch_size)
    indices = replay._last_sampled
    priorities = torch.randn(batch_size) * 5
    replay.update(priorities)
    torch.testing.assert_close(torch.tensor(replay._max_seen_priority), torch.tensor(max(priorities.abs().max() ** alpha, 1.)))

    no_duplicates = dict()  # sampling is with replacement, but we keep the last priority update only
    for idx, p in zip(indices, priorities):
        no_duplicates[idx] = p

    for idx, p_new in no_duplicates.items():
        p = replay.priorities[idx]
        p_expected = (p_new.abs() + replay.eps) ** alpha
        torch.testing.assert_close(torch.tensor(p), p_expected)
    torch.testing.assert_close(torch.tensor(replay._max_seen_priority), torch.tensor(max(priorities.abs().max() ** alpha, 1.)))

@pytest.mark.parametrize("capacity", [16, 1024, 2**14])
@pytest.mark.parametrize("usage", [0.1, 0.6, 1.])
@pytest.mark.parametrize("use_sumtree", [True, False])
def test_sampling_is_bounded(capacity, usage, use_sumtree):
    if use_sumtree:
        replay = PrioritizedReplayBufferTree(capacity, alpha=.9)
    else:
        replay = PrioritizedReplayBuffer(capacity, alpha=.9)

    usage_size = int(usage * capacity)

    # Fill up the buffer partially
    exp_gen = generate_exp()
    for i in range(usage_size):
        ob, action, reward, ob_next, terminated, truncated = next(exp_gen)
        replay.add(ob, action, reward, ob_next, terminated, truncated)
    assert len(replay) == usage_size

    replay.sample(batch_size=usage_size)
    indices = replay._last_sampled
    assert torch.all(torch.tensor(indices) < usage_size)



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

