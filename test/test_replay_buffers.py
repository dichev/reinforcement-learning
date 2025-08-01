import pytest
import torch
from numpy.core.numeric import indices

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
        B = PrioritizedReplayBufferTree(capacity, alpha=0., beta=.4)
    else:
        B = PrioritizedReplayBuffer(capacity, alpha=0., beta=.4)

    exp_gen = generate_exp()
    for _ in range(capacity + capacity//3):  # Fill beyond capacity
        ob, action, reward, ob_next, terminated, truncated = next(exp_gen)
        A.add(ob, action, reward, ob_next, terminated, truncated)
        B.add(ob, action, reward, ob_next, terminated, truncated)
    assert len(A) == len(B) == capacity

    # Update priorities in the prioritized buffer (should have no effect with alpha=0)
    batch, indices, weights = B.sample(batch_size)
    B.update(indices, priorities := torch.randn(batch_size) * 5)
    for exp_A, (p, exp_B) in zip(A, B):  # Test multiple samples
        assert exp_A == exp_B
        assert p == 1.



@pytest.mark.parametrize("capacity", [16, 1024, 2**14])
@pytest.mark.parametrize("alpha", [0, .1, .9, .1])
@pytest.mark.parametrize("batch_size", [1, 10, 99])
@pytest.mark.parametrize("use_sumtree", [False, True])
def test_priorities_are_updated(capacity, alpha, batch_size, use_sumtree):
    if use_sumtree:
        replay = PrioritizedReplayBufferTree(capacity, alpha, beta=.4)
    else:
        replay = PrioritizedReplayBuffer(capacity, alpha, beta=.4)

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
    batch, indices, weights = replay.sample(batch_size)
    priorities = torch.randn(batch_size) * 5
    replay.update(indices, priorities)
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
        replay = PrioritizedReplayBufferTree(capacity, alpha=.9, beta=.4)
    else:
        replay = PrioritizedReplayBuffer(capacity, alpha=.9, beta=.4)

    usage_size = int(usage * capacity)

    # Fill up the buffer partially
    exp_gen = generate_exp()
    for i in range(usage_size):
        ob, action, reward, ob_next, terminated, truncated = next(exp_gen)
        replay.add(ob, action, reward, ob_next, terminated, truncated)
    assert len(replay) == usage_size

    batch, indices, weights = replay.sample(batch_size=usage_size)
    assert torch.all(torch.tensor(indices) < usage_size)



@pytest.mark.parametrize("capacity", [32, 1024, 2**14])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("use_sumtree", [True, False])
@pytest.mark.parametrize("beta", [0.0, 0.4, 1.0])
def test_prioritized_replay_importance_sampling(capacity, batch_size, use_sumtree, beta):
    if use_sumtree:
        replay = PrioritizedReplayBufferTree(capacity, alpha=1., beta=beta)
    else:
        replay = PrioritizedReplayBuffer(capacity, alpha=1., beta=beta)

    # Fill the buffer
    exp_gen = generate_exp()
    for _ in range(capacity):
        ob, action, reward, ob_next, terminated, truncated = next(exp_gen)
        replay.add(ob, action, reward, ob_next, terminated, truncated)

    # Set predefined priorities:
    priorities = torch.arange(capacity) * 10
    probs = priorities / priorities.sum()
    for i, p in enumerate(priorities.tolist()):
        replay.priorities[i] = p

    # Sample with importance sampling
    batch, indices, weights = replay.sample(batch_size)
    assert torch.all(weights > 0.) and torch.all(weights <= 1.)

    # Verify weights
    if beta == 0.0: # No importance sampling correction
        torch.testing.assert_close(weights, torch.ones_like(weights))
    elif beta == 1.0: # Full importance sampling correction
        weighted_probs = weights * probs[indices]
        weighted_probs /= weighted_probs.sum()
        expected_uniform = torch.ones_like(weighted_probs) / len(weighted_probs)
        torch.testing.assert_close(weighted_probs, expected_uniform)
    else: # For intermediate beta values weights should be proportional to importance sampling
        expected_weights = (replay.size * probs[indices]) ** (-beta)
        expected_weights /= expected_weights.max()
        torch.testing.assert_close(weights, expected_weights)



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

