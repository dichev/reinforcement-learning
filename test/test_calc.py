import pytest
import torch
from lib.calc import discount_returns_loop, discount_returns, discount_n_steps_loop, discount_n_steps


@pytest.fixture(params=[
    torch.ones(10, 1).float(),
    torch.ones(10, 2).float(),
    torch.arange(30).view(-1, 1).float(),
    torch.randn(30, 2).float(),
], ids=["ones", "ones_batch", "arange", "randn"])
def rewards(request):
    return request.param



@pytest.mark.parametrize("gamma", [0, .1, .9, .99, 1])
def test_discount_functions_match(gamma, rewards):
    A = discount_returns_loop(rewards, gamma)
    B = discount_returns(rewards, gamma)

    torch.testing.assert_close(A, B)
    if gamma == 0:
        torch.testing.assert_close(A, rewards)
        torch.testing.assert_close(B, rewards)
    elif gamma == 1:
        torch.testing.assert_close(A, rewards.flip(0).cumsum(dim=0).flip(0).float())
        torch.testing.assert_close(B, rewards.flip(0).cumsum(dim=0).flip(0).float())


@pytest.mark.parametrize("gamma", [0, .1, .9, .99, 1])
@pytest.mark.parametrize("n", [1, 5, 10, 30, 50])
def test_discount_functions_match(gamma, n, rewards):
    A = discount_n_steps_loop(rewards, gamma, n)
    B = discount_n_steps(rewards, gamma, n)

    torch.testing.assert_close(A, B)
    if gamma == 0:
        torch.testing.assert_close(A, rewards)
        torch.testing.assert_close(B, rewards)
    elif n >= len(rewards):
        C = discount_returns_loop(rewards, gamma)
        torch.testing.assert_close(A, C)
        torch.testing.assert_close(B, C)
