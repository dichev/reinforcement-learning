import pytest
import torch
from lib.calc import discount_returns_precise, discount_returns



@pytest.mark.parametrize("gamma", [0.0, 0.1, 0.9, 0.99, 1.0])
@pytest.mark.parametrize("rewards", [torch.arange(30).view(-1, 1).float(), torch.randn(30, 2)])
def test_discount_functions_match(gamma, rewards):
    A = discount_returns_precise(rewards, gamma)
    B = discount_returns(rewards, gamma)

    assert torch.allclose(A, B, atol=1e-5)
    if gamma == 0:
        assert torch.allclose(A, rewards.float())
        assert torch.allclose(B, rewards.float())
    elif gamma == 1:
        assert torch.allclose(A, rewards.flip(0).cumsum(dim=0).flip(0).float())
        assert torch.allclose(B, rewards.flip(0).cumsum(dim=0).flip(0).float())