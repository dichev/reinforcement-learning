import torch


@torch.jit.script
def discount_returns_precise(rewards, gamma: float):
    assert rewards.dtype == torch.float, f"Expected float reward but got {rewards.dtype}"
    T, B = rewards.shape

    returns = torch.empty_like(rewards)
    G = torch.tensor(0., device=rewards.device)
    for t in range(T-1, -1, -1): # reversed(range(T)) but jit friendly
        G = rewards[t, :] + gamma * G
        returns[t, :] = G
    return returns


def discount_returns(rewards, gamma):
    assert rewards.dtype == torch.float, f"Expected float reward but got {rewards.dtype}"
    if gamma == 0: return rewards.clone()
    T, B = rewards.shape
    assert gamma ** T > torch.finfo(torch.float).tiny, "Zero discount detected due to precision issues, use discount_returns_precise"

    discounts = gamma ** torch.arange(T, device=rewards.device).view(T, 1)
    R = rewards * discounts
    G = R.flip(0).cumsum(dim=0).flip(0) / discounts
    return G


if __name__ == '__main__':
    import timeit
    print('Precise:   ', timeit.timeit(lambda : discount_returns_precise(torch.randn(1000, 2), .99), number=100, globals=globals()))
    print('Vectorized:', timeit.timeit(lambda : discount_returns(torch.randn(1000, 2), .99), number=100, globals=globals()))

