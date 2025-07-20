import torch
import torch.nn.functional as F


@torch.jit.script
def discount_returns_loop(rewards, gamma: float):
    assert rewards.dtype == torch.float, f"Expected float reward but got {rewards.dtype}"
    T, B = rewards.shape

    returns = torch.empty_like(rewards)
    G = torch.tensor(0., device=rewards.device, dtype=torch.float)
    for t in range(T-1, -1, -1): # reversed(range(T)) but jit friendly
        G = rewards[t, :] + gamma * G
        returns[t, :] = G
    return returns


@torch.jit.script
def discount_n_steps_loop(rewards, gamma: float, n: int):  # ignores rewards past n future steps
    assert rewards.dtype == torch.float, f"Expected float reward but got {rewards.dtype}"
    T, B = rewards.shape

    discounts = gamma ** torch.arange(n, device=rewards.device, dtype=torch.float).view(n, 1)
    returns = torch.empty_like(rewards)
    for t in range(T):  # G_t = R_{t+1} + γ R_{t+2} + ... + γ^{n-1} R_{t+n}
        end = min(t + n, T)
        returns[t] = (rewards[t:end] * discounts[:end - t]).sum(dim=0)

    return returns



def discount_returns(rewards, gamma):
    assert rewards.dtype == torch.float, f"Expected float reward but got {rewards.dtype}"
    if gamma == 0: return rewards.clone()
    T, B = rewards.shape
    assert gamma ** T > torch.finfo(torch.float).tiny, "Zero discount detected due to precision issues, use discount_returns_loop instead"

    discounts = gamma ** torch.arange(T, device=rewards.device, dtype=torch.float).view(T, 1)
    R = rewards * discounts
    G = R.flip(0).cumsum(dim=0).flip(0) / discounts
    return G


def discount_n_steps(rewards, gamma, n):  # ignores rewards past n future steps
    assert rewards.dtype == torch.float, f"Expected float reward but got {rewards.dtype}"
    T, B = rewards.shape
    n = min(n, T)

    discounts = gamma ** torch.arange(n, device=rewards.device, dtype=torch.float)   # (n,)  [1, γ, γ², ...]
    kernel = discounts.view(1, 1, n)                   # (C_out=1, C_in=1, W=n)
    x = F.pad(rewards.T.view(B, 1, T), pad=(0, n-1))   # (B, C_in=1, W=T+n-1)
    y = F.conv1d(x, kernel)                            # (B, C_out=1, T)
    returns = y.squeeze(1).T                           # (T, B)
    return returns



def distributional_bellman(support, probs, reward, gamma):  # todo: handle terminal states and batch dim
    """
        Categorical Algorithm 1 from "A Distributional Perspective on Reinforcement Learning" (https://arxiv.org/pdf/1707.06887)
    """
    z = support
    N, v_min, v_max = len(z), support.min(), support.max()
    dz = (v_max - v_min) / (N - 1)

    # Compute the projection onto the support z:
    z_proj = (reward + gamma * z).clip(v_min, v_max)
    b = (z_proj - v_min) / dz  # b is a float bin position
    l = b.floor().long()
    u = b.ceil().long()

    # Distribute probability of the projection:
    m = torch.zeros_like(probs)
    u_ratio = b - l          # split the prob mass between the lower and upper bin
    l_ratio = 1. - u_ratio   # note when b is an integer (b=l=u) the whole mass goes to the lower bin
    m.index_add_(0, l, probs * l_ratio)
    m.index_add_(0, u, probs * u_ratio)

    return m



if __name__ == '__main__':
    from lib.utils import measure

    measure('Discount Precise   ', lambda : discount_returns_loop(torch.randn(1000, 2), .99))
    measure('Discount Vectorized', lambda : discount_returns(torch.randn(1000, 2), .99))
    print('-------------------------------------------------------------')
    measure('N-step discount Loop      ', lambda : discount_n_steps_loop(torch.randn(1000, 2), .99, n=100))
    measure('N-step discount Vectorized', lambda : discount_n_steps(torch.randn(1000, 2), .99, n=100))








