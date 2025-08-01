import torch

def stratified_draws(total, k):
    segment = total / k
    draws = (torch.arange(k) + torch.rand(k)) * segment
    return draws.clip(0, total)  # ensure there will be no draws outside the range due to rounding errors
