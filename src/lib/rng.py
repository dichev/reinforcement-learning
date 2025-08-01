import torch

def stratified_draws(total, k):
    segment = total / k
    return (torch.arange(k) + torch.rand(k)) * segment
