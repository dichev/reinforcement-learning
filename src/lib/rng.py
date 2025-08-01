import numpy as np
import torch
import random


def seed_global(seed: int, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)



def stratified_draws(total, k):
    segment = total / k
    draws = (torch.arange(k) + torch.rand(k)) * segment
    return draws.clip(0, total)  # ensure there will be no draws outside the range due to rounding errors

