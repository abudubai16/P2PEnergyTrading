from typing import Dict

import torch
import numpy as np


def to_torch(x, device="cpu"):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_torch(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_torch(v, device) for v in x]
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    else:
        return torch.tensor(x).to(device)


def print_size(x: Dict[str, torch.Tensor]):
    t = {}
    for k, v in x.items():
        t[k] = v.shape
    print(t)
