import torch
import numpy as np
from modules.utils.get_lowest_gpu import *

# convert between numpy and torch
def to_torch(x):
    device = torch.device(get_lowest_gpu(pick_from=[0,1,2,3], verbose=False))
    return torch.from_numpy(x).float().to(device)
def to_numpy(x):
    device = torch.device(get_lowest_gpu(pick_from=[0,1,2,3], verbose=False))
    return x.detach().cpu().numpy()
def to_torch_cpu(x):
    device = torch.device('cpu')
    return torch.from_numpy(x).float().to(device)