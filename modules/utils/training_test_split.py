import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../'
import numpy as np
import torch

def to_torch(ndarray, device):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

def training_test_split(training_data, dimensions, device):
    N = len(training_data)
    p = np.random.permutation(N)
    training_data_shuffled = training_data[p[:]]
    
    input_len = dimensions + 1

    # split into train/val and convert to torch
    M = len(training_data_shuffled)
    split = int(0.8*M)
    x_train = to_torch(training_data_shuffled[:split, :input_len], device)
    y_train = to_torch(training_data_shuffled[:split, input_len:], device)
    x_val = to_torch(training_data_shuffled[split:, :input_len], device)
    y_val = to_torch(training_data_shuffled[split:, input_len:], device)

    return(x_train, y_train, x_val, y_val)