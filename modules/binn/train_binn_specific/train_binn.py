import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn.model_wrapper import model_wrapper
from modules.binn.build_binns import BINN
from modules.loaders.format_data import format_data

# Get GPU
device = torch.device('cuda')

# Load in data
path = '/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/'
file_name = 'spikes_data.npz'

# input: [x, t], output: [u, v]
inputs, output_u, output_v, shape_u, shape_v = format_data(path+file_name)
outputs = np.concatenate((output_u, output_v), axis=1)

def to_torch(ndarray):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

# split into train/val and convert to torch
N = len(inputs)
split = int(0.8*N)
p = np.random.permutation(N)
x_train = to_torch(inputs[p[:split]])
y_train = to_torch(outputs[p[:split]])
x_val = to_torch(inputs[p[split:]])
y_val = to_torch(outputs[p[split:]])
inputs = to_torch(inputs)
outputs = to_torch(outputs)

# initialize model
binn = BINN()
binn.to(device)

# compile 
parameters = binn.parameters()
opt = torch.optim.Adam(parameters, lr=1e-3)
model = model_wrapper(
    model=binn,
    optimizer=opt,
    loss=binn.loss,
    augmentation=None,
    save_name=f'{sys.argv[1]}')

epochs = int(1e6)
batch_size = 5000 # 25% of split
rel_save_thresh = 0.05

# train jointly
model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=None,
    verbose=1,
    validation_data=[x_val, y_val],
    early_stopping=5000,
    rel_save_thresh=rel_save_thresh)