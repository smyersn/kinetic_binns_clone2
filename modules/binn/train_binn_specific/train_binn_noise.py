import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn.model_wrapper import model_wrapper
from modules.binn.build_binns import BINN
from modules.loaders.format_data import format_data
from modules.loaders.visualize_training_data import plot_animation, animate_training_data
from modules.generate_data.simulate_system import simulate

# load params from configuration file
config = {}
exec(Path(f'{sys.argv[1]}/config.cfg').read_text(encoding="utf8"), {}, config)

training_data_path = config['training_data_path']
epsilon = config['epsilon']
num_points = config['num_points']

# Get GPU
device = torch.device('cuda')

# input: [x, t], output: [u, v]
inputs, output_u, output_v, shape_u, shape_v = format_data(training_data_path)
training_data = np.column_stack((inputs, output_u, output_v))

def to_torch(ndarray):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

# define new points to interpolate to and array to save interpolated data
x_interp = np.linspace(0, 10, num_points)
training_data_interp = np.zeros((0,4))

# reduce points in training data
for t in np.unique(training_data[:, 1]):
    time_step = training_data[training_data[:, 1] == t]
    u_interp = np.interp(x_interp, time_step[:, 0], time_step[:, 2])
    v_interp = np.interp(x_interp, time_step[:, 0], time_step[:, 3])
    time_step_interp = np.column_stack((x_interp, np.repeat(t, len(x_interp)), u_interp, v_interp))
    training_data_interp = np.row_stack((training_data_interp, time_step_interp))

# shuffle data
N = len(training_data_interp)
p = np.random.permutation(N)
training_data_shuffled = training_data_interp[p[:]]

# add noise to outputs
noise = np.random.normal(0, epsilon, training_data_shuffled[:, 2:].shape)
training_data_noise = training_data_shuffled
training_data_noise[:, 2:] = training_data_noise[:, 2:] + noise

# split into train/val and convert to torch
M = len(training_data_noise)
split = int(0.8*M)
x_train = to_torch(training_data_noise[:split, :2])
y_train = to_torch(training_data_noise[:split, 2:])
x_val = to_torch(training_data_noise[split:, :2])
y_val = to_torch(training_data_noise[split:, 2:])
animate_training_data(training_data_noise[:, :2], training_data_noise[:, 2:], save=True, name=f'training_data')

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
    save_name=f'{sys.argv[1]}/model_weights_best_val_model')

epochs = int(1e6)
#batch_size = 128 if len(inputs) >= 128 else len(inputs) # 25% of split
#batch_size = 32
batch_size = 5000
rel_save_thresh = 0.05

print('made it', flush=True)
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

