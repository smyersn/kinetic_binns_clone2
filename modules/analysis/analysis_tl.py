import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn.model_wrapper import model_wrapper
from modules.binn.build_binns import BINN
from modules.loaders.format_data import format_data
from modules.analysis.visualize_surface_tl import visualize_tl_surfaces
from modules.analysis.simulate_surface_tl import simulate_tl_surfaces

def to_torch(ndarray):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

# load params from configuration file
config = {}
exec(Path(f'{sys.argv[1]}/config.cfg').read_text(encoding="utf8"), {}, config)

training_data_path = config['training_data_path']
epsilon = config['epsilon']
num_points = config['num_points']
rounds_of_training = config['rounds_of_training']
dir_name = sys.argv[1]

# initialize model
device = torch.device('cpu')
binn = BINN(uv_layers=[128, 128, 128, 128, 128, 128, 128, 128, 2], f_layers=[32, 32, 32, 1])
binn.to(device)

# define training data
model_names = [f'model_{i+1}' for i in range(rounds_of_training)]

opt = torch.optim.Adam(list(binn.parameters()), lr=1e-3)
model = model_wrapper(
    model=binn,
    optimizer=opt,
    loss=binn.loss,
    augmentation=None)

# Analyze results
visualize_tl_surfaces(model, device, dir_name, model_names)
simulate_tl_surfaces(model, device, dir_name, model_names)