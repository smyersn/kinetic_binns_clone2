import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.symbolic_net.model_wrapper import model_wrapper
from modules.symbolic_net.build_symbolic_net import symbolic_net
from modules.utils.training_test_split import training_test_split
from modules.analysis.generate_loss_curves import generate_loss_curves
from modules.loaders.format_data import format_data
from modules.generate_data.simulate_system import reaction
from modules.symbolic_net.write_terms import write_terms
from modules.symbolic_net.visualize_surface import visualize_surface

config = {}
exec(Path(f'{sys.argv[1]}/config.cfg').read_text(encoding="utf8"), {}, config)

training_data_path = config['training_data_path']
species = int(config['species'])
degree = int(config['degree'])
nonzero_terms = int(config['nonzero_terms'])
l1reg = float(config['l1reg'])
coef_bounds = float(config['coef_bounds'])
density_weight = float(config['density_weight'])

dir_name = sys.argv[1]

# Set training hyperparameters
epochs = int(1e6)
rel_save_thresh = 0.05
device = 'cuda'

# Generate training data
xt, u, v, shape_u, shape_v = format_data(training_data_path, plot=False)
u_triangle_mesh, v_triangle_mesh = lltriangle(u, v)
u, v = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)

a, b ,k = 1, 1, 0.01
F_true = reaction(u, v, a, b, k)

training_data_nans = np.stack((u, v, F_true), axis=1)

# Remove nans from training data
mask = ~np.isnan(training_data_nans).any(axis=1)
training_data = training_data_nans[mask]

# Split training data
x_train, y_train, x_val, y_val = training_test_split(training_data, 1, device)

# initialize model and compile
sym_net = symbolic_net(species, degree, coef_bounds, device, l1reg,
                       nonzero_terms)
sym_net.to(device)

# initialize optimizer
parameters = sym_net.parameters()
opt = torch.optim.Adam(parameters, lr=0.001)

model = model_wrapper(
    model=sym_net,
    optimizer=opt,
    loss=sym_net.loss,
    augmentation=None,
    save_name=f'{dir_name}/binn')

# generate histogram for trainind data density if specified in config file
hist = None
edges = None

if density_weight != 0:
    u = y_train[:, 0].flatten()
    v = y_train[:, 1].flatten()

    hist_uncorrected = torchist.normalize(torchist.histogramdd(y_train, bins=10, 
                                low=[min(u).item(), min(v).item()], 
                                upp=[max(u).item(), max(v).item()]))[0]

    # change zero values in histogram so reciprocal density is not inf
    hist = torch.where(hist_uncorrected == 0, 0.001, hist_uncorrected)

    edges = torchist.histogramdd_edges(y_train, bins=10, 
                                    low=[min(u).item(), min(v).item()], 
                                    upp=[max(u).item(), max(v).item()])
    edges[0] = edges[0].to(device)
    edges[1] = edges[1].to(device)

# train jointly
train_loss_dict, val_loss_dict = model.fit(
    x=x_train,
    y=y_train,
    batch_size=int(0.1*len(training_data)),
    epochs=epochs,
    callbacks=None,
    verbose=1,
    validation_data=[x_val, y_val],
    early_stopping=10000,
    rel_save_thresh=rel_save_thresh,
    density_weight=density_weight,
    hist=hist,
    edges=edges)

generate_loss_curves(train_loss_dict, val_loss_dict, dir_name)

# Load model for analaysis
model.load(f"{dir_name}/binn_best_val_model", device=device)

# Write terms
fn = f'{dir_name}/equation.txt'
file = open(fn, 'w')
terms = write_terms(model.model.individual)

for term in terms:
    file.write(f'{term}\n')

file.close()

# Generate surface
visualize_surface(model, dir_name, training_data_path)