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
from modules.symbolic_net.individual import individual

config = {}
exec(Path(f'{sys.argv[1]}/config.cfg').read_text(encoding="utf8"), {}, config)

training_data_path = config['training_data_path']
species = int(config['species'])
degree = int(config['degree'])
nonzero_term_reg = int(config['nonzero_term_reg'])
l1_reg = float(config['l1_reg'])
param_bounds = float(config['param_bounds'])
density_weight = float(config['density_weight'])
learning_rate = float(config['learning_rate'])

dir_name = sys.argv[1]

# Set training hyperparameters
device = 'cpu'

# Generate training data
xt, u, v, shape_u, shape_v = format_data(training_data_path, plot=False)
u_triangle_mesh, v_triangle_mesh = lltriangle(u, v)
u, v = np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)

a, b, k = 1, 1, 0.01
F_true = reaction(u, v, a, b, k)

training_data_nans = np.stack((u, v, F_true), axis=1)

# Remove nans from training data
mask = ~np.isnan(training_data_nans).any(axis=1)
training_data = training_data_nans[mask]

# initialize model and compile
sym_net = symbolic_net(species, degree, param_bounds, device, l1_reg,
                       nonzero_term_reg)
sym_net.to(device)

### ANALYSIS

individuals = []

# Calculate AIC and BIC for all learned equations
parent_dir = '/'.join(dir_name.rstrip('/').split('/')[:-1])
print(dir_name, parent_dir)

child_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and 'binn_best_val_model' in os.listdir(os.path.join(parent_dir, d))]
print(child_dirs)

for child_dir in child_dirs:
    
    dir_path = f'{parent_dir}/{child_dir}'

    # load model    
    weights = torch.load(f"{dir_path}/binn_best_val_model", map_location=device)
    sym_net.load_state_dict(weights)

    ind = individual(sym_net.params, species, degree)
    
    ind.fix_insignificant_terms(torch.tensor(training_data))
    ind.fix_cheating_hill_functions(torch.tensor(training_data))
         
    true_vals = torch.from_numpy(training_data[:, -1])
    predicted_vals = ind.predict_f(torch.from_numpy(training_data[:, :-1]))
    
    aic, bic = ind.abic(true_vals, predicted_vals)
    
    individuals.append((child_dir, ind.params, aic, bic))

# Rank learned equations from least (best) to greatest (worst) AIC
individuals_sorted = sorted(individuals, key=lambda x: x[2])

# Write equations to file
fn = f'{parent_dir}/equations_ranked.txt'
file = open(fn, 'w')

for i in range(len(individuals_sorted)):
    # load model   
    ind = individual(individuals_sorted[i][1], species, degree)
    file.write(f'Directory: {individuals_sorted[i][0]}\n')
    file.write(f'AIC: {individuals_sorted[i][2]}\n')
    
    terms = ind.write_terms()
    for term in terms:
        file.write(f'{term}\n')
    file.write('\n')