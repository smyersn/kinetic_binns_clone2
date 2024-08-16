import sys, importlib, os
sys.path.append('../../')
from matplotlib.cm import ScalarMappable 

from modules.utils.imports import *
from modules.binn.model_wrapper import model_wrapper
from modules.binn.build_binns import BINN
from modules.utils.numpy_torch_conversion import *
from modules.loaders.format_data import format_data
from modules.loaders.visualize_training_data import plot_animation
from modules.generate_data.simulate_system import laplace_2d
import time

def simulate_tl_surfaces(model, device, model_dir, model_names):
    """Simulate true reaction surface and surfaces learned w/ transfer learning

    Args:
        model (object): model for predicting F from u and v
        device (string): where to run operations (cpu, gpu, etc.)
        model_dir (string): path to model weights
        model_names (list of strings): list of model names (used for naming
            model weights and other output files)
    """
    # make directory for saving results
    simulation_dir = model_dir + '/simulations'
    os.makedirs(simulation_dir, exist_ok=True)
    
    # Load in true data
    xt, u, v, shape_u, shape_v = format_data(f'/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/spikes_data.npz', plot=False)
    outputs = np.concatenate((u, v), axis=1)

    # Define parameters for simulations
    t_max = np.max(xt[:, 1])
    dt = 0.0001
    steps = int(t_max / dt)
    L = np.max(xt[:, 0])
    n = 500
    rows = int(steps / 5000) + 1
    x_array = np.linspace(0, L, n)
    dx = L / n
    
    # Create array for saving simulation results, save simulation results
    kym_all = np.zeros((n, rows, len(model_names)+1))
    kym_true = np.reshape(u, (int(2 * t_max + 1), n), order='F')
    kym_all[:, :, 0] = kym_true.T

    for i in range(len(model_names)):
        model.load(f"{model_dir}/{model_names[i]}_best_val_model", device='cpu')

        # collect initial conditions from loaded data
        io = np.concatenate((xt, outputs), axis=1)

        u_grid = np.zeros((n))
        v_grid = np.zeros((n))

        idx = 0
        for row in io:
            if row[1] == 0: # time = 0
                u_grid[idx] = row[2]
                v_grid[idx] = row[3]
                idx += 1
                
        # Set up storage (only records every 5000th step, needs +1 for t = 0)
        u_array = np.zeros((rows, n))
        v_array = np.zeros((rows, n))
        t_array = np.zeros(rows)

        # Make first row of storage initial condition
        u_array[0] = u_grid
        v_array[0] = v_grid

        # Solve
        for t in range(1, steps+1):
            # Calculate Laplacian
            Lu = laplace_2d(u_grid, dx)
            Lv = laplace_2d(v_grid, dx)
            # Calculate reaction
            F = to_numpy(model.model.reaction(to_torch(np.column_stack((u_grid, v_grid)))[:, None]))

            # Update
            diff_u = (Lu * 0.01 + F[:, 0, 0]) * dt
            diff_v = (Lv * 1 - F[:, 0, 0]) * dt
                
            u_grid += diff_u
            v_grid += diff_v
            
            # Print progress
            if (t / steps) * 100 % 10 == 0:
                print(f'{(t / steps) * 100}% complete')
            
            # Record every 5000th step
            if t % 5000 == 0:
                u_array[int(t / 5000)] = u_grid
                v_array[int(t / 5000)] = v_grid
                t_array[int(t / 5000)] = t * dt
            
            # Check for steady state (current minus previous frame below tolerance)    
            if t > 0 and t % 5000 == 0:
                if np.max(np.abs(u_array[int(t / 5000)] - u_array[int(t / 5000)-1])) < 0.0025:
                    print(f'Steady state at t = {t * dt}')
                    break
                
        # Create and save animation
        plot_animation(x_array, u_array, v_array, t_array, save=True, name=f"{simulation_dir}/{model_names[i]}_animation")

        # Make sure u and v don't go negative
        print(f'Minimum u-value during simulation: {np.min(u_array)}')
        print(f'Minimum v-value during simulation: {np.min(v_array)}')

        # Reshape data for kymograph, instantiate figure
        kym_all[:, :, i+1] = u_array.T

    # Create a figure with subplots for ground truth, training data, color bar
    num_axes = len(model_names)+1
    fig, axs = plt.subplots(1, num_axes, figsize=(25, 5))

    # Plot arrays using ax.imshow
    for i in range(num_axes):
        # Get data for single simulations and remove columns after reaching s.s.
        data = kym_all[:, :, i]
        zero_columns = np.all(data == 0, axis=0)
        data_stripped = np.delete(data, np.where(zero_columns), axis=1)
        # Define limits for axes
        t = (data_stripped.shape[-1] - 1) / 2
        d = 10
        # Plot
        axs[i].imshow(data_stripped, cmap='viridis', origin='upper', extent=[0,t,0,d])
        axs[i].set_xlabel('Time (s)')
        axs[i].set_aspect((axs[i].get_xlim()[1] - axs[i].get_xlim()[0]) / (axs[i].get_ylim()[1] - axs[i].get_ylim()[0]))

        if i == 0:
            axs[i].set_title(f"Ground Truth Sim.")
            axs[i].set_ylabel('Position')
            first_pos = axs[i].get_position()
        else:
            axs[i].set_title(f"{model_names[i-1]}")
            axs[i].set_yticklabels([])
        
    # Add a colorbar
    position = axs[-1].get_position()
    new_left = position.x1 + 0.1
    new_bottom = position.y0
    new_width = 0.01
    new_height = position.height

    cax = fig.add_axes([new_left, new_bottom, new_width, new_height])
    mappable = cm.ScalarMappable(cmap='viridis')
    mappable.set_clim(vmin=np.min(kym_all), vmax=np.max(kym_all))
    plt.colorbar(mappable, ax=axs.ravel().tolist(), shrink=0.6, cax=cax)

    # Show the plot
    plt.tight_layout()
    plt.savefig(simulation_dir + "/kymograph.png", bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    a