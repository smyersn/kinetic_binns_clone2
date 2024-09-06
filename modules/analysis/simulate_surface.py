import sys, importlib, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.binn.model_wrapper import model_wrapper
from modules.binn.build_binns_2d_diffusion import BINN
from modules.utils.numpy_torch_conversion import *
from modules.loaders.format_data import format_data_general
from modules.loaders.visualize_training_data import animate_data
from modules.generate_data.simulate_system import generate_initial_conditions, simulate
import time

def simulate_surface(model, device, dimensions, species, model_dir, training_data_path):
    # Load in data
    training_data = format_data_general(training_data_path, dimensions, species)
    xt = training_data[:, :dimensions+1]
    outputs = training_data[:, dimensions+1:]
    
    # Define parameters
    t_max = np.max(xt[:, 1])
    dt = 0.0001
    steps = int(t_max / dt)
    L = np.max(xt[:, 0])
    n = 500
    x_array = np.linspace(0, L, n)
    dx = L / n

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
    rows = int(steps / 5000) + 1
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
            
    # Remove zeros from arrays due to reaching steady state
    u_array = u_array[~np.all(u_array == 0, axis=1)]
    v_array = v_array[~np.all(v_array == 0, axis=1)]
    t_array = np.trim_zeros(t_array, 'b')
    
    formatted_data = format_data_general(dimensions, species, x_array=x_array, 
                        t_array=t_array, u_array=u_array, v_array=v_array)

    # Create and save animation
    plot_animation(formatted_data, dimensions, species, f'{model_dir}/f_mlp_animation')

    # Make sure u and v don't go negative
    print(f'Minimum u-value during simulation: {np.min(u_array)}')
    print(f'Minimum v-value during simulation: {np.min(v_array)}')

    # Create kymograph

    # Reshape data for kymograph, instantiate figure
    u_kymograph = np.reshape(outputs[:, 0], (int(2 * t_max + 1), n), order='F')

    fig = plt.figure(figsize=(5, 5), facecolor='w')
    fig.subplots_adjust(wspace=0.2)

    # Create axes
    ax1 = fig.add_axes([0.1, 0.2, 0.3, 0.3])
    ax2 = fig.add_axes([0.55, 0.2, 0.3, 0.3])
    # Create extra axis for colorbar
    ax3 = fig.add_axes([0.9, 0.2, 0.2, 0.3])
    ax3.axis('off')

    # Plot Data
    kymograph_sim = ax1.imshow(u_kymograph.T, aspect='auto', cmap='viridis', extent=[0,24.5,0,10]) # simulated
    kymograph_learned = ax2.imshow(u_array.T, aspect='auto', cmap='viridis', extent=[0,24.5,0,10]) # learned

    # Show colorbar
    cbar = fig.colorbar(kymograph_sim, ax=ax3, location='left', ticklocation='bottom')
    plt.text(0, 0.33, '[A] (uM)', rotation=270)

    # Format plots
    ax2.set_yticklabels([])
    ax1.set_ylabel('Space (uM)')
    ax1.set_xlabel('Time (s)')
    ax2.set_ylabel('Space (uM)')
    ax2.set_xlabel('Time (s)')
    ax1.title.set_text('Solution w/ F(A, B)')
    ax2.title.set_text('Solution w/ F*(A, B)')

    # Show the plot
    plt.savefig(f'{model_dir}/f_mlp_kymograph.png')
    plt.show()
    
def simulate_surface_general(model, device, dimensions, species, diffusion, model_dir, training_data_path):
    # Load in data
    training_data = format_data_general(dimensions, species, training_data_path)
    xt = training_data[:, :dimensions+1]
    outputs = training_data[:, dimensions+1:]
    
    # Get initial conditions from training data 
    L = np.max(xt[:, 0])
    T = np.max(xt[:, dimensions])

    points = len(np.unique(training_data[:, 0]))
    ic = training_data[training_data[:, dimensions] == 0]

    u0 = np.reshape(ic[:, dimensions+1], (points,)*dimensions)
    v0 = np.reshape(ic[:, dimensions+2], (points,)*dimensions)
    
    # Load model
    model.load(f'{model_dir}/binn_best_val_model', device=device)

    # Simulate surface from training data initial conditions and animate
    x_array, u_array, v_array, t_array = simulate(u0, v0, L, points, T, 
                                                  dimensions, species,
                                                  nn=model.model, 
                                                  diffusion=diffusion,
                                                  early_stop=False)
    
    sim_formatted = format_data_general(dimensions, species, x_array=x_array, 
                        t_array=t_array, u_array=u_array, v_array=v_array)
    
    animate_data(sim_formatted, dimensions, species, name=f'{model_dir}/f_mlp_animation_training_data_ic')

    # Make sure u and v don't go negative
    print(f'Minimum u-value during simulation: {np.min(u_array)}')
    print(f'Minimum v-value during simulation: {np.min(v_array)}')
    
    # Generate random initial conditions
    u0, v0 = generate_initial_conditions(1, 1.0246, points, dimensions, random=True)
    
    # Simulate surface from random initial conditions and animate
    x_array, u_array, v_array, t_array = simulate(u0, v0, L, points, T,
                                                  dimensions, species,
                                                  nn=model.model,
                                                  diffusion=diffusion,
                                                  early_stop=False)
    
    sim_formatted = format_data_general(dimensions, species, x_array=x_array, 
                    t_array=t_array, u_array=u_array, v_array=v_array)

    animate_data(sim_formatted, dimensions, species, name=f'{model_dir}/f_mlp_animation_random_ic')

    # Make sure u and v don't go negative
    print(f'Minimum u-value during simulation: {np.min(u_array)}')
    print(f'Minimum v-value during simulation: {np.min(v_array)}')

    # Create kymograph
    if dimensions == 1:
        # Reshape data for kymograph, instantiate figure
        u_kymograph = np.reshape(outputs[:, 0], (int(2 * T + 1), points), order='F')

        fig = plt.figure(figsize=(5, 5), facecolor='w')
        fig.subplots_adjust(wspace=0.2)

        # Create axes
        ax1 = fig.add_axes([0.1, 0.2, 0.3, 0.3])
        ax2 = fig.add_axes([0.55, 0.2, 0.3, 0.3])
        # Create extra axis for colorbar
        ax3 = fig.add_axes([0.9, 0.2, 0.2, 0.3])
        ax3.axis('off')

        # Plot Data
        kymograph_sim = ax1.imshow(u_kymograph.T, aspect='auto', cmap='viridis', extent=[0,24.5,0,10]) # simulated
        kymograph_learned = ax2.imshow(u_array.T, aspect='auto', cmap='viridis', extent=[0,24.5,0,10]) # learned

        # Show colorbar
        cbar = fig.colorbar(kymograph_sim, ax=ax3, location='left', ticklocation='bottom')
        plt.text(0, 0.33, '[A] (uM)', rotation=270)

        # Format plots
        ax2.set_yticklabels([])
        ax1.set_ylabel('Space (uM)')
        ax1.set_xlabel('Time (s)')
        ax2.set_ylabel('Space (uM)')
        ax2.set_xlabel('Time (s)')
        ax1.title.set_text('Solution w/ F(A, B)')
        ax2.title.set_text('Solution w/ F*(A, B)')

        # Show the plot
        plt.savefig(f'{model_dir}/f_mlp_kymograph.png')
        plt.show()
