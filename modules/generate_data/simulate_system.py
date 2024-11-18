import sys, importlib, os
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{file_dir}/../../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from modules.loaders.visualize_training_data import plot_animation
from modules.utils.numpy_torch_conversion import *

# Define 2-dimensional Laplacian
def laplace(M, dx, dim):
    if dim == 1:
        grid = (-2 * M + np.roll(M, 1) + np.roll(M, -1)) / dx**2
    
    elif dim == 2:
            grid = (-4 * M + np.roll(M, 1, axis=0) + np.roll(M, -1, axis=0) + 
                    np.roll(M, 1, axis=1) + np.roll(M, -1, axis=1)) / dx**2

    return grid

def calc_squared_wavenumbers(L, N, Du, Dv):
    kx = (2*np.pi/L) * 1j * np.hstack((np.arange(0, N//2), np.array([0]), np.arange(-N//2+1, 0)))
    ky = kx.copy()

    k2x = kx**2
    k2y = ky**2

    kxx, kyy = np.meshgrid(k2x, k2y)

    ksqu = Du * (kxx + kyy)
    ksqv = Dv * (kxx + kyy)
    
    return ksqu, ksqv

# Define reaction
def wave_pinning_reaction(u, v, params):
    a, b, k = params
    
    F = (a * u**2 * v) / (1 + k * u**2) - b * u
    return F

def alex_reaction(u, v, params):
    a, b = params
    
    F = a * u**2 * v - b * u
    return F

# Define update functions for simulation
def update_laplace(reaction_fn, u0, v0, params, Du, Dv, dt, dx, points, dim, nn=None,
              diffusion=False):
    if nn:
        F = to_numpy(nn.reaction(to_torch(
            np.column_stack((u0, v0)))[:, None]))
        F = np.reshape(F, (points,)*dim)
    else:
        F = reaction_fn(u0, v0, params)


    Lu = laplace(u0, dx, dim)
    Lv = laplace(v0, dx, dim)
    
    u1 = u0 + (Du * Lu + F) * dt
    v1 = v0 + (Dv * Lv - F) * dt
    
    return u1, v1

def update_fourier(reaction_fn, u0, v0, params, Du, Dv, dt, ksqu, ksqv, points, nn=None,
              diffusion=False):
    if nn:
        F = to_numpy(nn.reaction(to_torch(
            np.column_stack((u0.ravel(), v0.ravel())))[:, None]))
        F = np.reshape(F, (points,)*2)

    else:
        F = reaction_fn(u0, v0, params)
        
    u1r = u0 + F * dt
    v1r = v0 - F * dt
    
    u1r_hat = np.fft.fft2(u1r)
    v1r_hat = np.fft.fft2(v1r)
    
    u1 = np.real(np.fft.ifft2(u1r_hat / (1 - dt * ksqu)))
    v1 = np.real(np.fft.ifft2(v1r_hat / (1 - dt * ksqv)))
    
    return u1, v1  

def generate_initial_conditions(u0, v0, N, dim, spikes=0, custom=False, 
                                random=False):
    if spikes != 0 and dim == 1:
        u = np.full(N, u0) - np.cos(2 * spikes * np.pi * np.arange(N) / N) * 0.1 
        v = np.full(N, v0)
        
    if custom and dim == 1:
        # Generate custom initial conditions from file 
        initial_data_path = '../../data/custom_initial_state.csv'
        initial_data = np.loadtxt(initial_data_path, delimiter=',')

        # Get coordinates to interpolate from 
        xp = np.linspace(0, 10, 500)
        up = initial_data[0]
        vp = initial_data[1]

        # Interpolate
        xi = np.linspace(0, 10, N)
        u = np.interp(xi, xp, up)
        v = np.interp(xi, xp, vp)
        
    if random and dim == 1: 
        # Generate initial conditions with random noise
        u = u0 * (np.random.rand(N) * 2)
        v = v0 * np.ones(N)
        
    if dim > 1:
        u = (np.random.rand(*(N,) * dim) + 0.5) * u0
        v = np.ones((N,) * dim) * v0

    return u, v
    
def simulate(u0, v0, L, N, T, dim, params, reaction_fn, Du=0.01, Dv=1, nn=None, 
             diffusion=False, npz_path=None, early_stop=True):
    
    if diffusion == True:
        D = nn.diffusion_fitter()
        Du = D[0].detach().numpy()
        Dv = D[1].detach().numpy()

    # Define system parameters
    if dim == 1:
        dt = 0.0001
        ss_tolerance = 0.005
        dx = L / N
    else:
        dt = 0.0001
        ss_tolerance = 0.05
        ksqu, ksqv = calc_squared_wavenumbers(L, N, Du, Dv)
        dx = L / N
                
    nits = int(T / dt)
    half_sec_nits = 0.5 / dt

    x_array = np.linspace(0, L, N)
                   
    # Set up storage (only records at half second intervals)
    rows = int(nits / half_sec_nits) + 1

    u_array = np.zeros(((rows,) + (N,)*dim))
    v_array = np.zeros(((rows,) + (N,)*dim))
    t_array = np.zeros(rows)
    
    # Set initial conditions
    u, v = u0, v0
    
    # Solve
    for t in range(nits):
        if t % (nits // 10) == 0:
            print(f"Progress: {int((t / nits) * 100)}%", flush=True)
        if t % half_sec_nits == 0:
            u_array[int(t/half_sec_nits), Ellipsis] = u
            v_array[int(t/half_sec_nits), Ellipsis] = v
            t_array[int(t / half_sec_nits)] = t * dt
            
            if early_stop == True:        
                if t > 0 and np.max(np.abs(u_array[int(t/half_sec_nits)] - u_array[int(t/half_sec_nits)-1])) < ss_tolerance:
                    print(f'Steady state at t = {t * dt}', flush=True)
                    break
        
        u, v = update_laplace(reaction_fn, u, v, params, Du, Dv, dt, dx, N, dim, nn, diffusion)

    # Remove zeros from arrays due to reaching steady state
    u_array = u_array[~np.all(u_array == 0, axis=1)]
    v_array = v_array[~np.all(v_array == 0, axis=1)]
    t_array = np.trim_zeros(t_array, 'b')    
    
    if npz_path is not None:
        np.savez(npz_path, x_array, u_array, v_array, t_array)
    else:
        return x_array, u_array, v_array, t_array
    
if __name__ == '__main__':
    species_totals = [1, 1.0246]
    species_totals_high_u = [4, 1]
    species_totals_high_v = [1, 4]
    species_totals_high_equal = [4, 4.0246]
    params = [1, 1, 0.01]
    
    x, u, v, t = simulate(species_totals_high_equal, params, 100, 500, 10, random=True)
    np.savez('../../data/high_equal_random_data', x_random, u_random, v_random, t_random)
    plot_animation(x, u, v, t, save = True, name = 'high_equal_random')
