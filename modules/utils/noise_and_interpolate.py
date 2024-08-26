import numpy as np
import itertools
from scipy.interpolate import RegularGridInterpolator

def noise_and_interpolate(training_data, num_points, epsilon, dimensions, species):
    
    # get times and points from training data
    times = np.unique(training_data[:, dimensions])
    points = np.unique(training_data[:, 0])
    
    if num_points == 0 or num_points > len(points):
        num_points = len(points)
    
    # define new points for interpolation
    x_interp = np.linspace(np.min(points), np.max(points), num_points)
    x_interp_pairs = np.array(list(itertools.product(x_interp, repeat=dimensions)))

    # create array to store all interpolated data
    training_data_interp = np.zeros((0, dimensions + species + 1))
   
    # iterate over frames in training data
    for t in np.unique(times):
        frame = training_data[training_data[:, dimensions] == t]
        
        # create array to store interpolated data for each frame
        training_data_temp = np.hstack((x_interp_pairs, 
                                        np.expand_dims(np.repeat(t, len(x_interp_pairs)), axis=1)))
                                       
        for specie in range(species):
            # reshape training data to meshgrid
            values = np.reshape(frame[:, dimensions + specie + 1], 
                                (len(points),) * dimensions)
            
            # define interpolator function and interpolate at new points
            interp_fn = RegularGridInterpolator((points,)*dimensions, values, method='linear')
            interpolated_values = interp_fn(x_interp_pairs)
            
            # append frame data to all data
            training_data_temp = np.hstack((training_data_temp,
                                           np.expand_dims(interpolated_values, axis=1)))

        training_data_interp = np.vstack((training_data_interp, training_data_temp))
            
            
    # add noise to outputs and set negative values to zero
    noise = np.random.normal(0, epsilon, training_data_interp[:, dimensions+1:].shape)
    training_data_noise = training_data_interp.copy()
    training_data_noise[:, dimensions+1:] = np.maximum(training_data_noise[:, dimensions+1:] + noise, 0)
    
    return training_data_noise