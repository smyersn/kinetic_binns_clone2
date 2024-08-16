import sys
sys.path.append('../../')

import numpy as np
from modules.generate_data.simulate_system import simulate
from modules.loaders.visualize_training_data import plot_animation

species_totals = [1, 1.0246]
species_totals_high_u = [4, 1]
species_totals_high_v = [1, 4]
species_totals_high_equal = [4, 4.0246]
params = [1, 1, 0.01]

# simulate system with spikes initial conditions
#x_spikes, u_spikes, v_spikes, t_spikes = simulate(species_totals, params, 50, 500, 10, spikes=1)
#np.savez('../../data/spikes_data', x_spikes, u_spikes, v_spikes, t_spikes)

# simulate system with custom initial conditions
#x_custom, u_custom, v_custom, t_custom = simulate(species_totals, params, 50, 500, 10, custom=True)
#np.savez('../../data/custom_data', x_custom, u_custom, v_custom, t_custom)

# simulate system with random initial conditions
#x_random, u_random, v_random, t_random = simulate(species_totals, params, 50, 500, 10, random=True)
#np.savez('../../data/random_data', x_random, u_random, v_random, t_random)

#x_random, u_random, v_random, t_random = simulate(species_totals_high_u, params, 100, 500, 10, random=True)
#np.savez('../../data/high_u_random_data', x_random, u_random, v_random, t_random)
#plot_animation(x_random, u_random, v_random, t_random, save = True, name = 'high_u_random')

#x_random, u_random, v_random, t_random = simulate(species_totals_high_v, params, 100, 500, 10, random=True)
#np.savez('../../data/high_v_random_data', x_random, u_random, v_random, t_random)
#plot_animation(x_random, u_random, v_random, t_random, save = True, name = 'high_v_random')

x_random, u_random, v_random, t_random = simulate(species_totals_high_equal, params, 100, 500, 10, random=True)
np.savez('../../data/high_equal_random_data', x_random, u_random, v_random, t_random)
plot_animation(x_random, u_random, v_random, t_random, save = True, name = 'high_equal_random')