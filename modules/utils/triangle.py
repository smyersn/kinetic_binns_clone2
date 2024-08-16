import numpy as np

# create triangular mesh grid from training data
def lltriangle(u, v):
    # generate points bewteen min and max values to create mesh
    #u = np.reshape(u, (50, 500), order='F')
    #v = np.reshape(v, (50, 500), order='F')

    u_points = np.linspace(np.min(u), np.max(u), 101)
    v_points = np.linspace(np.min(v), np.max(v), 101)
    u_mesh, v_mesh = np.meshgrid(u_points, v_points)

    # get lower left corner
    u_rot = np.rot90(u_mesh)
    u_rot[np.triu_indices(u_rot.shape[0], 0)] = np.nan
    u_triangle = np.rot90(u_rot, 3)
    
    v_rot = np.rot90(v_mesh)
    v_rot[np.triu_indices(v_rot.shape[0], 0)] = np.nan
    v_triangle = np.rot90(v_rot, 3)

    return u_triangle, v_triangle