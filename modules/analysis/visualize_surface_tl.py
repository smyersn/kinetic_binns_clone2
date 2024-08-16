import sys, importlib, os
sys.path.append('../../')

from modules.utils.imports import *
from modules.utils.numpy_torch_conversion import *
from modules.binn.build_binns import BINN
from modules.binn.model_wrapper import model_wrapper
from modules.loaders.format_data import format_data
from modules.generate_data.simulate_system import reaction

def visualize_tl_surfaces(model, device, model_dir, model_names): 
    """Plot true reaction surface and surfaces learned w/ transfer learning

    Args:
        model (object): model for predicting F from u and v
        device (string): where to run operations (cpu, gpu, etc.)
        model_dir (string): path to model weights
        model_names (list of strings): list of model names (used for naming
            model weights and other output files)
        
    """
    # Load and format training data
    xt, u, v, shape_u, shape_v = format_data(f'/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/spikes_data.npz', plot=False)
    u_triangle_mesh, v_triangle_mesh = lltriangle(u, v)

    # Calculate true reaction surface
    true_params = [1, 1, 0.01]
    F_true = reaction(u_triangle_mesh, v_triangle_mesh, true_params)[0]

    F_data = np.zeros((101, 101, len(model_names)+1))
    F_data[:, :, 0] = F_true
        
    for i in range(len(model_names)):
        model.load(f"{model_dir}/{model_names[i]}_best_val_model", device=device)
        
        # Calculate MLP reaction surface
        uv = np.column_stack((np.ravel(u_triangle_mesh), np.ravel(v_triangle_mesh)))
        F_mlp_stacked = to_numpy(model.model.reaction(to_torch(uv)[:, None]))
        F_data[:, :, i + 1] = np.reshape(F_mlp_stacked, (101, 101))
        
    plot_num = F_data.shape[-1]

    # Plot
    fig = make_subplots(rows=1, cols=plot_num,
                        specs=[[{'type': 'scene'}] * plot_num],
                        subplot_titles=('F(A, B)',) +  tuple(model_names))


    fig.update_annotations(font_size=18, font_color='#000000', y=1)

    for i in range(plot_num):
        fig.add_trace(
            go.Surface(z=F_data[:, :, i], x=u_triangle_mesh, y=v_triangle_mesh,
                                    colorscale='mint',
                                    showscale=False,
                                    colorbar=dict(
                                        x=1.15,
                                        title='True',
                                        len=0.5)),
            row=1, col=i+1)

    common_scene=dict( 
        xaxis_title='[A] (uM)',
        yaxis_title='[B] (uM)',
        zaxis_title='F',
        xaxis = dict(
            tick0 = 0,
            dtick = 2,
            tickfont = dict(size=12)),
        yaxis = dict(
            tick0 = 0.2,
            dtick = 0.4,
            tickfont = dict(size=12)),
        zaxis = dict(
            tick0 = 0,
            dtick = 2,
            tickfont = dict(size=12),
            range=[-1.5, 11]),
        camera=dict(eye=dict(x=1, y=-2.5, z=1)))

    for i in range(plot_num):
        fig.update_layout(autosize=True,
            width=350*plot_num, 
            height=500,
            font=dict(color = '#000000',
                    size=12),
            
        **{f'scene{"" if i == 0 else str(i + 1)}': common_scene})

    fig.update_coloraxes(showscale=False)

    fig.show()
    fig.write_image(model_dir + '/f_mlp_surface.png')