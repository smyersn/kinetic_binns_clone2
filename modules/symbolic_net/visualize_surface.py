import sys, importlib, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.utils.numpy_torch_conversion import *
from modules.generate_data.simulate_system import reaction
from modules.loaders.format_data import format_data_general
    
def visualize_surface(model_dir, u_triangle_mesh, v_triangle_mesh, F_true,
                      F_mlp, filename=None):
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type':'scene'}, {'type':'scene'}]],
                        subplot_titles=('F(u, v)', 'F*(u, v)'),
                        horizontal_spacing = 0)

    fig.layout.annotations[0].update(y=0.8)
    fig.layout.annotations[1].update(y=0.8)
    fig.update_annotations(font_size=24, font_color='#000000')

    fig.add_trace(
        go.Surface(z=F_true, x=u_triangle_mesh, y=v_triangle_mesh,
                                    colorscale='mint',
                                    showscale=False,
                                    colorbar=dict(
                                        x=1.15,
                                        title='True',
                                        len=0.5)),
        row=1, col=1)
    fig.add_trace(
        go.Surface(z=F_mlp, x=u_triangle_mesh, y=v_triangle_mesh,
                                    colorscale='mint',
                                    showscale=False,
                                    colorbar=dict(
                                        title='MLP',
                                        len=0.5)),
        row=1, col=2)

    fig.update_layout(autosize=True,
        width=1500, 
        height=800,
        font=dict(color = '#000000',
                size=20),
        
        scene=dict( 
        xaxis_title='[u] (uM)',
        yaxis_title='[v] (uM)',
        zaxis_title='F',
        xaxis = dict(
            tick0 = 0,
            dtick = 2,
            tickfont = dict(size=18)),
        yaxis = dict(
            tick0 = 0.2,
            dtick = 0.4,
            tickfont = dict(size=18)),
        zaxis = dict(
            tick0 = 0,
            dtick = 2,
            tickfont = dict(size=18),
            range=[-1.5, 11]),

        camera=dict(eye=dict(x=1, y=-2.5, z=1)),),
                    
        scene2=dict(
        xaxis_title='[A] (uM)',
        yaxis_title='[B] (uM)',
        zaxis_title='F*',
        xaxis = dict(
            tick0 = 0,
            dtick = 2,
            tickfont = dict(size=18)),
        yaxis = dict(
            tick0 = 0.2,
            dtick = 0.4,
            tickfont = dict(size=18)),
        zaxis = dict(
            tick0 = 0,
            dtick = 2,
            tickfont = dict(size=18),
            range=[-1.5, 11]),

        camera=dict(eye=dict(x=1, y=-2.5, z=1))))

    fig.update_coloraxes(showscale=False)

    fig.show()
    
    if filename is not None:
        fig.write_image(f'{model_dir}/{filename}.png')
    else:
        fig.write_image(f'{model_dir}/f_mlp_surface.png')