
import plotly.graph_objects as go
import numpy as np
import torch as th
import matplotlib.pyplot as plt


def get_figure(sdf, input_type="SDF", res=128, point_count=2500, r=1):
    if isinstance(sdf, th.Tensor):
        sdf = sdf.detach().cpu().numpy()
    if len(sdf.shape) < 3:
        if len(sdf.shape) == 1:
            sdf = sdf.reshape(res, res, res)
        elif len(sdf.shape) == 2:
            sdf = sdf[:, 0].reshape(res, res, res)

    if input_type == "SDF":
        valid_points = np.stack(np.where(sdf <= 0), -1) / res
    else:
        # Its occupancy
        valid_points = np.stack(np.where(sdf == 1), -1) / res

    boundary_points = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    n_points = valid_points.shape[0]
    if n_points > point_count:
        # sample LIM points
        valid_points = valid_points[
            np.random.choice(n_points, point_count, replace=False)
        ]
    valid_points = np.concatenate([boundary_points, valid_points], 0)
    points = valid_points  # np.stack(np.where(sample == 1), -1)
    colors = (np.random.uniform(size=points.shape) * 255).astype(np.int8)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(size=r, color=colors)
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True)
            )
        )
    )
    return fig

