import numpy as np
import torch as th
from .constants import EPSILON, ACOS_EPSILON


### Functions useful for 2D pattern creation.
# Function Documentation TBD

def nonsdf2d_tile_uv(points: th.Tensor, tile: th.Tensor, height=None, width=None, mode="bicubic") -> th.Tensor:
    """
    Parameters:
        points: Input coordinates, shape [batch, num_points, 2]
        tile: Tile to splat, shape [batch, height, width, n_channel]
        height: Optional height for reshaping
        width: Optional width for reshaping
        mode: Interpolation mode

    Returns:
        Tensor: Tiled output
    """
    # using torch.nn.functional.grid_sample
    # rearrange points in B,H,W,2 format
    squeeze_out = False
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
        squeeze_out = True
    if len(tile.shape) == 3:
        tile = tile.unsqueeze(0)
    ps = points.shape
    bs = ps[0]
    if height is None:
        height = np.sqrt(ps[1]).astype(int)
        width = height
    # Doesn't handle the case where one is given    
    cur_points = points.reshape(bs, height, width, 2)
    cur_tile = tile.permute(0, 3, 1, 2)
    output = th.nn.functional.grid_sample(cur_tile, cur_points, align_corners=True, mode=mode)
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(bs, ps[1], -1)
    if squeeze_out:
        output = output.squeeze(0)
    return output

def nonsdf2d_sin_x(points: th.Tensor, freq: th.Tensor, phase_shift: th.Tensor) -> th.Tensor:
    """
    Parameters:
        points: Input coordinates, shape [batch, num_points, 2]
        freq: Frequency parameter
        phase_shift: Phase shift parameter

    Returns:
        Tensor: Sine wave along x-axis
    """
    base_sdf = th.sin(2 * np.pi * freq * points[..., 0] + phase_shift)
    return base_sdf

def nonsdf2d_sin_y(points, freq, phase_shift):
    base_sdf = th.sin(2 * np.pi * freq * points[..., 1] + phase_shift)
    return base_sdf

def nonsdf2d_sin_diagonal(points, freq, phase_shift):
    base_sdf = th.sin(2 * np.pi * freq * (points[..., 1] + points[..., 0]) + phase_shift)
    return base_sdf

def nonsdf2d_sin_diagonal_flip(points, freq, phase_shift):
    base_sdf = th.sin(2 * np.pi * freq * (points[..., 1] - points[..., 0]) + phase_shift)
    return base_sdf

def nonsdf2d_sin_radial(points, freq, phase_shift):
    base_sdf = th.sin(2 * np.pi * freq * (th.norm(points, dim=-1)) + phase_shift)
    return base_sdf

def nonsdf2d_squiggle_lines_y(points, freq, phase_shift, shift_amount, freq_2, phase_shift_2):
    
    shift = shift_amount * th.sin(2 * np.pi * freq_2 * points[..., 0] + phase_shift_2)
    base_sdf = th.sin(2 * np.pi * freq * (points[..., 1] + shift) + phase_shift)
    return base_sdf


def nonsdf2d_squiggle_lines_x(points, freq, phase_shift, shift_amount, freq_2, phase_shift_2):
    
    shift = shift_amount * th.sin(2 * np.pi * freq_2 * points[..., 1] + phase_shift_2)
    base_sdf = th.sin(2 * np.pi * freq * (points[..., 0] + shift) + phase_shift)
    return base_sdf


def nonsdf2d_squiggle_diagonal(points, freq, phase_shift, shift_amount, freq_2, phase_shift_2):
    
    shift = shift_amount * th.sin(2 * np.pi * freq_2 * (points[..., 0] + points[..., 1]) + phase_shift_2)
    base_sdf = th.sin(2 * np.pi * freq * ((points[..., 0] - points[..., 1]) + shift) + phase_shift)
    return base_sdf


def nonsdf2d_squiggle_diagonal_flip(points, freq, phase_shift, shift_amount, freq_2, phase_shift_2):
    
    shift = shift_amount * th.sin(2 * np.pi * freq_2 * (points[..., 0] - points[..., 1]) + phase_shift_2)
    base_sdf = th.sin(2 * np.pi * freq * ((points[..., 0] + points[..., 1]) + shift) + phase_shift)
    return base_sdf

def nonsdf2d_squiggle_radial(points, freq, phase_shift, shift_amount, freq_2, phase_shift_2):
    theta = th.atan2(points[..., 1], points[..., 0])
    shift = shift_amount * th.sin(freq_2 * theta + phase_shift_2)
    base_sdf = th.sin(2 * np.pi * freq * (th.norm(points, dim=-1) + shift) + phase_shift)
    return base_sdf

def nonsdf2d_squiggle_radial_distortion(points, freq, phase_shift, shift_amount, freq_2, phase_shift_2):
    theta = th.atan2(points[..., 1:2], points[..., 0:1])
    shift = shift_amount * th.sin(freq_2 * theta + phase_shift_2)
    base_sdf = th.sin(2 * np.pi * freq * (th.norm(points + shift, dim=-1)) + phase_shift)
    return base_sdf

def nonsdf2d_sin_along_axis_y(points, freq, phase_shift, scale):
    
    target = (1 +  th.sin(2 * np.pi * freq * points[..., 1] + phase_shift)) * scale
    base_sdf = th.abs(points[..., 0]) - target 
    return base_sdf

def nonsdf2d_instantiated_prim(points, instance, height=None, width=None, mode="bicubic"):
    return instance

