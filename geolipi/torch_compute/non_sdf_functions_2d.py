import numpy as np
import torch as th
from .constants import EPSILON, ACOS_EPSILON


### Functions useful for 2D pattern creation.
# Function Documentation TBD


def nonsdf2d_tile_uv(points: th.Tensor, tile: th.Tensor,
                     height: int = None, width: int = None,
                     mode: str = "bicubic") -> th.Tensor:
    """
    Sample a 2D tile texture at given UV coordinates using grid_sample.

    Parameters
    ----------
    points:
        Input coordinates. Supported shapes:
        - [num_points, 2]
        - [batch, num_points, 2]
    tile:
        Tile image tensor. Supported shapes:
        - [H, W, C]
        - [batch, H, W, C]
    height, width:
        Optional explicit output height/width. When not provided, they are
        inferred from the number of points as sqrt(num_points).
    mode:
        Interpolation mode for grid_sample (e.g. "bilinear", "bicubic").

    Returns
    -------
    Tensor
        Sampled tile values with shape:
        - [num_points, C]   if input points were [num_points, 2]
        - [batch, num_points, C] otherwise.
    """
    # Normalize points shape to [B, N, 2]
    squeeze_batch = False
    if points.dim() == 2:
        points = points.unsqueeze(0)
        squeeze_batch = True
    assert points.dim() == 3 and points.size(-1) == 2, \
        f"points must be [B, N, 2] or [N, 2], got {points.shape}"

    # Normalize tile shape to [B, H, W, C]
    if tile.dim() == 3:
        tile = tile.unsqueeze(0)
    assert tile.dim() == 4, f"tile must be [H, W, C] or [B, H, W, C], got {tile.shape}"

    B, N, _ = points.shape

    # Infer height/width from number of points if not provided
    if height is None or width is None:
        side = int(np.sqrt(N))
        if side * side != N:
            raise ValueError(
                f"Cannot infer square height/width from {N} points; "
                f"please provide height and width explicitly."
            )
        height = side
        width = side

    # Reshape points to [B, H, W, 2] for grid_sample
    cur_points = points.reshape(B, height, width, 2)

    # Tile to [B, C, H, W]
    cur_tile = tile.permute(0, 3, 1, 2)

    # Sample
    output = th.nn.functional.grid_sample(cur_tile, cur_points,
                                          align_corners=True, mode=mode)
    # [B, C, H, W] -> [B, H, W, C] -> [B, N, C]
    output = output.permute(0, 2, 3, 1).reshape(B, N, -1)

    if squeeze_batch:
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

