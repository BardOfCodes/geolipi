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


########## POLYLINE SDF

def helper_polyline_sdf_arc(points, sc, ra):
    p = th.cat([th.abs(points[..., 0:1]), points[..., 1:2]], -1)
    # p[..., 0] = p[..., 0].abs()
    if sc[0, 0] < 0.0:
        d = -sc * ra
    else:
        d = sc * ra
        
    dist_1 = th.norm(p - d, dim=-1) 
    dist_2 = th.abs(th.norm(p, dim=-1) - ra)
    sdf = th.where(sc[0, 1] * p[..., 0] > sc[0, 0] * p[..., 1], dist_1, dist_2)
    return sdf

def helper_polyline_sc_from_ht(tan_half_x):
    tan_half_sq = tan_half_x * tan_half_x
    denom = 1.0 + tan_half_sq
    # sc = th.stack([1.0 - tan_half_sq, 2.0 * tan_half_x], -1) / denom
    sc = th.stack([(2.0 * tan_half_x), (1.0 - tan_half_sq)], -1) / denom
    return sc

def helper_polyline_bulge_arc(points, a, b, bulge):
    """
    points: (N, 2)
    a: (1, 2)
    b: (1, 2)
    bulge: (1,)
    """
    ba = b - a 
    l = th.norm(ba, dim=-1)
    ortho_ba = th.stack([ba[..., 1], -ba[..., 0]], -1)
    if l < EPSILON:
        l = l + EPSILON
    
    # Compute radius and center
    tan_half_theta = bulge
    sin_theta = 2.0 * tan_half_theta / (1.0 + tan_half_theta * tan_half_theta)
    cos_theta = (1.0 - tan_half_theta * tan_half_theta) / (1.0 + tan_half_theta * tan_half_theta)
    
    radius = l / (2.0 * th.abs(sin_theta))
    center = (a + b) / 2.0 + th.sign(bulge) * radius * cos_theta * ortho_ba / l
    
    # Compute angles
    start_angle = th.atan2((a - center)[..., 1], (a - center)[..., 0])
    end_angle = th.atan2((b - center)[..., 1], (b - center)[..., 0])
    
    # Compute SDF
    dist_to_center = th.norm(points - center, dim=-1)
    arc_sdf = th.abs(dist_to_center - radius)
    
    return arc_sdf

def helper_polyline_line_parallel(points, vertices_a, vertices_b):
    """
    Compute line SDF in parallel for all line segments
    points: (N, 2)
    vertices_a: (K, 2) - start points of line segments
    vertices_b: (K, 2) - end points of line segments
    
    Returns: (N, K, 2) where last dim is [distance, sign]
    """
    N = points.shape[0]
    K = vertices_a.shape[0]
    
    # Expand dimensions for broadcasting
    p = points.unsqueeze(1)  # (N, 1, 2)
    a = vertices_a.unsqueeze(0)  # (1, K, 2)
    b = vertices_b.unsqueeze(0)  # (1, K, 2)
    
    # Vector from a to b
    ba = b - a  # (1, K, 2)
    # Vector from a to point
    pa = p - a  # (N, K, 2)
    
    # Project pa onto ba
    ba_dot_ba = th.sum(ba * ba, dim=-1, keepdim=True)  # (1, K, 1)
    ba_dot_ba = th.clamp(ba_dot_ba, min=EPSILON)  # Avoid division by zero
    
    pa_dot_ba = th.sum(pa * ba, dim=-1, keepdim=True)  # (N, K, 1)
    t = th.clamp(pa_dot_ba / ba_dot_ba, 0.0, 1.0)  # (N, K, 1)
    
    # Closest point on line segment
    closest = a + t * ba  # (N, K, 2)
    
    # Distance to closest point
    diff = p - closest  # (N, K, 2)
    distance = th.norm(diff, dim=-1)  # (N, K)
    
    # Sign computation (simplified - assuming clockwise winding)
    cross = pa[..., 0] * ba[..., 1] - pa[..., 1] * ba[..., 0]  # (N, K)
    sign = th.sign(cross)  # (N, K)
    
    return th.stack([distance, sign], dim=-1)  # (N, K, 2)

def nonsdf2d_polyline_smooth(points, vertices, smoothness):
    """
    Computes the SDF for a polyline in 2D.
    vertices is a list of (x, y, bulge) format of size (k, 3).
    points are (x, y) points on which SDF is to be computed, of size (N, 2).
    """
    # Ideally, gather all ther vertices where bulge = 0
    # and similarly compute all ther vertices where bulge != 0
    # then in parallel for all the points compute both the sign and distance
    # Final distance will be min of the distances, and final sign will be multiplication of all the signs. 
    d = points[..., 0] * 0.0 + 100.0
    vertices_a = vertices[:, :2]
    # roll
    vertices_b = th.roll(vertices_a, -1, 0)
    # Do the parallel computation. 
    ds = helper_polyline_line_parallel(points, vertices_a, vertices_b)
    # Ds should be N, k, 2
    distances = ds[..., 0]
    signs = ds[..., 1]
    # TO make it differentiable -> convert it into a temperature guided summation.
    d = th.amin(distances, 1)
    s = th.prod(signs, 1)
    
    return s * d

def nonsdf2d_polyline(points, vertices):
    """
    Computes the SDF for a polyline in 2D.
    vertices is a list of (x, y, bulge) format of size (k, 3).
    points are (x, y) points on which SDF is to be computed, of size (N, 2).
    """
    d = points[..., 0] * 0.0 + 100.0
    s = points[..., 0] * 0.0 + 1.0
    for i in range(vertices.shape[0]):
        a = vertices[i:i+1, :2]
        b = vertices[(i+1) % vertices.shape[0]:((i+1) % vertices.shape[0])+1, :2]
        bulge = vertices[i, 2]
        
        if th.abs(bulge) < EPSILON:
            # Line segment
            ba = b - a
            pa = points - a
            h = th.clamp(th.sum(pa * ba, dim=-1, keepdim=True) / th.sum(ba * ba, dim=-1, keepdim=True), 0.0, 1.0)
            seg_d = th.norm(pa - h * ba, dim=-1)
            # Sign computation
            cross = pa[..., 0] * ba[..., 1] - pa[..., 1] * ba[..., 0]
            seg_s = th.sign(cross)
        else:
            # Arc segment
            seg_d = helper_polyline_bulge_arc(points, a, b, bulge)
            seg_s = th.ones_like(seg_d)  # Simplified sign for arcs
        
        d = th.minimum(d, seg_d)
        s = s * seg_s
    
    return s * d
