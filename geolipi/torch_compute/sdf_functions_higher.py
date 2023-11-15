
import torch as th
import torch as th
import numpy as np
from .common import EPSILON

# The return of a curve distance should be:
# 1) The parameter t on the curve (0, 1) which corresponds to the closest point
# 3) The projection of the point into the plane with normal plane_normal

def sdf3d_linear_extrude(points, start_point, end_point, theta, line_plane_normal=None):
    # points shape [batch, num_points, 3]
    # start_point shape [batch, 3]
    # end_point shape [batch, 3]
    # theta shape [batch, 1]
    prune_batch_dim = False
    if len(points.shape) == 2:
        prune_batch_dim = True
        points = points.unsqueeze(0)
        
    line_vec = end_point - start_point
    if line_plane_normal is None:
        line_plane_normal = perpendicular_vectors(line_vec)
    line_vec_scale = th.norm(line_vec, dim=-1, keepdim=True)
    line_vec_normalized = line_vec/(line_vec_scale + EPSILON)

    point_vec = points - start_point[..., None, :]
    projection = (point_vec * line_vec_normalized[..., None, :]).sum(dim=-1)
    line_param = projection/(line_vec_scale[...,  :] + EPSILON)

    closest_point = start_point[..., None, :] + projection[..., None] * line_vec_normalized[..., None, :]
    disp_from_line = points - closest_point
    along_plane_component = th.sum(disp_from_line * line_plane_normal[..., None, :], dim=-1)
    off_plane_comonent = th.norm(disp_from_line - along_plane_component[..., None] * line_plane_normal[..., None, :], dim=-1)
    # pretend the third component is the height of a cylinder.
    distance_field_2d = th.stack([along_plane_component, off_plane_comonent], dim=-1)
    # Now rotate by theta:
    c = th.cos(theta)
    s = th.sin(theta)
    rotation_matrix = th.stack([c, -s, s, c], dim=-1).view(-1, 2, 2)
    if rotation_matrix.shape[0] == 1:
        rotation_matrix = rotation_matrix.squeeze(0)
        distance_field_2d = th.matmul(distance_field_2d, rotation_matrix)
    else:
        distance_field_2d = th.bmm(distance_field_2d, rotation_matrix)
    
    parameterized_points = th.cat([distance_field_2d, line_param[..., None]], dim=-1)
    return parameterized_points, line_vec_scale

# based on: https://www.shadertoy.com/view/MlKcDD
# TODO: Make it batched.
def sdf3d_quadratic_bezier_extrude(points, start_point, control_point, end_point, theta, plane_normal=None):

    # first project to plane:
    if plane_normal is None:
        # find normal with the three points:
        vec1 = control_point - start_point
        vec2 = end_point - control_point
        plane_normal = th.cross(vec1, vec2)

    z_axis = plane_normal/(th.norm(plane_normal) + EPSILON)
    x_axis = end_point - start_point
    x_axis = x_axis/(th.norm(x_axis) + EPSILON)
    y_axis = th.cross(z_axis, x_axis)
    y_axis = y_axis/(th.norm(y_axis) + EPSILON)
    rotation_matrix = th.stack([y_axis, x_axis, z_axis], dim=-1)

    rotated_points = th.matmul(points, rotation_matrix)
    quad_points = th.stack([start_point, control_point, end_point], dim=0)

    shift = th.dot(control_point, z_axis)
    rotated_quad_points = th.matmul(quad_points, rotation_matrix)
    rotated_points[..., 2] -= shift

    xy_points = rotated_points[..., :2]
    quad_proj = rotated_quad_points[..., :2]

    # Solve with cordano's method for nearest points:
    A = quad_proj[0:1]
    B = quad_proj[1:2]
    C = quad_proj[2:3]
    pos = xy_points
    a = B - A
    b = A - 2.0 * B + C
    c = a * 2.0
    d = A - pos  # d is n x 2

    kk = 1.0 / (th.sum(b * b, dim=-1) + EPSILON)
    kx = kk * th.sum(a * b, dim=-1)

    # n x 2
    ky = kk * (2.0 * th.sum(a * a, dim=-1) + th.sum(d * b, dim=-1)) / 3.0
    kz = kk * th.sum(d * a, dim=-1)

    p = ky - kx * kx
    q = kx * (2.0 * kx * kx - 3.0 * ky) + kz
    p3 = p * p * p
    q2 = q * q
    original_h = q2 + 4.0 * p3

    # Solve for both parts- single and multiple solutions:
    # Single root
    h = original_h.clone()
    h = th.abs(h)
    h = th.sqrt(h + EPSILON)
    x_1 = (th.stack([h, -h], -1) - q.unsqueeze(1))/2

    k = (1 - p3/(q2 + EPSILON)) * p3/(q + EPSILON)
    x_2 = th.stack([k, -k-q], -1)
    p_stacked = p.unsqueeze(1).expand(-1, 2)
    x = th.where(th.abs(p_stacked) < 0.001, x_2, x_1)

    uv = th.sign(x) * th.pow(th.abs(x)+EPSILON, 1.0/3.0)
    t_1 = uv[..., 0] + uv[..., 1] - kx
    usable_t = t_1.unsqueeze(1)
    q_diff = d + (c + b * usable_t) * usable_t
    solution_1 = dot2(q_diff)
    sign_1 = cro(c + 2 * b * usable_t, q_diff)

    # Multiple roots:
    z = th.sqrt(th.abs(p) + EPSILON)
    temp = q / (p * z * 2.0 + EPSILON)
    temp = th.clamp(temp, -1 + EPSILON, 1 - EPSILON)
    v = th.acos(temp) / 3.0
    m = th.cos(v)
    n = th.sin(v) * 1.732050808
    new_vec = th.stack([m + m, -n - m], dim=1)
    t_2 = new_vec * z.unsqueeze(1) - kx.unsqueeze(0)
    usable_t = t_2  # th.clamp(t_2, 0.0, 1.0)

    sol_1_t = usable_t[..., 0:1]
    qx = d + (c + b * sol_1_t) * sol_1_t
    dx = dot2(qx)
    sx = cro(c + 2.0 * b * sol_1_t, qx)
    sol_2_t = usable_t[..., 1:2]
    qy = d + (c + b * sol_2_t) * sol_2_t
    dy = dot2(qy)
    sy = cro(c + 2.0 * b * sol_2_t, qy)

    solution_2 = th.where(dx < dy, dx, dy)
    sign_2 = th.where(dx < dy, sx, sy)
    t_2 = th.where(dx < dy, sol_1_t[..., 0], sol_2_t[..., 0])

    solution = th.where(original_h >= 0.0, solution_1, solution_2)
    sign = th.where(original_h >= 0.0, sign_1, sign_2)
    t = th.where(original_h >= 0.0, t_1, t_2)
    solution = th.abs(solution)
    solution = th.sqrt(solution + EPSILON) * th.sign(sign)


    distance_field_2d = th.stack([solution, rotated_points[..., 2]], dim=-1)
    # Now rotate by theta:
    c = th.cos(theta)
    s = th.sin(theta)
    rotation_matrix = th.stack([c, -s, s, c], dim=-1).view(-1, 2, 2)
    if rotation_matrix.shape[0] == 1:
        rotation_matrix = rotation_matrix.squeeze(0)
        distance_field_2d = th.matmul(distance_field_2d, rotation_matrix)
    else:
        distance_field_2d = th.bmm(distance_field_2d, rotation_matrix)
    
    parameterized_points = th.cat([distance_field_2d, t[..., None]], dim=-1)
    
    # approximate scale as length of quad curve is expensive to compute
    scale_factor = th.norm(end_point - control_point) + \
        th.norm(control_point - start_point)
    return parameterized_points, scale_factor

def perpendicular_vectors(vec, normalize=True):
    if th.all(vec == 0):
        raise ValueError("The input vector should not be the zero vector.")
    
    new_vec = th.zeros_like(vec)
    alternate_vec = th.zeros_like(vec)
    new_vec[..., 2] = 1
    #  remove projection of new_vec onto vec from new_vec
    perp_vec = new_vec - (new_vec * vec).sum(-1, keepdim=True) * vec
    
    cond = th.norm(perp_vec, dim=-1, keepdim=True) < EPSILON
    alternate_vec[..., 1] = 1
    perp_vec = th.where(cond, alternate_vec, perp_vec)
    if normalize:
        perp_vec = perp_vec/(th.norm(perp_vec, dim=-1, keepdim=True) + EPSILON)

    return perp_vec


def dot2(tensor):
    return th.sum(tensor * tensor, dim=-1)


def cro(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def sdf3d_revolution(points, sdf2d_func, o):
    q = th.stack([th.norm(points[..., :2], dim=-1) - o, points[..., 2]], dim=-1)
    return sdf2d_func(q)

def sdf3d_simple_extrusion(points, sdf2d_func, h):
    d = sdf2d_func(points[..., :2])
    w = th.stack([d, th.abs(points[..., 2]) - h], dim=-1)
    w_2 = th.clamp(th.amin(w, dim=-1), max=0)
    base_sdf = w_2 + th.norm(th.clamp(w, min=0.0))
    return base_sdf

def sdf1d_linear_curve():
    ...

def sdf1d_quadratic_curve():
    ...
