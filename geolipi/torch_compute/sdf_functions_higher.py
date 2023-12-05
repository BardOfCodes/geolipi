
import torch as th
import numpy as np
from .common import EPSILON, ACOS_EPSILON
from .sdf_functions_2d import SQRT_3
# The return of a curve distance should be:
# 1) The parameter t on the curve (0, 1) which corresponds to the closest point
# 3) The projection of the point into the plane with normal plane_normal

def sdf3d_linear_extrude(points, start_point, end_point, theta, line_plane_normal=None):
    # points shape [batch, num_points, 3]
    # start_point shape [batch, 3]
    # end_point shape [batch, 3]
    # theta shape [batch, 1]
    if len(points.shape) == 2:
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
# TODO: Still buggy.
def sdf3d_quadratic_bezier_extrude(points, start_point, control_point, end_point, theta, plane_normal=None):

    # stack into a batch
    # points = th.stack([points, points], dim=0)
    # start_point = th.stack([start_point, start_point], dim=0)
    # control_point = th.stack([control_point, control_point], dim=0)
    # end_point = th.stack([end_point, end_point], dim=0)
    # theta = th.stack([theta, theta], dim=0)
    # if len(points.shape) == 2:
    #     points = points.unsqueeze(0)
    # first project to plane:
    if plane_normal is None:
        # find normal with the three points:
        vec1 = control_point - start_point
        vec2 = end_point - control_point
        plane_normal = th.cross(vec1, vec2, dim=-1)
        plane_norms = th.norm(plane_normal, dim=-1, keepdim=True)
        z_axis_original = th.zeros_like(plane_normal)
        z_axis_original[..., 2] = 1.0
        plane_normal = th.where(plane_norms < EPSILON, z_axis_original, plane_normal)
        

    z_axis = plane_normal/(th.norm(plane_normal, dim=-1, keepdim=True) + EPSILON)
    x_axis = end_point - start_point
    x_axis = x_axis/(th.norm(x_axis, dim=-1, keepdim=True) + EPSILON)
    y_axis = th.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis/(th.norm(y_axis, dim=-1, keepdim=True) + EPSILON)
    rotation_matrix = th.stack([y_axis, x_axis, z_axis], dim=-1)

    rotated_points = th.matmul(points, rotation_matrix)
    quad_points = th.stack([start_point, control_point, end_point], dim=-1).swapaxes(-2, -1)

    shift = (control_point * z_axis).sum(-1, keepdim=True)
    rotated_quad_points = th.matmul(quad_points, rotation_matrix)
    rotated_points[..., 2] -= shift

    xy_points = rotated_points[..., :2]
    quad_proj = rotated_quad_points[..., :2]

    # Solve with cordano's method for nearest points:
    A = quad_proj[..., 0:1, :]
    B = quad_proj[..., 1:2, :]
    C = quad_proj[..., 2:3, :]
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
    p = th.where(th.abs(p) < EPSILON, EPSILON * th.sign(p), p)
    q = kx * (2.0 * kx * kx - 3.0 * ky) + kz
    q = th.where(th.abs(q) < EPSILON, EPSILON * th.sign(q), q)
    p3 = p * p * p
    q2 = q * q
    original_h = q2 + 4.0 * p3

    # Solve for both parts- single and multiple solutions:
    # Single root
    h = original_h.clone()
    h = th.abs(h)
    h = th.sqrt(h + EPSILON)
    x_1 = (th.stack([h, -h], -1) - q.unsqueeze(-1))/2

    k = (1 - p3/q2) * p3/q
    x_2 = th.stack([k, -k-q], -1)
    x = th.where(p[..., :, None] < 0.001, x_2, x_1)

    uv = th.sign(x) * th.pow(th.abs(x)+EPSILON, 1.0/3.0)
    t_1 = uv[..., 0] + uv[..., 1] - kx
    usable_t = t_1.unsqueeze(-1)
    q_diff = d + (c + b * usable_t) * usable_t
    solution_1 = dot2(q_diff)
    sign_1 = cro(c + 2 * b * usable_t, q_diff)

    # Multiple roots:
    z = th.sqrt(th.abs(p) + EPSILON)
    acos_in = q / (p * z * 2.0)
    acos_in = th.clamp(acos_in, -1 + ACOS_EPSILON, 1 - ACOS_EPSILON)
    v = th.acos(acos_in) / 3.0
    m = th.cos(v)
    n = th.sin(v) * SQRT_3
    new_vec = th.stack([m + m, -n - m], dim=-1)
    t_2 = new_vec * z.unsqueeze(-1) - kx.unsqueeze(1)
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
    if len(points.shape) == 2:
        rotation_matrix = th.stack([c, -s, s, c], dim=-1).view(2, 2)
    else:
        rotation_matrix = th.stack([c, -s, s, c], dim=-1).view(-1, 2, 2)
    distance_field_2d = th.matmul(distance_field_2d, rotation_matrix)
    
    parameterized_points = th.cat([distance_field_2d, t[..., None]], dim=-1)
    
    # approximate scale as length of quad curve is expensive to compute
    scale_factor = th.norm(end_point - control_point, dim=-1) + \
        th.norm(control_point - start_point, dim=-1)
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


def sdf3d_revolution(points, o):
    param_x = th.norm(points[..., :2], dim=-1) - o
    param_y = points[..., 2]
    param_z = th.ones_like(param_x)
    scale_factor = 0
    parameterized_points = th.stack([param_x, param_y, param_z], dim=-1)
    return parameterized_points, scale_factor

def sdf3d_simple_extrusion(points, h):
    # points shape [batch, num_points, 3]
    # h shape [batch, 1]
    parameterized_points = points[..., :2]
    to_scale = (points[..., 2:3] + h/2) / (h + EPSILON)
    parameterized_points = th.cat([parameterized_points, to_scale], dim=-1)
    return parameterized_points, h


def linear_curve_1d(points, point1, point2):
    # points shape [batch, num_points, 1]
    # point1 shape [batch, 2]
    # point2 shape [batch, 2]
    demon = point2[..., 0] - point1[..., 0]
    demon = th.where(demon==0, EPSILON, demon)
    m = (point2[..., 1] - point1[..., 1])/demon
    c = point1[..., 1] - m * point1[..., 0]
    output = points[..., 0] * m + c
    return output

def quadratic_curve_1d(points, param_a, param_b, param_c):
    # points shape [batch, num_points, 1]
    # param_a shape [batch, 1]
    # param_b shape [batch, 1]
    # param_c shape [batch, 1]
    output = param_a * points[..., 0] * points[..., 0] + param_b * points[..., 0] + param_c
    return output


