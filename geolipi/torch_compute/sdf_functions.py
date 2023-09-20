import numpy as np
import torch as th
from .common import EPSILON

SQRT_3 = np.sqrt(3, dtype=np.float32)

def sdf2d_rectangle(points, params):
    points = th.abs(points)
    points[..., 0] -= params[0] / 2
    points[..., 1] -= params[1] / 2
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + \
        th.clip(th.amax(points, -1), max=0)
    return base_sdf

def sdf2d_circle(points, params):
    return sdf3d_sphere(points, params)

def sdf2d_no_param_rectangle(points, _):
    points = th.abs(points)
    points[..., 0] -= 0.5
    points[..., 1] -= 0.5
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + \
        th.clip(th.amax(points, -1), max=0)
    return base_sdf

def sdf2d_no_param_circle(points, _):
    return sdf3d_no_param_sphere(points, _)

def sdf3d_cuboid(points, params):
    points = th.abs(points)
    points[..., 0] -= params[0] / 2
    points[..., 1] -= params[1] / 2
    points[..., 2] -= params[2] / 2
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + \
        th.clip(th.amax(points, -1), max=0)
    return base_sdf


def sdf3d_sphere(points, raidus):
    base_sdf = points.norm(dim=-1)
    base_sdf = base_sdf - raidus
    return base_sdf

def sdf2d_triangle(points, params):
    """ Assuming params is of shape (3, 2)"""
    # How to do it for batch of triangles? TBD
    p0 = params[0:1]
    p1 = params[1:2]
    p2 = params[2:3]
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2
    all_es = th.cat([e0, e1, e2], dim=0)
    all_vs = points.unsqueeze(-2) - params
    all_pq = all_vs - all_es.unsqueeze(0) * th.clamp(all_vs[..., 0] * all_es[..., 0] + all_vs[..., 1] * all_es[..., 1]/\
        th.norm(all_es, dim=-1), 0.0, 1.0).unsqueeze(-1)
    s = th.sign(th.det(th.cat([e0, e2], dim=0)))
    
    dim_1 = th.norm(all_pq, dim=-1)
    val = -th.sqrt(th.min(dim_1, -1)[0])
    dim_2 = s * (all_vs[..., 0] * all_es[..., 1] - all_vs[..., 1] * all_es[..., 0])
    sign = th.sign(th.min(dim_2, -1)[0])
    # final = th.stack([th.min(dim_1, -1), th.min(dim_2, -1)], -2)
    sdf = val * sign
    return sdf
    
def sdf2d_equilateral_triangle(points, params):
    k = th.tensor(SQRT_3).to(points.device)
    r = params[..., 0]
    points[..., 0] = th.abs(points[..., 0]) - r
    points[..., 1] = points[..., 1] + r/k
    condition = (points[..., 0] + (k * points[..., 1]) > 0)
    new_points = th.stack([points[..., 0] - k * points[..., 1],
                            -k * points[..., 0] - points[..., 1]], -1) / 2.0
    points = th.where(condition.unsqueeze(-1), new_points, points)
    points[..., 0] = points[..., 0] - th.clamp(points[..., 0], -2 * r, 0)
    sdf = -th.norm(points, dim=-1) * th.sign(points[..., 1])
    return sdf
    
def sdf2d_no_param_triangle(points, _):
    """Equilateral triangle"""
    k = th.tensor(SQRT_3).to(points.device)
    r = 0.5
    points[..., 0] = th.abs(points[..., 0]) - r
    points[..., 1] = points[..., 1] + r/k
    condition = (points[..., 0] + (k * points[..., 1]) > 0)
    new_points = th.stack([points[..., 0] - k * points[..., 1],
                            -k * points[..., 0] - points[..., 1]], -1) / 2.0
    points = th.where(condition.unsqueeze(-1), new_points, points)
    points[..., 0] = points[..., 0] - th.clamp(points[..., 0], -2 * r, 0)
    sdf = -th.norm(points, dim=-1) * th.sign(points[..., 1])
    return sdf
    

def sdf3d_cylinder(points, params):
    r = params[0]
    h = params[1] / 2
    xy_vec = th.norm(points[:, :, :, :2], dim=-1) - r
    height = th.abs(points[:, :, :, 2]) - h
    vec2 = th.stack([xy_vec, height], -1)
    sdf = th.amax(vec2, 3) + th.norm(th.clip(vec2, min=0.0) + EPSILON, -1)
    return sdf


def sdf3d_no_param_cuboid(points, _):
    points = th.abs(points)
    points -= 0.5
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + \
        th.clip(th.amax(points, -1), max=0)
    return base_sdf


def sdf3d_no_param_sphere(points, _):
    base_sdf = points.norm(dim=-1)
    base_sdf = base_sdf - 0.5
    return base_sdf


def sdf3d_no_param_cylinder(points, _):
    r = 0.5
    h = 0.5
    xy_vec = th.norm(points[..., :2], dim=-1) - r
    height = th.abs(points[..., 2]) - h
    vec2 = th.stack([xy_vec, height], -1)
    base_sdf = th.amax(vec2, -1) + th.norm(th.clip(vec2, min=0.0) + EPSILON, -1)
    return base_sdf


def sdf_union(*args):
    sdf = th.min(th.stack(args, dim=-1), dim=-1)[0]
    return sdf


def sdf_intersection(*args):
    sdf = th.max(th.stack(args, dim=-1), dim=-1)[0]
    return sdf

def sdf_difference(sdf_a, sdf_b):
    sdf = th.maximum(sdf_a, -sdf_b)
    return sdf

def sdf_complement(sdf_a):
    sdf = -sdf_a
    return sdf