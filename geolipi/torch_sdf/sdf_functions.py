import torch as th
from .common import EPSILON


def sdf_cuboid(points, params):
    points = th.abs(points)
    points[..., 0] -= params[0] / 2
    points[..., 1] -= params[1] / 2
    points[..., 2] -= params[2] / 2
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + \
        th.clip(th.amax(points, -1), max=0)
    return base_sdf


def sdf_sphere(points, raidus):
    base_sdf = points.norm(dim=-1)
    base_sdf = base_sdf - raidus
    return base_sdf


def sdf_cylinder(points, params):
    r = params[0]
    h = params[1] / 2
    xy_vec = th.norm(points[:, :, :, :2], dim=-1) - r
    height = th.abs(points[:, :, :, 2]) - h
    vec2 = th.stack([xy_vec, height], -1)
    sdf = th.amax(vec2, 3) + th.norm(th.clip(vec2, min=0.0) + EPSILON, -1)
    return sdf


def no_param_cuboid(points, _):
    points = th.abs(points)
    points[..., 0] -= 0.5
    points[..., 1] -= 0.5
    points[..., 2] -= 0.5
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + \
        th.clip(th.amax(points, -1), max=0)
    return base_sdf


def no_param_sphere(points, _):
    base_sdf = points.norm(dim=-1)
    base_sdf = base_sdf - 0.5
    return base_sdf


def no_param_cylinder(points, _):
    r = 0.5
    h = 0.5
    xy_vec = th.norm(points[:, :, :, :2], dim=-1) - r
    height = th.abs(points[:, :, :, 2]) - h
    vec2 = th.stack([xy_vec, height], -1)
    base_sdf = th.amax(vec2, 3) + th.norm(th.clip(vec2, min=0.0) + EPSILON, -1)
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