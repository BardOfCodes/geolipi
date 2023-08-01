import torch as th


def sdf_cuboid(points, params):
    points = th.abs(points)
    points[..., 0] -= params[0] / 2
    points[..., 1] -= params[1] / 2
    points[..., 2] -= params[2] / 2
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + \
        th.clip(th.amax(points, -1), max=0)
    return base_sdf


def fixed_cuboid(points):
    points = th.abs(points)
    points[..., 0] -= 0.5
    points[..., 1] -= 0.5
    points[..., 2] -= 0.5
    base_sdf = th.norm(th.clip(points, min=0), dim=-1) + \
        th.clip(th.amax(points, -1), max=0)
    return base_sdf


def fixed_sphere(points):
    base_sdf = points.norm(dim=-1)
    base_sdf = base_sdf - 0.5
    return base_sdf


def sdf_sphere(points, raidus):
    base_sdf = points.norm(dim=-1)
    base_sdf = base_sdf - raidus
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
