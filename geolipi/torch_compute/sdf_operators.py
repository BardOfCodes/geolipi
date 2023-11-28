import numpy as np
import torch as th
from .common import EPSILON


def sdf_union(*args):
    sdf = th.amin(th.stack(args, dim=-1), dim=-1)
    return sdf

def sdf_intersection(*args):
    sdf = th.amax(th.stack(args, dim=-1), dim=-1)
    return sdf

def sdf_difference(sdf_a, sdf_b):
    sdf = th.maximum(sdf_a, -sdf_b)
    return sdf

def sdf_switched_difference(sdf_a, sdf_b):
    sdf = th.maximum(sdf_b, -sdf_a)
    return sdf

def sdf_complement(sdf_a):
    sdf = -sdf_a
    return sdf

def sdf_smooth_union(sdf_a, sdf_b, k):
    h = th.clamp(k - th.abs(sdf_b - sdf_a), min=0.0)
    sdf = th.minimum(sdf_a, sdf_b) - h * h * 0.25 / (k + EPSILON)
    return sdf

def sdf_smooth_intersection(sdf_a, sdf_b, k):
    return -sdf_smooth_union(-sdf_a, -sdf_b, k)

def sdf_smooth_difference(sdf_a, sdf_b, k):
    return sdf_smooth_intersection(sdf_a, -sdf_b, k)

def sdf_dilate(sdf_a, k):
    return sdf_a - k

def sdf_erode(sdf_a, k):
    return sdf_a + k

def sdf_onion(sdf_a, k):
    return th.abs(sdf_a) - k
