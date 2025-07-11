import numpy as np
import torch as th
from .common import EPSILON


def sdf_union(*args):
    """
    Computes the union of multiple SDFs. The union operation finds the minimum SDF value at each point.

    Parameters:
        *args (tuple of torch.Tensor): An arbitrary number of SDF tensors to be united.

    Returns:
        torch.Tensor: The resulting SDF representing the union of the input SDFs.
    """
    sdf = th.amin(th.stack(args, dim=-1), dim=-1)
    return sdf


def sdf_intersection(*args):
    """
    Computes the intersection of multiple SDFs. The intersection operation finds the maximum SDF value at each point.

    Parameters:
        *args (tuple of torch.Tensor): An arbitrary number of SDF tensors to be intersected.

    Returns:
        torch.Tensor: The resulting SDF representing the intersection of the input SDFs.
    """
    sdf = th.amax(th.stack(args, dim=-1), dim=-1)
    return sdf


def sdf_difference(sdf_a, sdf_b):
    """
    Computes the difference between two SDFs (sdf_a - sdf_b).

    Parameters:
        sdf_a (torch.Tensor): The SDF from which the second SDF is subtracted.
        sdf_b (torch.Tensor): The SDF to be subtracted from the first SDF.

    Returns:
        torch.Tensor: The resulting SDF representing the difference between sdf_a and sdf_b.
    """
    sdf = th.maximum(sdf_a, -sdf_b)
    return sdf


def sdf_switched_difference(sdf_a, sdf_b):
    """
    Computes the difference between two SDFs (sdf_b - sdf_a), opposite of sdf_difference.

    Parameters:
        sdf_a (torch.Tensor): The SDF to be subtracted from the second SDF.
        sdf_b (torch.Tensor): The SDF from which the first SDF is subtracted.

    Returns:
        torch.Tensor: The resulting SDF representing the difference between sdf_b and sdf_a.
    """
    sdf = th.maximum(sdf_b, -sdf_a)
    return sdf


def sdf_complement(sdf_a):
    """
    Computes the complement of an SDF.

    Parameters:
        sdf_a (torch.Tensor): The SDF for which the complement is calculated.

    Returns:
        torch.Tensor: The resulting SDF representing the complement of sdf_a.
    """
    sdf = -sdf_a
    return sdf


def sdf_smooth_union(sdf_a, sdf_b, k):
    """
    Computes a smooth union of two SDFs using a smoothing parameter k.

    Parameters:
        sdf_a (torch.Tensor): The first SDF.
        sdf_b (torch.Tensor): The second SDF.
        k (float): The smoothing parameter.

    Returns:
        torch.Tensor: The resulting SDF representing the smooth union of sdf_a and sdf_b.
    """
    h = th.clamp(k - th.abs(sdf_b - sdf_a), min=0.0)
    sdf = th.minimum(sdf_a, sdf_b) - h * h * 0.25 / (k + EPSILON)
    return sdf


def sdf_smooth_intersection(sdf_a, sdf_b, k):
    """
    Computes a smooth intersection of two SDFs using a smoothing parameter k.

    Parameters:
        sdf_a (torch.Tensor): The first SDF.
        sdf_b (torch.Tensor): The second SDF.
        k (float): The smoothing parameter.

    Returns:
        torch.Tensor: The resulting SDF representing the smooth intersection of sdf_a and sdf_b.
    """
    return -sdf_smooth_union(-sdf_a, -sdf_b, k)


def sdf_smooth_difference(sdf_a, sdf_b, k):
    """
    Computes a smooth difference between two SDFs using a smoothing parameter k.

    Parameters:
        sdf_a (torch.Tensor): The first SDF.
        sdf_b (torch.Tensor): The second SDF.
        k (float): The smoothing parameter.

    Returns:
        torch.Tensor: The resulting SDF representing the smooth difference of sdf_a and sdf_b.
    """
    return sdf_smooth_intersection(sdf_a, -sdf_b, k)


def sdf_dilate(sdf_a, k):
    """
    Dilates an SDF by a factor k.

    Parameters:
        sdf_a (torch.Tensor): The SDF to be dilated.
        k (float): The dilation factor.

    Returns:
        torch.Tensor: The resulting SDF after dilation.
    """
    return sdf_a - k


def sdf_erode(sdf_a, k):
    """
    Erodes an SDF by a factor k.

    Parameters:
        sdf_a (torch.Tensor): The SDF to be eroded.
        k (float): The erosion factor.

    Returns:
        torch.Tensor: The resulting SDF after erosion.
    """
    return sdf_a + k


def sdf_onion(sdf_a, k):
    """
    Creates an "onion" effect on an SDF by subtracting a constant k from the absolute value of the SDF.

    Parameters:
        sdf_a (torch.Tensor): The SDF to be transformed.
        k (float): The onion factor.

    Returns:
        torch.Tensor: The resulting SDF after applying the onion effect.
    """
    return th.abs(sdf_a) - k


def sdf_null_op(points):
    """
    A null operation for SDFs.

    Parameters:
        points (torch.Tensor): A tensor representing points in space.

    Returns:
        torch.Tensor: A tensor of zeros with a shape matching the first dimension of the input points.
    """
    return th.zeros_like(points[..., 0]) + EPSILON

def sdf_xor(sdf_a, sdf_b):
    """
    Computes the XOR of two SDFs.
    """
    return th.maximum(th.minimum(sdf_a, sdf_b), -th.maximum(sdf_a, sdf_b))