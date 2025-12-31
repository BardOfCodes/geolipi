import numpy as np
import torch as th
from .constants import EPSILON


def sdf_union(*args: th.Tensor) -> th.Tensor:
    """
    Parameters:
        *args: SDF tensors to be united

    Returns:
        Tensor: Union SDF (minimum values)
    """
    sdf = th.amin(th.stack(args, dim=-1), dim=-1)
    return sdf


def sdf_intersection(*args: th.Tensor) -> th.Tensor:
    """
    Parameters:
        *args: SDF tensors to be intersected

    Returns:
        Tensor: Intersection SDF (maximum values)
    """
    sdf = th.amax(th.stack(args, dim=-1), dim=-1)
    return sdf


def sdf_difference(sdf_a: th.Tensor, sdf_b: th.Tensor) -> th.Tensor:
    """
    Parameters:
        sdf_a: SDF from which to subtract
        sdf_b: SDF to subtract

    Returns:
        Tensor: Difference SDF (sdf_a - sdf_b)
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

def mix(x: th.Tensor, y: th.Tensor, a: th.Tensor) -> th.Tensor:
    """
    Parameters:
        x: First tensor
        y: Second tensor
        a: Mixing factor

    Returns:
        Tensor: Linear interpolation result
    """
    return x * (1 - a) + y * a

def sdf_smooth_union(sdf_a: th.Tensor, sdf_b: th.Tensor, k: th.Tensor) -> th.Tensor:
    """
    Parameters:
        sdf_a: First SDF
        sdf_b: Second SDF
        k: Smoothing parameter

    Returns:
        Tensor: Smooth union SDF
    """
    h = th.clamp(0.5 + 0.5 * (sdf_b - sdf_a) / (k + EPSILON), min=0.0, max=1.0)
    # h = th.clamp(k - th.abs(sdf_b - sdf_a), min=0.0)
    sdf = mix(sdf_b, sdf_a, h) - k * h * (1 - h);
    # sdf = th.minimum(sdf_a, sdf_b) - h * h * 0.25 / (k + EPSILON)
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
    h = th.clamp(0.5 - 0.5 * (sdf_b - sdf_a) / (k + EPSILON), min=0.0, max=1.0)
    sdf = mix(sdf_b, sdf_a, h) + k * h * (1 - h);
    return sdf


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
    h = th.clamp(0.5 - 0.5 * (sdf_b + sdf_a) / (k + EPSILON), min=0.0, max=1.0)
    sdf = mix(sdf_a, -sdf_b, h) + k * h * (1 - h);
    return sdf

def sdf_smooth_union_k_variable(*args,):
    """
    Computes a smooth union of K SDFs using per-SDF smoothing parameters.

    Parameters:
        sdf_list (List[torch.Tensor]): List of SDF tensors [K, ...].
        k_list (List[float or torch.Tensor]): List of K smoothing parameters.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: The resulting smooth union SDF.
    """
    sdf_list = args[:-1]
    k_tensor = args[-1]
    sdf_stack = th.stack(sdf_list, dim=0)  # shape: [K, ...]
    k_tensor = k_tensor.view(-1, *([1] * (sdf_stack.ndim - 1)))  # shape: [K, 1, ..., 1]
    sdf_min = th.amin(sdf_stack, dim=0, keepdim=True)  # shape: [...]
    diffs = sdf_stack - sdf_min  # shape: [K, ...]
    h = th.clamp(k_tensor - th.abs(diffs), min=0.0)
    blend = th.sum((h ** 2) * 0.25 / (k_tensor + EPSILON), dim=0)
    return sdf_min - blend

def sdf_smooth_intersection_k_variable(*args):
    """
    Computes a smooth intersection of K SDFs using per-SDF smoothing parameters.

    Parameters:
        *args: A sequence of torch.Tensor SDFs followed by a tensor of smoothing parameters.

    Returns:
        torch.Tensor: The resulting smooth intersection SDF.
    """
    sdf_list = args[:-1]
    k_tensor = args[-1]
    negated_sdfs = [-s for s in sdf_list]
    return -sdf_smooth_union_k_variable(*negated_sdfs, k_tensor)


def sdf_dilate(sdf_a: th.Tensor, k: th.Tensor) -> th.Tensor:
    """
    Parameters:
        sdf_a: SDF to dilate
        k: Dilation factor

    Returns:
        Tensor: Dilated SDF
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


def sdf_neg_only_onion(sdf_a, k):
    """
    Creates an "onion" effect on an SDF by subtracting a constant k from the absolute value of the SDF.

    Parameters:
        sdf_a (torch.Tensor): The SDF to be transformed.
        k (float): The onion factor.

    Returns:
        torch.Tensor: The resulting SDF after applying the onion effect.
    """
    alternate = th.abs(sdf_a) - k
    out = th.where(sdf_a <=0, alternate, sdf_a)
    return out


def sdf_onion(sdf_a: th.Tensor, k: th.Tensor) -> th.Tensor:
    """
    Parameters:
        sdf_a: SDF to transform
        k: Onion thickness factor

    Returns:
        Tensor: Onion SDF
    """
    return th.abs(sdf_a) - k
    

def sdf_null_op(points: th.Tensor) -> th.Tensor:
    """
    Parameters:
        points: Input points

    Returns:
        Tensor: Null SDF (small positive values)
    """
    return th.zeros_like(points[..., 0]) + EPSILON

def sdf_xor(sdf_a, sdf_b):
    """
    Computes the XOR of two SDFs.
    """
    return th.maximum(th.minimum(sdf_a, sdf_b), -th.maximum(sdf_a, sdf_b))