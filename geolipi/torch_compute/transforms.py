import torch as th
import numpy as np
from .common import EPSILON

CONVENTION = ("X", "Y", "Z")

def get_affine_translate_2D(matrix, param):
    if not isinstance(param, th.Tensor):
        param = th.tensor(param, dtype=matrix.dtype, device=matrix.device)
    matrix[:2, 2] = -param
    return matrix


def get_affine_scale_2D(matrix, param):
    if not isinstance(param, th.Tensor):
        param = th.tensor(param, dtype=matrix.dtype, device=matrix.device)
    matrix[0, 0] = 1 / (EPSILON + param[0])
    matrix[1, 1] = 1 / (EPSILON + param[1])
    return matrix

# Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L461

def get_affine_reflection_2D(matrix, param):
    if not isinstance(param, th.Tensor):
        param = th.tensor(param, dtype=matrix.dtype, device=matrix.device)
    vector = param / (th.norm(param) + EPSILON)
    outer_product = th.outer(vector, vector)
    identity = th.eye(2, dtype=matrix.dtype, device=matrix.device)
    reflection_matrix = identity - 2 * outer_product
    matrix[:2, :2] = reflection_matrix
    return matrix
    
def get_affine_rotate_2D(matrix, param):
    if not isinstance(param, th.Tensor):
        param = th.tensor(param, dtype=matrix.dtype, device=matrix.device)
    c = th.cos(-param)
    s = th.sin(-param)
    
    matrix[0, 0] = c
    matrix[0, 1] = -s
    matrix[1, 0] = s
    matrix[1, 1] = c
    return matrix

def get_affine_translate_3D(matrix, param):
    if not isinstance(param, th.Tensor):
        param = th.tensor(param, dtype=matrix.dtype, device=matrix.device)
    matrix[:3, 3] = -param
    return matrix


def get_affine_scale_3D(matrix, param):
    if not isinstance(param, th.Tensor):
        param = th.tensor(param, dtype=matrix.dtype, device=matrix.device)
    matrix[0, 0] = 1 / (EPSILON + param[0])
    matrix[1, 1] = 1 / (EPSILON + param[1])
    matrix[2, 2] = 1 / (EPSILON + param[2])
    return matrix

# Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L461

def get_affine_reflection_3D(matrix, param):
    if not isinstance(param, th.Tensor):
        param = th.tensor(param, dtype=matrix.dtype, device=matrix.device)
    vector = param / (th.norm(param) + EPSILON)
    outer_product = th.outer(vector, vector)
    identity = th.eye(3, dtype=matrix.dtype, device=matrix.device)
    reflection_matrix = identity - 2 * outer_product
    matrix[:3, :3] = reflection_matrix
    return matrix
    

    

def _axis_angle_rotation_3D(axis: str, angle: th.Tensor) -> th.Tensor:

    cos = th.cos(angle)
    sin = th.sin(angle)
    one = th.ones_like(angle)
    zero = th.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return th.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def get_affine_rotate_euler_3D(matrix, param):
    if not isinstance(param, th.Tensor):
        param = th.tensor(param, dtype=matrix.dtype, device=matrix.device)
    matrices = [
        _axis_angle_rotation_3D(c, e)
        for c, e in zip(CONVENTION, th.unbind(param, -1))
    ]
    rotation_matrix = th.matmul(
        th.matmul(matrices[0], matrices[1]), matrices[2])

    matrix[:3, :3] = rotation_matrix
    return matrix