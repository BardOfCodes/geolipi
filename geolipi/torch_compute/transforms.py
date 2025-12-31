import torch as th
import numpy as np
from .constants import EPSILON
from .settings import Settings


def get_affine_matrix_2D(matrix: th.Tensor, params: th.Tensor) -> th.Tensor:
    """
    Parameters:
        matrix: Affine transformation matrix
        params: Transformation parameters

    Returns:
        Tensor: Updated transformation parameters
    """
    # matrix = matrix @ params
    return params

def get_affine_translate_2D(matrix: th.Tensor, param: th.Tensor) -> th.Tensor:
    """
    Parameters:
        matrix: Affine transformation matrix to modify
        param: Translation parameters [dx, dy]

    Returns:
        Tensor: Modified affine transformation matrix
    """
    matrix[:2, 2] = -param
    return matrix


def get_affine_scale_2D(matrix: th.Tensor, param: th.Tensor) -> th.Tensor:
    """
    Parameters:
        matrix: Affine transformation matrix to modify
        param: Scaling factors [sx, sy]

    Returns:
        Tensor: Modified affine transformation matrix
    """
    matrix[0, 0] = 1 / (EPSILON + param[0])
    matrix[1, 1] = 1 / (EPSILON + param[1])
    return matrix


# Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L461


def get_affine_reflection_2D(matrix, param):
    """
    Applies a 2D reflection to the given affine transformation matrix based on a line of reflection.

    Parameters:
        matrix (torch.Tensor): The affine transformation matrix to be modified.
        param (list or torch.Tensor): The reflection line vector.

    Returns:
        torch.Tensor: The modified affine transformation matrix.
    """
    vector = param / (th.norm(param) + EPSILON)
    outer_product = th.outer(vector, vector)
    identity = th.eye(2, dtype=matrix.dtype, device=matrix.device)
    reflection_matrix = identity - 2 * outer_product
    matrix[:2, :2] = reflection_matrix
    return matrix


def get_affine_rotate_2D(matrix: th.Tensor, param: th.Tensor) -> th.Tensor:
    """
    Parameters:
        matrix: Affine transformation matrix to modify
        param: Rotation angle in radians

    Returns:
        Tensor: Modified affine transformation matrix
    """
    c = th.cos(-param)
    s = th.sin(-param)

    matrix[0, 0] = c
    matrix[0, 1] = -s
    matrix[1, 0] = s
    matrix[1, 1] = c
    return matrix

def get_affine_matrix_3D(matrix, param):
    """
    Applies a given affine transformation matrix.
    """
    return param

def get_affine_translate_3D(matrix, param):
    """
    Applies a 3D translation to the given affine transformation matrix.

    Parameters:
        matrix (torch.Tensor): The affine transformation matrix to be modified.
        param (list or torch.Tensor): The translation parameters [dx, dy, dz].

    Returns:
        torch.Tensor: The modified affine transformation matrix.
    """
    matrix[:3, 3] = -param
    return matrix


def get_affine_scale_3D(matrix, param):
    """
    Applies a 3D scaling to the given affine transformation matrix.

    Parameters:
        matrix (torch.Tensor): The affine transformation matrix to be modified.
        param (list or torch.Tensor): The scaling factors [sx, sy, sz].

    Returns:
        torch.Tensor: The modified affine transformation matrix.
    """
    scale_var = 1 / (param + EPSILON)

    matrix[0, 0] = scale_var[0]
    matrix[1, 1] = scale_var[1]
    matrix[2, 2] = scale_var[2]
    return matrix


def get_affine_reflection_3D(matrix, param):
    """
    Applies a 3D reflection to the given affine transformation matrix based on a plane of reflection.

    Parameters:
        matrix (torch.Tensor): The affine transformation matrix to be modified.
        param (list or torch.Tensor): The reflection plane normal vector.

    Returns:
        torch.Tensor: The modified affine transformation matrix.
    """
    vector = param / (th.norm(param) + EPSILON)
    outer_product = th.outer(vector, vector)
    identity = th.eye(3, dtype=matrix.dtype, device=matrix.device)
    reflection_matrix = identity - 2 * outer_product
    matrix[:3, :3] = reflection_matrix
    return matrix

def get_affine_shear_3D(matrix, param):
    """
    Applies a 3D shear transformation to the given affine matrix.

    Parameters:
        matrix (torch.Tensor): The affine transformation matrix to be modified.
        param (list or torch.Tensor): The shear factors for each axis.

    Returns:
        torch.Tensor: The modified affine transformation matrix.
    """
    shear_matrix = th.eye(3, dtype=matrix.dtype, device=matrix.device)
    shear_matrix[0, 1] = param[0]
    shear_matrix[0, 2] = param[1]
    shear_matrix[1, 2] = param[2]
    matrix[:3, :3] = shear_matrix
    return matrix


def get_affine_shear_2D(matrix, param):
    """
    Applies a 2D shear transformation to the given affine matrix.

    Parameters:
        matrix (torch.Tensor): The affine transformation matrix to be modified.
        param (float or torch.Tensor): The shear factor.

    Returns:
        torch.Tensor: The modified affine transformation matrix.
    """
    shear_matrix = th.eye(2, dtype=matrix.dtype, device=matrix.device)
    shear_matrix[0, 1] = param
    matrix[:2, :2] = shear_matrix
    return matrix

def get_affine_rotate_euler_3D(matrix, param):
    """
    Applies a 3D rotation to the given affine transformation matrix using Euler angles.

    Parameters:
        matrix (torch.Tensor): The affine transformation matrix to be modified.
        param (list or torch.Tensor): The Euler angles for rotation.

    Returns:
        torch.Tensor: The modified affine transformation matrix.
    """
    matrices = [
        _axis_angle_rotation_3D(c, e) for c, e in zip(Settings.ROT_ORDER, th.unbind(param, -1))
    ]
    rotation_matrix = th.matmul(th.matmul(matrices[0], matrices[1]), matrices[2])

    matrix[:3, :3] = rotation_matrix
    return matrix




# Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L461
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


def axis_angle_to_rotation_matrix(axis_angle: th.Tensor) -> th.Tensor:
    """
    Convert an axis-angle vector (..., 3) to a rotation matrix (..., 3, 3)
    using the Rodrigues' rotation formula.
    """
    theta = th.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)  # (..., 1)
    axis = axis_angle / theta  # normalized axis (..., 3)

    x, y, z = axis.unbind(-1)  # each (...,)
    zero = th.zeros_like(x)
    K = th.stack([
        zero, -z,    y,
         z,  zero,  -x,
        -y,   x,   zero
    ], dim=-1).reshape(axis.shape[:-1] + (3, 3))  # (..., 3, 3)

    I = th.eye(3, device=axis.device, dtype=axis.dtype).expand(K.shape)
    sin = th.sin(theta)[..., None]
    cos = th.cos(theta)[..., None]

    R = I + sin * K + (1 - cos) * (K @ K)  # Rodrigues' formula

    return R
    
def get_affine_rotate_axis_angle_3D(matrix, param):
    """
    Applies a 3D rotation to the given affine transformation matrix using axis-angle representation.
    """
    rotation_matrix = axis_angle_to_rotation_matrix(param)
    matrix[:3, :3] = rotation_matrix
    return matrix

def get_affine_rotate_matrix_3D(matrix, param):
    """
    Applies a 3D rotation to the given affine transformation matrix using a rotation matrix.
    """
    matrix[:3, :3] = param
    return matrix

    
def position_distort(positions: th.Tensor, k: th.Tensor) -> th.Tensor:
    """
    Parameters:
        positions: Original positions
        k: Distortion magnitude

    Returns:
        Tensor: Distorted positions
    """
    positions = positions + th.rand_like(positions) * k
    return positions


def position_twist(positions: th.Tensor, k: th.Tensor) -> th.Tensor:
    """
    Parameters:
        positions: Original positions
        k: Twist magnitude based on z-coordinate

    Returns:
        Tensor: Twisted positions
    """
    k = -k
    c = th.cos(k * positions[..., 2])
    s = th.sin(k * positions[..., 2])
    rot = th.stack([c, -s, s, c], dim=-1).reshape(*c.shape, 2, 2)
    q = th.cat(
        [th.bmm(rot, positions[..., :2, None])[..., 0], positions[..., 2:]], dim=-1
    )
    return q


def position_cheap_bend(positions: th.Tensor, k: th.Tensor) -> th.Tensor:
    """
    Parameters:
        positions: Original positions
        k: Bend magnitude based on x-coordinate

    Returns:
        Tensor: Bent positions
    """
    c = th.cos(k * positions[..., 0])
    s = th.sin(k * positions[..., 0])
    m = th.stack([c, -s, s, c], dim=-1).reshape(*c.shape, 2, 2)
    q = th.cat(
        [th.bmm(m, positions[..., :2, None])[..., 0], positions[..., 2:]], dim=-1
    )
    return q

