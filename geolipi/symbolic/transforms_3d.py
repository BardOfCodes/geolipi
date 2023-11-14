from typing import Tuple, List

import sympy
from .base_symbolic import GLExpr, GLFunction
from .common import expr_type, param_type_1D, param_type_2D, param_type_3D, param_type_4D, sig_check


class Modifier3D(GLFunction):
    ...


class Transform3D(Modifier3D):
    ...


class PositionalTransform3D(Transform3D):
    ...


class Macro3D(Modifier3D):
    ...


class SDFModifier3D(Modifier3D):
    ...


class Translate3D(Transform3D):
    ...


class EulerRotate3D(Transform3D):
    ...


class Scale3D(Transform3D):
    ...


class QuaternionRotate3D(Transform3D):
    ...

# For Continuous optimization of rotation
# TODO: Higher order for differentiable optimization of rotation
# Ref: https://arxiv.org/abs/1812.07035


class Rotate5D(Transform3D):
    ...


class Rotate6D(Transform3D):
    ...


class Rotate9D(Transform3D):
    ...


class Shear3D(Transform3D):
    ...


class Distort3D(PositionalTransform3D):
    ...


class Twist3D(PositionalTransform3D):
    ...


class Bend3D(PositionalTransform3D):
    ...


class ReflectCoords3D(Transform3D):
    """Simply reflects the coordinates about the origin, w.r.t. the normal vector specified by param."""
    ...


# TODO: Implementation
class Reflect3D(Macro3D):
    """Performs union of canvas and its reflection about the origin, 
    with the reflection plane's normal vector specified by param.
    """
    ...


class ReflectX3D(Reflect3D):
    ...


class ReflectY3D(ReflectX3D):
    ...


class ReflectZ3D(ReflectX3D):
    ...


class AxialReflect3D(Reflect3D):
    ...


class TranslationSymmetry3D(Macro3D):
    ...


class AxialTranslationSymmetry3D(TranslationSymmetry3D):
    ...


class TranslationSymmetryX3D(TranslationSymmetry3D):
    ...


class TranslationSymmetryY3D(TranslationSymmetryX3D):
    ...


class TranslationSymmetryZ3D(TranslationSymmetryX3D):
    ...


class RotationSymmetry3D(Macro3D):
    ...


class AxialRotationSymmetry3D(RotationSymmetry3D):
    ...


class RotationSymmetryX3D(RotationSymmetry3D):
    ...


class RotationSymmetryY3D(RotationSymmetryX3D):
    ...


class RotationSymmetryZ3D(RotationSymmetryX3D):
    ...


class ScaleSymmetry3D(Macro3D):
    ...


class AxialScaleSymmetry3D(ScaleSymmetry3D):
    ...


class ColorTree3D(SDFModifier3D):
    ...


class Dilate3D(SDFModifier3D):
    ...


class Erode3D(SDFModifier3D):
    ...


class Onion3D(SDFModifier3D):
    ...
