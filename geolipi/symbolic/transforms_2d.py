from typing import Tuple, List
from .base_symbolic import GLExpr, GLFunction
import sympy

class Modifier2D(GLFunction):
    ...


class Transform2D(Modifier2D):
    ...


class PositionalTransform2D(Modifier2D):
    ...


class Macro2D(Modifier2D):
    ...


class SDFModifier2D(Modifier2D):
    ...


class Translate2D(Transform2D):
    ...


class EulerRotate2D(Transform2D):
    ...


class Scale2D(Transform2D):
    ...


class Shear2D(Transform2D):
    ...


class Distort2D(PositionalTransform2D):
    ...


class ReflectCoords2D(Transform2D):
    """Simply reflects the coordinates about the origin, w.r.t. the normal vector specified by param."""
    ...


class Reflect2D(Macro2D):
    """Performs union of canvas and its reflection about the origin, 
    with the reflection plane's normal vector specified by param.
    """


class ReflectX2D(Reflect2D):
    ...


class ReflectY2D(ReflectX2D):
    ...


class AxialReflect2D(Macro2D):
    ...


class TranslationSymmetry2D(Macro2D):
    ...


class AxialTranslationSymmetry2D(TranslationSymmetry2D):
    ...


class TranslationSymmetryX2D(TranslationSymmetry2D):
    ...


class TranslationSymmetryY2D(TranslationSymmetryX2D):
    ...


class RotationSymmetry2D(Macro2D):
    ...


class ScaleSymmetry2D(Macro2D):
    ...


class AxialScaleSymmetry2D(ScaleSymmetry2D):
    ...


class Dilate2D(SDFModifier2D):
    ...


class Erode2D(SDFModifier2D):
    ...


class Onion2D(SDFModifier2D):
    ...
