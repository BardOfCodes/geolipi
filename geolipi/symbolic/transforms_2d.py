from .base import GLFunction
from .registry import register_symbol

class Modifier2D(GLFunction):
    """Base Class for all 2D modifiers."""

class Transform2D(Modifier2D):
    """Base Class for all 2D transforms."""

class PositionalTransform2D(Modifier2D):
    """Base Class for all 2D positional transforms."""

class Macro2D(Modifier2D):
    """Base Class for all 2D macros."""


class SDFModifier2D(Modifier2D):
    """Base Class for all 2D SDF modifiers."""

@register_symbol
class Translate2D(Transform2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_translate_2D
    Read evaluator specific documentation for more.
    """

@register_symbol
class EulerRotate2D(Transform2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_rotate_2D
    Read evaluator specific documentation for more.
    """

@register_symbol
class Scale2D(Transform2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_scale_2D
    Read evaluator specific documentation for more.
    """

@register_symbol
class Shear2D(Transform2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_shear_2D
    Read evaluator specific documentation for more.
    """

@register_symbol
class Affine2D(Transform2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_matrix_2D
    Read evaluator specific documentation for more.
    """

@register_symbol
class Distort2D(PositionalTransform2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.position_distort
    Read evaluator specific documentation for more.
    """

@register_symbol
class ReflectCoords2D(Transform2D):
    """
    Simply reflects the coordinates about the origin, w.r.t. the normal vector specified by param.
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_reflection_2D
    Read evaluator specific documentation for more.
    """

@register_symbol
class Reflect2D(Macro2D):
    """
    Performs union of canvas and its reflection about the origin,
    with the reflection plane's normal vector specified by param.
    """

@register_symbol
class ReflectX2D(Reflect2D):
    """Macro for reflecting about X axis."""

@register_symbol
class ReflectY2D(ReflectX2D):
    """Macro for reflecting about Y axis."""

@register_symbol
class AxialReflect2D(Macro2D):
    """Macro for reflecting about a specified axis."""

@register_symbol
class TranslationSymmetry2D(Macro2D):
    """Performs union of canvas and its translation by param."""

@register_symbol
class AxialTranslationSymmetry2D(TranslationSymmetry2D):
    """Performs union of canvas and its translation by param along a specified axis."""

@register_symbol
class TranslationSymmetryX2D(TranslationSymmetry2D):
    """Performs union of canvas and its translation by param along X axis."""

@register_symbol
class TranslationSymmetryY2D(TranslationSymmetryX2D):
    """Performs union of canvas and its translation by param along Y axis."""

@register_symbol
class RotationSymmetry2D(Macro2D):
    """Performs union of canvas and its rotation by param."""

@register_symbol
class ScaleSymmetry2D(Macro2D):
    """Performs union of canvas and its scaling by param."""

@register_symbol
class AxialScaleSymmetry2D(ScaleSymmetry2D):
    """Performs union of canvas and its scaling by param along a specified axis."""

@register_symbol
class Dilate2D(SDFModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_dilate
    Read evaluator specific documentation for more.
    """

@register_symbol
class Erode2D(SDFModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_erode
    Read evaluator specific documentation for more.
    """

@register_symbol
class Onion2D(SDFModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_onion
    Read evaluator specific documentation for more.
    """
