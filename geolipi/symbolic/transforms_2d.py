from .base_symbolic import GLFunction


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


class Translate2D(Transform2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_translate_2D
    Read evaluator specific documentation for more.
    """


class EulerRotate2D(Transform2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_rotate_2D
    Read evaluator specific documentation for more.
    """


class Scale2D(Transform2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_scale_2D
    Read evaluator specific documentation for more.
    """


class Shear2D(Transform2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_shear_2D
    Read evaluator specific documentation for more.
    """


class Distort2D(PositionalTransform2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.position_distort
    Read evaluator specific documentation for more.
    """


class ReflectCoords2D(Transform2D):
    """
    Simply reflects the coordinates about the origin, w.r.t. the normal vector specified by param.
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_reflection_2D
    Read evaluator specific documentation for more.
    """


class Reflect2D(Macro2D):
    """
    Performs union of canvas and its reflection about the origin,
    with the reflection plane's normal vector specified by param.
    """


class ReflectX2D(Reflect2D):
    """Macro for reflecting about X axis."""


class ReflectY2D(ReflectX2D):
    """Macro for reflecting about Y axis."""


class AxialReflect2D(Macro2D):
    """Macro for reflecting about a specified axis."""


class TranslationSymmetry2D(Macro2D):
    """Performs union of canvas and its translation by param."""


class AxialTranslationSymmetry2D(TranslationSymmetry2D):
    """Performs union of canvas and its translation by param along a specified axis."""


class TranslationSymmetryX2D(TranslationSymmetry2D):
    """Performs union of canvas and its translation by param along X axis."""


class TranslationSymmetryY2D(TranslationSymmetryX2D):
    """Performs union of canvas and its translation by param along Y axis."""


class RotationSymmetry2D(Macro2D):
    """Performs union of canvas and its rotation by param."""


class ScaleSymmetry2D(Macro2D):
    """Performs union of canvas and its scaling by param."""


class AxialScaleSymmetry2D(ScaleSymmetry2D):
    """Performs union of canvas and its scaling by param along a specified axis."""


class Dilate2D(SDFModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_dilate
    Read evaluator specific documentation for more.
    """


class Erode2D(SDFModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_erode
    Read evaluator specific documentation for more.
    """


class Onion2D(SDFModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_onion
    Read evaluator specific documentation for more.
    """
