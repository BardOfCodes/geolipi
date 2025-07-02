from .base import GLFunction


class Modifier3D(GLFunction):
    """Base Class for all 3D modifiers."""


class Transform3D(Modifier3D):
    """Base Class for all 3D transforms."""


class PositionalTransform3D(Modifier3D):
    """Base Class for all 3D positional transforms."""


class Macro3D(Modifier3D):
    """Base Class for all 3D macros."""


class SDFModifier3D(Modifier3D):
    """Base Class for all 3D SDF modifiers."""


class Translate3D(Transform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_translate_3D
    Read evaluator specific documentation for more.
    """


class EulerRotate3D(Transform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_rotate_euler_3D
    Read evaluator specific documentation for more.
    """

class AxisAngleRotate3D(Transform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_rotate_axis_angle_3D
    Read evaluator specific documentation for more.
    """

class RotateMatrix3D(Transform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_rotate_matrix_3D
    Read evaluator specific documentation for more.
    """

class Scale3D(Transform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_scale_3D
    Read evaluator specific documentation for more.
    """


class QuaternionRotate3D(Transform3D):
    """
    This class is mapped to the following evaluator function(s):
    #TODO: Implement this
    Read evaluator specific documentation for more.
    """


class Rotate5D(Transform3D):
    """
    # For Continuous optimization of rotation
    # Higher order for differentiable optimization of rotation
    # Ref: https://arxiv.org/abs/1812.07035
    This class is mapped to the following evaluator function(s):
    #TODO: Implement this
    Read evaluator specific documentation for more.
    """


class Rotate6D(Transform3D):
    """
    # For Continuous optimization of rotation
    # Higher order for differentiable optimization of rotation
    # Ref: https://arxiv.org/abs/1812.07035
    This class is mapped to the following evaluator function(s):
    #TODO: Implement this
    Read evaluator specific documentation for more.
    """


class Rotate9D(Transform3D):
    """
    # For Continuous optimization of rotation
    # Higher order for differentiable optimization of rotation
    # Ref: https://arxiv.org/abs/1812.07035
    This class is mapped to the following evaluator function(s):
    #TODO: Implement this
    Read evaluator specific documentation for more.
    """


class Shear3D(Transform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_shear_3D
    Read evaluator specific documentation for more.
    """


class Distort3D(PositionalTransform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.position_distort
    Read evaluator specific documentation for more.
    """


class Twist3D(PositionalTransform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.position_twist
    Read evaluator specific documentation for more.
    """


class Bend3D(PositionalTransform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.position_bend
    Read evaluator specific documentation for more.
    """


class ReflectCoords3D(Transform3D):
    """
    Simply reflects the coordinates about the origin, w.r.t. the normal vector specified by param.
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_reflection_3D
    Read evaluator specific documentation for more.
    """


class Reflect3D(Macro3D):
    """
    Performs union of canvas and its reflection about the origin,
    with the reflection plane's normal vector specified by param.
    """


class ReflectX3D(Reflect3D):
    """Macro for reflecting about X axis."""


class ReflectY3D(ReflectX3D):
    """Macro for reflecting about Y axis."""


class ReflectZ3D(ReflectY3D):
    """Macro for reflecting about Z axis."""


class AxialReflect3D(Reflect3D):
    """Macro for reflecting about a specified axis."""


class TranslationSymmetry3D(Macro3D):
    """Performs union of canvas and its translation by param."""


class AxialTranslationSymmetry3D(TranslationSymmetry3D):
    """Performs union of canvas and its translation by param along a specified axis."""


class TranslationSymmetryX3D(TranslationSymmetry3D):
    """Performs union of canvas and its translation by param along X axis."""


class TranslationSymmetryY3D(TranslationSymmetryX3D):
    """Performs union of canvas and its translation by param along Y axis."""


class TranslationSymmetryZ3D(TranslationSymmetryY3D):
    """Performs union of canvas and its translation by param along Z axis."""


class RotationSymmetry3D(Macro3D):
    """Performs union of canvas and its rotation by param."""


class AxialRotationSymmetry3D(RotationSymmetry3D):
    """Performs union of canvas and its rotation by param around a specified axis."""


class RotationSymmetryX3D(RotationSymmetry3D):
    """Performs union of canvas and its rotation by param around X axis."""


class RotationSymmetryY3D(RotationSymmetryX3D):
    """Performs union of canvas and its rotation by param around Y axis."""


class RotationSymmetryZ3D(RotationSymmetryY3D):
    """Performs union of canvas and its rotation by param around Z axis."""


class ScaleSymmetry3D(Macro3D):
    """Performs union of canvas and its scaling by param."""


class AxialScaleSymmetry3D(ScaleSymmetry3D):
    """Performs union of canvas and its scaling by param along a specified axis."""


class Dilate3D(SDFModifier3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_dilate
    Read evaluator specific documentation for more.
    """


class Erode3D(SDFModifier3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_erode
    Read evaluator specific documentation for more.
    """


class Onion3D(SDFModifier3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_onion
    Read evaluator specific documentation for more.
    """
