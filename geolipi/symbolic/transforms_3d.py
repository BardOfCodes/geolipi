from .base import GLFunction
from .registry import register_symbol

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

@register_symbol
class Affine3D(Transform3D):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "matrix": {"type": "Matrix[4,4]"}
        }


@register_symbol
class Translate3D(Transform3D):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "offset": {"type": "Vector[3]"}
        }

@register_symbol
class EulerRotate3D(Transform3D):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "angles": {"type": "Vector[3]"}
        }

@register_symbol
class AxisAngleRotate3D(Transform3D):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "axis": {"type": "Vector[3]"},
            "angle": {"type": "float"}
        }

@register_symbol
class RotateMatrix3D(Transform3D):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "matrix": {"type": "Matrix[3,3]"}
        }

@register_symbol
class Scale3D(Transform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_scale_3D
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "scale": {"type": "Vector[3]"}
        }

@register_symbol
class QuaternionRotate3D(Transform3D):
    """
    This class is mapped to the following evaluator function(s):
    #TODO: Implement this
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "quat": {"type": "Vector[4]"}
        }

@register_symbol
class Rotate5D(Transform3D):
    """
    # For Continuous optimization of rotation
    # Higher order for differentiable optimization of rotation
    # Ref: https://arxiv.org/abs/1812.07035
    This class is mapped to the following evaluator function(s):
    #TODO: Implement this
    Read evaluator specific documentation for more.
    """

@register_symbol
class Rotate6D(Transform3D):
    """
    # For Continuous optimization of rotation
    # Higher order for differentiable optimization of rotation
    # Ref: https://arxiv.org/abs/1812.07035
    This class is mapped to the following evaluator function(s):
    #TODO: Implement this
    Read evaluator specific documentation for more.
    """

@register_symbol
class Rotate9D(Transform3D):
    """
    # For Continuous optimization of rotation
    # Higher order for differentiable optimization of rotation
    # Ref: https://arxiv.org/abs/1812.07035
    This class is mapped to the following evaluator function(s):
    #TODO: Implement this
    Read evaluator specific documentation for more.
    """

@register_symbol
class Shear3D(Transform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_shear_3D
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "shear": {"type": "Vector[6]"}
        }

@register_symbol
class Distort3D(PositionalTransform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.position_distort
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "amount": {"type": "float"}
        }

@register_symbol
class Twist3D(PositionalTransform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.position_twist
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "amount": {"type": "float"}
        }

@register_symbol
class Bend3D(PositionalTransform3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.position_bend
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "amount": {"type": "float"}
        }

@register_symbol
class ReflectCoords3D(Transform3D):
    """
    Simply reflects the coordinates about the origin, w.r.t. the normal vector specified by param.
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_reflection_3D
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "normal": {"type": "Vector[3]"}
        }

@register_symbol
class Reflect3D(Macro3D):
    """
    Performs union of canvas and its reflection about the origin,
    with the reflection plane's normal vector specified by param.
    """
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "normal": {"type": "Vector[3]"}}

@register_symbol
class ReflectX3D(Reflect3D):
    """Macro for reflecting about X axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}}

@register_symbol
class ReflectY3D(ReflectX3D):
    """Macro for reflecting about Y axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}}

@register_symbol
class ReflectZ3D(ReflectY3D):
    """Macro for reflecting about Z axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}}

@register_symbol
class AxialReflect3D(Reflect3D):
    """Macro for reflecting about a specified axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "axis": {"type": "Enum[\"AX3D\"|\"AY3D\"|\"AZ3D\"]"}}

@register_symbol
class TranslationSymmetry3D(Macro3D):
    """Performs union of canvas and its translation by param."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}}

@register_symbol
class AxialTranslationSymmetry3D(TranslationSymmetry3D):
    """Performs union of canvas and its translation by param along a specified axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}, "axis": {"type": "Enum[\"AX3D\"|\"AY3D\"|\"AZ3D\"]"}}

@register_symbol
class TranslationSymmetryX3D(TranslationSymmetry3D):
    """Performs union of canvas and its translation by param along X axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}}

@register_symbol
class TranslationSymmetryY3D(TranslationSymmetryX3D):
    """Performs union of canvas and its translation by param along Y axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}}

@register_symbol
class TranslationSymmetryZ3D(TranslationSymmetryY3D):
    """Performs union of canvas and its translation by param along Z axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}}

@register_symbol
class RotationSymmetry3D(Macro3D):
    """Performs union of canvas and its rotation by param."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "angle": {"type": "float"}, "count": {"type": "int"}}

@register_symbol
class AxialRotationSymmetry3D(RotationSymmetry3D):
    """Performs union of canvas and its rotation by param around a specified axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "angle": {"type": "float"}, "count": {"type": "int"}, "axis": {"type": "Enum[\"AX3D\"|\"AY3D\"|\"AZ3D\"]"}}

@register_symbol
class RotationSymmetryX3D(RotationSymmetry3D):
    """Performs union of canvas and its rotation by param around X axis."""

@register_symbol
class RotationSymmetryY3D(RotationSymmetryX3D):
    """Performs union of canvas and its rotation by param around Y axis."""

@register_symbol
class RotationSymmetryZ3D(RotationSymmetryY3D):
    """Performs union of canvas and its rotation by param around Z axis."""

@register_symbol
class ScaleSymmetry3D(Macro3D):
    """Performs union of canvas and its scaling by param."""

@register_symbol
class AxialScaleSymmetry3D(ScaleSymmetry3D):
    """Performs union of canvas and its scaling by param along a specified axis."""

@register_symbol
class Dilate3D(SDFModifier3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_dilate
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "k": {"type": "float"}}

@register_symbol
class Erode3D(SDFModifier3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_erode
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "k": {"type": "float"}}

@register_symbol
class Onion3D(SDFModifier3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_onion
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "k": {"type": "float"}}


@register_symbol
class NegOnlyOnion3D(SDFModifier3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_onion
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "k": {"type": "float"}}
