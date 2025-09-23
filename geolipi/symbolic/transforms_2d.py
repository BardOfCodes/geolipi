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
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "offset": {"type": "Vector[2]"}
        }

@register_symbol
class EulerRotate2D(Transform2D):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "angle": {"type": "float"}
        }

@register_symbol
class Scale2D(Transform2D):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "scale": {"type": "Vector[2]"}
        }

@register_symbol
class Shear2D(Transform2D):
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "shear": {"type": "Vector[2]"}
        }

@register_symbol
class Affine2D(Transform2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_matrix_2D
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "matrix": {"type": "Matrix[3,3]"}
        }

@register_symbol
class Distort2D(PositionalTransform2D):
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
class ReflectCoords2D(Transform2D):
    """
    Simply reflects the coordinates about the origin, w.r.t. the normal vector specified by param.
    This class is mapped to the following evaluator function(s):
    - torch_compute.transforms.get_affine_reflection_2D
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {
            "expr": {"type": "Expr"},
            "normal": {"type": "Vector[2]"}
        }

@register_symbol
class Reflect2D(Macro2D):
    """
    Performs union of canvas and its reflection about the origin,
    with the reflection plane's normal vector specified by param.
    """
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "normal": {"type": "Vector[2]"}}

@register_symbol
class ReflectX2D(Reflect2D):
    """Macro for reflecting about X axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}}

@register_symbol
class ReflectY2D(ReflectX2D):
    """Macro for reflecting about Y axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}}

@register_symbol
class AxialReflect2D(Macro2D):
    """Macro for reflecting about a specified axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "axis": {"type": "Enum[\"AX2D\"|\"AY2D\"]"}}

@register_symbol
class TranslationSymmetry2D(Macro2D):
    """Performs union of canvas and its translation by param."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}}

@register_symbol
class AxialTranslationSymmetry2D(TranslationSymmetry2D):
    """Performs union of canvas and its translation by param along a specified axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}, "axis": {"type": "Enum[\"AX2D\"|\"AY2D\"]"}}

@register_symbol
class TranslationSymmetryX2D(TranslationSymmetry2D):
    """Performs union of canvas and its translation by param along X axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}}

@register_symbol
class TranslationSymmetryY2D(TranslationSymmetryX2D):
    """Performs union of canvas and its translation by param along Y axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}}

@register_symbol
class RotationSymmetry2D(Macro2D):
    """Performs union of canvas and its rotation by param."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "angle": {"type": "float"}, "count": {"type": "int"}}

@register_symbol
class ScaleSymmetry2D(Macro2D):
    """Performs union of canvas and its scaling by param."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}}

@register_symbol
class AxialScaleSymmetry2D(ScaleSymmetry2D):
    """Performs union of canvas and its scaling by param along a specified axis."""
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "distance": {"type": "float"}, "count": {"type": "int"}, "axis": {"type": "Enum[\"AX2D\"|\"AY2D\"]"}}

@register_symbol
class Dilate2D(SDFModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_dilate
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "k": {"type": "float"}}

@register_symbol
class Erode2D(SDFModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_erode
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "k": {"type": "float"}}

@register_symbol
class Onion2D(SDFModifier2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_operators.sdf_onion
    Read evaluator specific documentation for more.
    """
    @classmethod
    def default_spec(cls):
        return {"expr": {"type": "Expr"}, "k": {"type": "float"}}
