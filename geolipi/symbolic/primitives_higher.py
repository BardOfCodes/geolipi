from .base import GLFunction
from .registry import register_symbol

class HigherOrderPrimitives3D(GLFunction):
    """Functions for declaring higher order Primitives."""

@register_symbol
class Revolution3D(HigherOrderPrimitives3D):
    pass


@register_symbol
class CurvePrimitive3D(HigherOrderPrimitives3D):
    """Base class for curve extrusion primitives."""


@register_symbol
class SimpleExtrusion3D(CurvePrimitive3D):

    pass


@register_symbol
class LinearExtrude3D(CurvePrimitive3D):
    pass


@register_symbol
class QuadraticBezierExtrude3D(CurvePrimitive3D):
    pass


@register_symbol
class PolyQuadBezierExtrude3D(CurvePrimitive3D):
    pass


@register_symbol
class CubicBezierExtrude3D(CurvePrimitive3D):
    """
    # Ref: https://www.shadertoy.com/view/4sKyzW
    # https://www.shadertoy.com/view/llyXDV
    This class is mapped to the following evaluator function(s):
    # TODO: Implement this
    Read evaluator specific documentation for more.
    """


class Primitive1D(GLFunction):
    pass

@register_symbol
class LinearCurve1D(Primitive1D):
    pass

@register_symbol
class QuadraticCurve1D(Primitive1D):
    pass

@register_symbol
class PolyStraightLineCurve1D(Primitive1D):
    pass
