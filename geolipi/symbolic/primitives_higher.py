from .base import GLFunction


class HigherOrderPrimitives3D(GLFunction):
    """Functions for declaring higher order Primitives."""


class Revolution3D(HigherOrderPrimitives3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_functions_higher.sdf3d_revolution
    Read evaluator specific documentation for more.
    """


class CurvePrimitive3D(HigherOrderPrimitives3D):

    """
    Base class for Curve Extrusion based 3D primitives.
    """


class SimpleExtrusion3D(CurvePrimitive3D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_functions_higher.sdf3d_simple_extrusion
    Read evaluator specific documentation for more.
    """


class LinearExtrude3D(CurvePrimitive3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_functions_higher.sdf3d_linear_extrude
    Read evaluator specific documentation for more.
    """


class QuadraticBezierExtrude3D(CurvePrimitive3D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.sdf_functions_higher.sdf3d_quadratic_bezier_extrude
    Read evaluator specific documentation for more.
    """


class PolyQuadBezierExtrude3D(CurvePrimitive3D):
    """
    This class is mapped to the following evaluator function(s):
    # TODO: Implement this
    Read evaluator specific documentation for more.
    """


class CubicBezierExtrude3D(CurvePrimitive3D):
    """
    # Ref: https://www.shadertoy.com/view/4sKyzW
    # https://www.shadertoy.com/view/llyXDV
    This class is mapped to the following evaluator function(s):
    # TODO: Implement this
    Read evaluator specific documentation for more.
    """


class Primitive1D(GLFunction):
    """
    This class is mapped to the following evaluator function(s):
    # TODO: Implement this
    Read evaluator specific documentation for more.
    """


class LinearCurve1D(Primitive1D):
    """
    This class is mapped to the following evaluator function(s):
    # TODO: Implement this
    Read evaluator specific documentation for more.
    """


class QuadraticCurve1D(Primitive1D):
    """
    This class is mapped to the following evaluator function(s):
    # TODO: Implement this
    Read evaluator specific documentation for more.
    """


class PolyStraightLineCurve1D(Primitive1D):
    """
    This class is mapped to the following evaluator function(s):
    # TODO: Implement this
    Read evaluator specific documentation for more.
    """
