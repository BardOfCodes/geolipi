from typing import Tuple, List

from .base_symbolic import GLExpr, GLFunction
from .primitives_2d import Primitive2D
from .primitives_3d import Primitive3D

""" 3D Curves will be used for Curve Extrude Primitives. 
    2D Curves can be used for SVG creation."""


class HigherOrderPrimitives3D(GLFunction):
    """Functions for declaring higher order Primitives."""
    ...
# Extrusion of 2D: Used for GeoCode.


class CurvePrimitive3D(HigherOrderPrimitives3D):
    """Functions for declaring 3D curve primitives."""
    ...


class StraightExtrude3D(CurvePrimitive3D):
    ...


class QuadraticBezierExtrude3D(CurvePrimitive3D):
    ...


class PolyQuadBezierExtrude3D(CurvePrimitive3D):
    ...


class CubicBezierExtrude3D(CurvePrimitive3D):
    ...
    # Ref: https://www.shadertoy.com/view/4sKyzW
    # https://www.shadertoy.com/view/llyXDV

# Used for defining the scale along curve.


class Primitive1D(GLFunction):
    """Functions for declaring 1D primitives."""
    ...


class StraightLineCurve1D(Primitive1D):
    # y = mx + c
    ...


class QuadraticCurve1D(Primitive1D):
    # y = ax2 + bx + c
    ...


class PolyStraightLineCurve1D(Primitive1D):
    # set of (x, y) points smoothly connected by cubic bezier curves.
    ...
