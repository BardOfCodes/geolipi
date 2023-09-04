# Revolution of 2D
from typing import Tuple, List
from .base_symbolic import GLExpr, GLFunction
from .types import param_type_1D, param_type_2D, param_type_3D, sig_check
from .primitives_2d import Primitive2D
from .primitives_3d import Primitive3D
""" 3D Curves will be used for Curve Extrude Primitives. 2D Curves can be used for SVG creation."""
""" Combining multiple 2D Curves into a single closed curve will be used for 3D Extrude Primitives. ExtrudeNet principle"""

# Extrusion of 2D

class Curve3D():
    ...


class StraightLine(Curve3D):
    ...


class QuadraticBezier(Curve3D):
    ...


class CubicBezier(Curve3D):
    ...

class Curve2D(Primitive2D):
    ...
class GeneralizedCylinder():
    ...


class RevolutedPrimitve():
    ...
