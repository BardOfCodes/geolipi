from typing import Tuple, List, Union
from sympy import Function, Expr
from .common import param_type

class Primitive3D(Function):
    """Functions for declaring 3D primitives."""
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True

class Sphere(Primitive3D):
    @classmethod
    def eval(cls, radius: param_type = 1.0):
        return None

class Cuboid(Primitive3D):
    @classmethod
    def eval(cls, size_x: param_type = 1.0, size_y: param_type = 1.0, size_z: param_type = 1.0):
        return None
    
class Cylinder(Primitive3D):
    @classmethod
    def eval(cls, radius: param_type = 1.0, height: param_type = 1.0):
        return None

class Cone(Primitive3D):
    @classmethod
    def eval(cls, radius_top: param_type = 1.0, param_type: float = 1.0, height: param_type = 1.0):
        return None


class NoParamSphere(Sphere):
    @classmethod
    def eval(cls,):
        return None

class NoParamCuboid(Sphere):
    @classmethod
    def eval(cls,):
        return None

class NoParamCylinder(Sphere):
    @classmethod
    def eval(cls,):
        return None
# TODO: Add others from README.md

class Torus:
    ...

class Link:
    ...

class InfiniteCylinder:
    ...

class InfiniteCone:
    ...

class HalfPlane:
    ...

class HexagonalPrism:
    ...

class TriangularPrism:
    ...

class Capsule:
    ...
class ellipsoid:
    ...
    
class solidAngle:
    ...

class RevolvedVesica:
    ...

class Rhombus:
    ...

class Octahedron:
    ...

class Pyramid:
    ...

class Triangle:
    ...

class Quadrilateral:
    ...
