from typing import Tuple, List, Union
from sympy import Function


class Primitive(Function):
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True


class Cuboid(Primitive):
    @classmethod
    def eval(cls, size_x: float = 1.0, size_y: float = 1.0, size_z: float = 1.0):
        return None

class Sphere(Primitive):
    @classmethod
    def eval(cls, radius: float = 1.0):
        return None

class Cylinder(Primitive):
    @classmethod
    def eval(cls, radius: float = 1.0, height: float = 1.0):
        return None


class Cone(Primitive):
    @classmethod
    def eval(cls, radius_top: float = 1.0, radius_bottom: float = 1.0, height: float = 1.0):
        return None

# TODO: Add others from README.md
