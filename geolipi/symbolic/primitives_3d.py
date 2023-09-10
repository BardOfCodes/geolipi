from typing import Tuple, List, Union
from .base_symbolic import GLExpr, GLFunction
from .common import param_type_1D, param_type_2D, param_type_3D, sig_check


class Primitive3D(GLFunction):
    """Functions for declaring 3D primitives."""
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True


class Sphere3D(Primitive3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(radius: param_type_1D):
        return sig_check([(radius, param_type_1D)])


class Cuboid3D(Primitive3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        elif cls._signature_2(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(size_x: param_type_1D, size_y: param_type_1D, size_z: param_type_1D):
        return sig_check([(size_x, param_type_1D), (size_y, param_type_1D), (size_z, param_type_1D)])

    @staticmethod
    def _signature_2(size: param_type_3D):
        return sig_check([(size, param_type_3D)])


class Cylinder3D(Primitive3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        elif cls._signature_2(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(radius: param_type_1D, height: param_type_1D):
        return sig_check([(radius, param_type_1D), (height, param_type_1D)])

    @staticmethod
    def _signature_2(size: param_type_2D):
        return sig_check([(size, param_type_2D)])


class NoParamSphere3D(Primitive3D):
    @classmethod
    def eval(cls,):
        return None


class NoParamCuboid3D(Primitive3D):
    @classmethod
    def eval(cls,):
        return None


class NoParamCylinder3D(Primitive3D):
    @classmethod
    def eval(cls,):
        return None

# TODO: Add others from README.md

# TODO: CSGStump


class Cone3D:
    ...


class InfiniteCylinder3D:
    ...


class InfiniteCone3D:
    ...

# TODO: BSP + SQ


class HalfPlane:
    ...


class SuperQuadric:
    ...

# TODO: Others(IQ)


class Torus:
    ...


class Link:
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
