from typing import Tuple, List, Union
from .base_symbolic import GLExpr, GLFunction
from .common import param_type_1D, param_type_2D, param_type_3D, sig_check
from sympy import Tuple as SympyTuple

class Primitive3D(GLFunction):
    """Functions for declaring 3D primitives."""
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True

    
    def pretty_print(self, tabs=0, tab_str="\t"):
        args = self.args
        n_tabs = tab_str * tabs
        replaced_args = [self.lookup_table.get(arg, arg) for arg in args]
        str_args = []
        for arg in replaced_args:
            if isinstance(arg, (GLExpr, GLFunction)):
                str_args.append(arg.pretty_print(tabs=tabs+1, tab_str=tab_str))
            else:
                if isinstance(arg, SympyTuple):
                    item = [f"{x:.3f}" for x in arg]
                    item = ', '.join(item)
                    str_args.append(f"({item})")
                else:
                    str_args.append(str(arg))
        if str_args:
            n_tabs_1 = tab_str * (tabs + 1)
            # str_args = [""] + str_args
            str_args = f", ".join(str_args)
            final = f"{self.func.__name__}({str_args})"
        else:
            final = f"{self.func.__name__}()"
        return final


class OpenPrimitive3D(Primitive3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        return None

class Sphere3D(Primitive3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        elif cls._signature_2(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(radius: param_type_1D, *args, **kwargs):
        return sig_check([(radius, param_type_1D)])
    # User for Primal CSG
    @staticmethod
    def _signature_2(size: param_type_3D, position: param_type_3D, *args, **kwargs):
        return sig_check([(size, param_type_3D), (position, param_type_3D)])


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
    def _signature_1(size_x: param_type_1D=None, size_y: param_type_1D=None, size_z: param_type_1D=None):
        return sig_check([(size_x, param_type_1D), (size_y, param_type_1D), (size_z, param_type_1D)])

    # User for Primal CSG
    @staticmethod
    def _signature_2(size: param_type_3D, position: param_type_3D):
        return sig_check([(size, param_type_3D), (position, param_type_3D)])


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
