from typing import Tuple, List
from .base_symbolic import GLExpr, GLFunction
from .types import param_type_1D, param_type_2D, param_type_3D, sig_check


class Primitive2D(GLFunction):
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True


class Circle2D(Primitive2D):
    """
    Basic 2D Circle.
    Signature 1: Circle(radius: param_type_1D)
    """
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(radius: param_type_1D):
        return sig_check([(radius, param_type_1D)])


class Rectangle2D(Primitive2D):
    """
    Basic 2D Rectangle.
    Signature 1: Rectangle(size_a: param_type_1D, size_b: param_type_1D)
    Signature 2: Rectangle(size: param_type_2D)
    """
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        elif cls._signature_2(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(size_x: param_type_1D, size_y: param_type_1D):
        return sig_check([(size_x, param_type_1D), (size_y, param_type_1D)])

    @staticmethod
    def _signature_2(size: param_type_2D):
        return sig_check([(size, param_type_2D)])


class Triangle2D(Primitive2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        elif cls._signature_2(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(point_a: param_type_2D = None, point_b: param_type_2D = None, point_c: param_type_2D = None):
        return sig_check([(point_a, param_type_2D), (point_b, param_type_2D), (point_c, param_type_2D)])
    @staticmethod
    def _signature_2(params: param_type_3D = None):
        return True



class TriangleIsosceles2D(Primitive2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")
    # shorthand to make a equilateral triangle

    @staticmethod
    def _signature_1(size_x: param_type_1D, size_y: param_type_1D):
        return sig_check([(size_x, param_type_1D), (size_y, param_type_1D)])


class TriangleEquilateral2D(Primitive2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")
    # shorthand to make a equilateral triangle

    @staticmethod
    def _signature_1(size: param_type_1D):
        return sig_check([(size, param_type_1D)])


class NoParamCircle2D(Circle2D):
    @classmethod
    def eval(cls,):
        return None


class NoParamRectangle2D(Rectangle2D):
    @classmethod
    def eval(cls,):
        return None


class NoParamTriangle2D(Rectangle2D):
    @classmethod
    def eval(cls,):
        return None
