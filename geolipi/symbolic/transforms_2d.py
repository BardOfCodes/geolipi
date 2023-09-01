from typing import Tuple, List
from .base_symbolic import GLExpr, GLFunction
import sympy
from .types import expr_type, param_type_1D, param_type_2D, param_type_3D, sig_check


class Modifier2D(GLFunction):
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True


class Transform2D(Modifier2D):
    ...


class Translate2D(Transform2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        elif cls._signature_2(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, param: param_type_2D):
        return sig_check([(expr, expr_type), (param, param_type_2D)])

    @staticmethod
    def _signature_2(expr: expr_type, move_x: param_type_1D, move_y: param_type_1D):
        return sig_check([(expr, expr_type), (move_x, param_type_1D), (move_y, param_type_1D)])


class EulerRotate2D(Transform2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, param: param_type_1D):
        return sig_check([(expr, expr_type), (param, param_type_1D)])


class Scale2D(Transform2D):
    """Perform uniform scaling if there is only one variable.
    """
    true_sdf = False

    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        elif cls._signature_2(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, param: param_type_2D):
        return sig_check([(expr, expr_type), (param, param_type_2D)])

    @staticmethod
    def _signature_2(expr: expr_type, scale_x: param_type_1D, scale_y: param_type_1D):
        return sig_check([(expr, expr_type), (scale_x, param_type_1D), (scale_y, param_type_1D)])


# TODO: Implementation
class Reflect2D(Modifier2D):
    """Performs union of canvas and its reflection about the origin, 
    with the reflection plane's normal vector specified by param.
    """
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, param: param_type_2D):
        return sig_check([(expr, expr_type), (param, param_type_2D)])

class ReflectX2D(Reflect2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type):
        return sig_check([(expr, expr_type)])


class ReflectY2D(ReflectX2D):
    ...

class AxialReflect2D(Modifier2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, axis: sympy.Symbol):
        return sig_check([(expr, expr_type), (axis, sympy.Symbol)])

class ReflectCoords2D(Transform2D):
    """Simply reflects the coordinates about the origin, w.r.t. the normal vector specified by param."""
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, param: param_type_2D):
        return sig_check([(expr, expr_type), (param, param_type_2D)])

class TranslationSymmetry2D(Transform2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        elif cls._signature_2(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, translate_delta: param_type_2D, n_count: param_type_1D):
        return sig_check([(expr, expr_type), (translate_delta, param_type_3D), (n_count, param_type_1D)])
    @staticmethod
    def _signature_2(expr: expr_type, delta_x: param_type_1D, delta_y: param_type_1D, n_count: param_type_1D):
        return sig_check([(expr, expr_type), (delta_x, param_type_1D), (delta_y, param_type_1D), (n_count, param_type_1D)])
    
class RotationSymmetry2D(Transform2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, rotate_delta: param_type_1D, n_count: param_type_1D):
        return sig_check([(expr, expr_type), (rotate_delta, param_type_1D), (n_count, param_type_1D)])
        

class ScaleSymmetry2D(Transform2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, size_delta: param_type_2D, n_count: param_type_1D):
        return sig_check([(expr, expr_type), (size_delta, param_type_1D), (n_count, param_type_1D)])
        

class ColorTree2D(Modifier2D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, color: param_type_1D):
        return sig_check([(expr, expr_type), (color, param_type_1D)])
    
# ShapeAssembly functions
# TODO: Implementation

class Repeat(Modifier2D):
    ...


class Attach(Modifier2D):
    ...


class Squeeze(Modifier2D):
    ...


class Elongate(Modifier2D):
    ...


class Round(Modifier2D):
    ...


class Onion(Modifier2D):
    ...


class Twist(Modifier2D):
    ...
