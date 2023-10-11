from typing import Tuple, List

import sympy
from .base_symbolic import GLExpr, GLFunction
from .common import expr_type, param_type_1D, param_type_2D, param_type_3D, param_type_4D, sig_check


class Modifier3D(GLFunction):
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True


class Transform3D(Modifier3D):
    ...


class Translate3D(Transform3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, param: param_type_3D):
        return sig_check([(expr, expr_type), (param, param_type_3D)])


class EulerRotate3D(Transform3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, param: param_type_3D):
        return sig_check([(expr, expr_type), (param, param_type_3D)])


class Scale3D(Transform3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, param: param_type_3D):
        return sig_check([(expr, expr_type), (param, param_type_3D)])


class QuaternionRotate3D(Transform3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, param: param_type_4D):
        return sig_check([(expr, expr_type), (param, param_type_4D)])

# For Continuous optimization of rotation


# TODO: Higher order for differentiable optimization of rotation
# Ref: https://arxiv.org/abs/1812.07035


class Rotate5D(Transform3D):
    ...


class Rotate6D(Transform3D):
    ...


class Rotate9D(Transform3D):
    ...


# helper macros
def TranslateX3D(expr: expr_type, param: param_type_1D):
    ...


def TranslateY3D(expr: expr_type, param: param_type_1D):
    ...


def TranslateZ3D(expr: expr_type, param: param_type_1D):
    ...


# TODO: Implementation
class Reflect3D(Modifier3D):
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
    def _signature_1(expr: expr_type, param: param_type_3D):
        return sig_check([(expr, expr_type), (param, param_type_3D)])


class ReflectX3D(Reflect3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type):
        return sig_check([(expr, expr_type)])


class ReflectY3D(ReflectX3D):
    ...


class ReflectZ3D(ReflectX3D):
    ...


class ReflectCoords3D(Transform3D):
    """Simply reflects the coordinates about the origin, w.r.t. the normal vector specified by param."""
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, param: param_type_3D):
        return sig_check([(expr, expr_type), (param, param_type_3D)])


class AxialReflect3D(Modifier3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, axis: sympy.Symbol):
        return sig_check([(expr, expr_type), (axis, sympy.Symbol)])


class TranslationSymmetry3D(Transform3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, translate_delta: param_type_3D, n_count: param_type_1D):
        return sig_check([(expr, expr_type), (translate_delta, param_type_3D), (n_count, param_type_1D)])
    

class AxialTranslationSymmetry3D(TranslationSymmetry3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, axis: sympy.Symbol, translate_delta: param_type_1D, n_count: param_type_1D):
        return sig_check([(expr, expr_type), (axis, sympy.Symbol), (translate_delta, param_type_1D), (n_count, param_type_1D)])


class TranslationSymmetryX3D(TranslationSymmetry3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type):
        return sig_check([(expr, expr_type)])


class TranslationSymmetryY3D(TranslationSymmetryX3D):
    ...

class TranslationSymmetryZ3D(TranslationSymmetryX3D):
    ...
    
    
class RotationSymmetry3D(Transform3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, rotate_delta: param_type_3D, n_count: param_type_1D):
        return sig_check([(expr, expr_type), (rotate_delta, param_type_3D), (n_count, param_type_1D)])
        
class AxialRotationSymmetry3D(RotationSymmetry3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, axis: sympy.Symbol, translate_delta: param_type_1D, n_count: param_type_1D):
        return sig_check([(expr, expr_type), (axis, sympy.Symbol), (translate_delta, param_type_1D), (n_count, param_type_1D)])
    
class RotationSymmetryX3D(RotationSymmetry3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type):
        return sig_check([(expr, expr_type)])


class RotationSymmetryY3D(RotationSymmetryX3D):
    ...

class RotationSymmetryZ3D(RotationSymmetryX3D):
    ...
        
class ScaleSymmetry3D(Transform3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, size_delta: param_type_3D, n_count: param_type_1D):
        return sig_check([(expr, expr_type), (size_delta, param_type_1D), (n_count, param_type_1D)])
        
class AxialScaleSymmetry3D(ScaleSymmetry3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, axis: sympy.Symbol, translate_delta: param_type_1D, n_count: param_type_1D):
        return sig_check([(expr, expr_type), (axis, sympy.Symbol), (translate_delta, param_type_1D), (n_count, param_type_1D)])
    

class ColorTree3D(Modifier3D):
    @classmethod
    def eval(cls, *args, **kwargs):
        if cls._signature_1(*args, **kwargs):
            return None
        else:
            raise TypeError("Invalid arguments for the function.")

    @staticmethod
    def _signature_1(expr: expr_type, color: param_type_1D):
        return sig_check([(expr, expr_type), (color, param_type_1D)])
    
