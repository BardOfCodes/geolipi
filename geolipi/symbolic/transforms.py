from typing import Tuple, List, Union
from sympy import Function
from .primitives import Primitive

canvas_type = Union[Function, Primitive]

class Transform(Function):
    """Transform will affect the 3D cooridinates used as input when evaluating SDFs. 
    On blender graphs, they will be mesh transformation nodes.
    """
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True

class Translate(Transform):
    @classmethod
    def eval(cls, canvas: canvas_type, param: Tuple[float, ...]):
        return None


class EulerRotate(Transform):
    @classmethod
    def eval(cls, canvas: canvas_type, param: Tuple[float, ...]):
        return None


class QuaternionRotate(Transform):
    @classmethod
    def eval(cls, canvas: canvas_type, param: Tuple[float, ...]):
        return None


class Scale(Transform):
    """Perform uniform scaling if there is only one variable.
    """
    true_sdf = False

    @classmethod
    def eval(cls, canvas: canvas_type, param: Tuple[float, ...]):
        return None

# helper macros


class TranslateX(Translate):
    @classmethod
    def eval(cls, canvas: canvas_type, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Translate(canvas, (param, 0, 0))


class TranslateY(Translate):
    @classmethod
    def eval(cls, canvas: canvas_type, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Translate(canvas, (0, param, 0))


class TranslateZ(Translate):
    @classmethod
    def eval(cls, canvas: canvas_type, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Translate(canvas, (0, 0, param))


class EulerRotateX(EulerRotate):
    @classmethod
    def eval(cls, canvas: canvas_type, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return EulerRotate(canvas, (param, 0, 0))


class EulerRotateY(EulerRotate):
    @classmethod
    def eval(cls, canvas: canvas_type, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return EulerRotate(canvas, (0, param, 0))


class EulerRotateZ(EulerRotate):
    @classmethod
    def eval(cls, canvas: canvas_type, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return EulerRotate(canvas, (0, 0, param))


class ScaleX(Scale):
    @classmethod
    def eval(cls, canvas: canvas_type, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Scale(canvas, (param, 1, 1))


class ScaleY(Scale):
    @classmethod
    def eval(cls, canvas: canvas_type, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Scale(canvas, (1, param, 1))


class ScaleZ(Scale):
    @classmethod
    def eval(cls, canvas: canvas_type, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Scale(canvas, (1, 1, param))


class ReflectTransform(Transform):

    @classmethod
    def eval(cls, canvas: canvas_type, param: float):
        return None
# Macros