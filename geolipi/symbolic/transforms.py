from typing import Tuple, List, Union
from sympy import Function, Expr
from .primitives_3d import Primitive3D
from .common import param_type

class Modifier(Function):
    """Transforms take expression inputs and return expression outputs.
    """
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True

class Transform(Modifier):
    """Transform will affect the 3D cooridinates used as input when evaluating SDFs. 
    On blender graphs, they will be mesh transformation nodes."""
    ...

class Translate(Transform):
    @classmethod
    def eval(cls, expression: Expr, param: param_type):
        return None

class EulerRotate(Transform):
    @classmethod
    def eval(cls, expression: Expr, param: param_type):
        return None

class QuaternionRotate(Transform):
    @classmethod
    def eval(cls, expression: Expr, param: param_type):
        return None

# For Continuous optimization of rotation 
class Rotate5D(Transform):
    ...
    
class Rotate6D(Transform):
    ...

class Rotate9D(Transform):
    ...

class Scale(Transform):
    """Perform uniform scaling if there is only one variable.
    """
    true_sdf = False

    @classmethod
    def eval(cls, expression: Expr, param: param_type):
        return None

# helper macros


class TranslateX(Translate):
    @classmethod
    def eval(cls, expression: Expr, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Translate(canvas, (param, 0, 0))


class TranslateY(Translate):
    @classmethod
    def eval(cls, expression: Expr, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Translate(canvas, (0, param, 0))


class TranslateZ(Translate):
    @classmethod
    def eval(cls, expression: Expr, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Translate(canvas, (0, 0, param))


class EulerRotateX(EulerRotate):
    @classmethod
    def eval(cls, expression: Expr, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return EulerRotate(canvas, (param, 0, 0))


class EulerRotateY(EulerRotate):
    @classmethod
    def eval(cls, expression: Expr, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return EulerRotate(canvas, (0, param, 0))


class EulerRotateZ(EulerRotate):
    @classmethod
    def eval(cls, expression: Expr, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return EulerRotate(canvas, (0, 0, param))


class ScaleX(Scale):
    @classmethod
    def eval(cls, expression: Expr, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Scale(canvas, (param, 1, 1))


class ScaleY(Scale):
    @classmethod
    def eval(cls, expression: Expr, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Scale(canvas, (1, param, 1))


class ScaleZ(Scale):
    @classmethod
    def eval(cls, expression: Expr, param: float):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        return Scale(canvas, (1, 1, param))


# TODO: Implementation

class Reflect(Modifier):
    """Performs union of canvas and its reflection about the origin, 
    with the reflection plane's normal vector specified by param.
    """
    @classmethod
    def eval(cls, expression: Expr, param: param_type):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        expression = Union(canvas, Reflect(canvas, param))
        return expression


class ReflectX(Reflect):
    @classmethod
    def eval(cls, expression: Expr):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        expression = Reflect(canvas, (1, 0, 0))
        return expression.doit(deep=deep, **hints)


class ReflectY(Reflect):
    @classmethod
    def eval(cls, expression: Expr):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        expression = Reflect(canvas, (0, 1, 0))
        return expression.doit(deep=deep, **hints)


class ReflectZ(Reflect):
    @classmethod
    def eval(cls, expression: Expr):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        expression = Reflect(canvas, (0, 0, 1))
        return expression.doit(deep=deep, **hints)

# ShapeAssembly functions

class Repeat(Modifier):
    """Performs 
    """
    ...



class Attach(Modifier):
    """Performs 
    """
    ...

class Squeeze(Modifier):
    ...

class Elongate(Modifier):
    ...

class Round(Modifier):
    ...

class Onion(Modifier):
    ...

class Twist(Modifier):
    ...