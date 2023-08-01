from typing import Tuple, List, Union
from sympy import Function
from sympy.core.numbers import Float
from .primitives import Primitive
from .transforms import ReflectTransform

canvas_type = Union[Function, Primitive]

class Combinator(Function):
    """Takes an arbitrary number of primitives as input.
    """
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True

class Union(Combinator):
    @classmethod
    def eval(cls, *args):
        return None


class Intersection(Combinator):
    @classmethod
    def eval(cls, *args):
        return None

class Complement(Combinator):
    @classmethod
    def eval(cls, *args):
        return None


class Difference(Combinator):
    @classmethod
    def eval(cls, canvas_a: canvas_type, canvas_b: canvas_type):
        return None

    def doit(self, deep=False, **hints):
        canvas_a, canvas_b = self.args
        if deep:
            canvas_a = canvas_a.doit(deep=deep, **hints)
            canvas_b = canvas_b.doit(deep=deep, **hints)
        expression = Intersection(canvas_a, Complement(canvas_b))
        return expression

class Reflect(Combinator):
    """Performs union of canvas and its reflection about the origin, 
    with the reflection plane's normal vector specified by param.
    """
    @classmethod
    def eval(cls, canvas: canvas_type, param: Tuple[float, ...]):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        expression = Union(canvas, ReflectTransform(canvas, param))
        return expression


class ReflectX(Reflect):
    @classmethod
    def eval(cls, canvas: canvas_type):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        expression = Reflect(canvas, (1, 0, 0))
        return expression.doit(deep=deep, **hints)


class ReflectY(Reflect):
    @classmethod
    def eval(cls, canvas: canvas_type):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        expression = Reflect(canvas, (0, 1, 0))
        return expression.doit(deep=deep, **hints)


class ReflectZ(Reflect):
    @classmethod
    def eval(cls, canvas: canvas_type):
        return None

    def doit(self, deep=False, **hints):
        canvas, param = self.args
        if deep:
            canvas = canvas.doit(deep=deep, **hints)
        expression = Reflect(canvas, (0, 0, 1))
        return expression.doit(deep=deep, **hints)

# ShapeAssembly functions


class Repeat(Combinator):
    """Performs 
    """
    ...


class Attach(Combinator):
    """Performs 
    """
    ...


class Squeeze(Combinator):
    ...
