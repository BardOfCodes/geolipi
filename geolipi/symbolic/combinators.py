from typing import Tuple, List, Union
# from .base_expr_class import Expr, Function
from sympy import Function, Expr
from sympy.core.numbers import Float
from .primitives_3d import Primitive3D
from .base_expr_class import Expr

class Combinator(Function):
    """No Free variables.
    """
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True

class Union(Combinator):
    ...

class Intersection(Combinator):
    @classmethod
    def eval(cls, *args):
        return None

class Complement(Combinator):
    @classmethod
    def eval(cls, *args):
        return None

    def doit(self, deep=False, **hints):
        exprs = self.args
        # We can push the complement down to the primitives.
        raise NotImplementedError

class Difference(Combinator):
    """A - B"""
    @classmethod
    def eval(cls, expr_a: Expr, expr_b: Expr):
        return None

    def doit(self, deep=False, **hints):
        expr_a, expr_b = self.args
        if deep:
            expr_a = expr_a.doit(deep=deep, **hints)
            expr_b = expr_b.doit(deep=deep, **hints)
        expression = Intersection(expr_a, Complement(expr_b))
        return expression

class SwitchedDifference(Combinator):
    """B - A"""
    @classmethod
    def eval(cls, expr_a: Expr, expr_b: Expr):
        return None

    def doit(self, deep=False, **hints):
        expr_a, expr_b = self.args
        if deep:
            expr_a = expr_a.doit(deep=deep, **hints)
            expr_b = expr_b.doit(deep=deep, **hints)
        expression = Intersection(expr_b, Complement(expr_a))
        return expression