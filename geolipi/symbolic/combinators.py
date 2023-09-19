from typing import Tuple, List
from .base_symbolic import GLExpr, GLFunction
# from sympy import Function, Expr
from sympy.core.numbers import Float
from .primitives_3d import Primitive3D
from .base_symbolic import Expr


class Combinator(GLFunction):
    """No Free variables.
    """
    has_sdf = True
    has_occupancy = True
    has_blender = True
    true_sdf = True


class Union(Combinator):
    @classmethod
    def eval(cls, *args):
        return None

class PseudoUnion(Union):
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
    """A - B"""
    @classmethod
    def eval(cls, expr_a: Expr, expr_b: Expr):
        return None


class SwitchedDifference(Combinator):
    """B - A"""
    @classmethod
    def eval(cls, expr_a: Expr, expr_b: Expr):
        return None
