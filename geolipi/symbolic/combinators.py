from typing import Tuple, List
from sympy.core.numbers import Float

from .base_symbolic import Expr
from .base_symbolic import GLExpr, GLFunction


class Combinator(GLFunction):
    ...


class Union(Combinator):
    ...


class JoinUnion(Union):
    # For Blender Only
    ...


class Intersection(Combinator):
    ...


class Complement(Combinator):
    ...


class Difference(Combinator):
    ...


class SwitchedDifference(Combinator):
    ...


class SmoothUnion(Combinator):
    ...


class SmoothIntersection(Combinator):
    ...


class SmoothDifference(Combinator):
    ...
