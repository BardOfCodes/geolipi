from typing import Tuple, List
from .base_symbolic import GLExpr, GLFunction
from sympy import Tuple as SympyTuple


class Primitive2D(GLFunction):
    """Functions for declaring 2D primitives."""
    ...

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


class Circle2D(Primitive2D):
    ...


class RoundedBox2D(Primitive2D):
    ...


class Box2D(Primitive2D):
    ...

# Depreciate this.


class Rectangle2D(Primitive2D):
    ...


class OrientedBox2D(Primitive2D):
    ...


class Rhombus2D(Primitive2D):
    ...


class Trapezoid2D(Primitive2D):
    ...


class Parallelogram2D(Primitive2D):
    ...


class EquilateralTriangle2D(Primitive2D):
    ...


class IsoscelesTriangle2D(Primitive2D):
    ...


class Triangle2D(Primitive2D):
    ...


class UnevenCapsule2D(Primitive2D):
    ...


class RegularPentagon2D(Primitive2D):
    ...


class RegularHexagon2D(Primitive2D):
    ...


class Hexagram2D(Primitive2D):
    ...


class Star2D(Primitive2D):
    ...


class RegularStar2D(Primitive2D):
    ...


class Pie2D(Primitive2D):
    ...


class CutDisk2D(Primitive2D):
    ...


class Arc2D(Primitive2D):
    ...


class HorseShoe2D(Primitive2D):
    ...


class Vesica2D(Primitive2D):
    ...


class OrientedVesica2D(Primitive2D):
    ...


class Moon2D(Primitive2D):
    ...


class RoundedCross2D(Primitive2D):
    ...


class Egg2D(Primitive2D):
    ...


class Heart2D(Primitive2D):
    ...


class Cross2D(Primitive2D):
    ...


class RoundedX2D(Primitive2D):
    ...


class Polygon2D(Primitive2D):
    ...


class Ellipse2D(Primitive2D):
    ...


class Parabola2D(Primitive2D):
    ...


class ParabolaSegment2D(Primitive2D):
    ...


class BlobbyCross2D(Primitive2D):
    ...


class Tunnel2D(Primitive2D):
    ...


class Stairs2D(Primitive2D):
    ...


class QuadraticCircle2D(Primitive2D):
    ...


class CoolS2D(Primitive2D):
    ...


class CircleWave2D(Primitive2D):
    ...


class Hyperbola2D(Primitive2D):
    ...


class QuadraticBezierCurve2D(Primitive2D):
    ...


class Segment2D(Primitive2D):
    ...


class NoParamRectangle2D(Primitive2D):
    ...


class NoParamCircle2D(Primitive2D):
    ...


class NoParamTriangle2D(Primitive2D):
    ...

class NullExpression2D(Primitive2D):
    ...