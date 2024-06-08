from .base_symbolic import GLExpr, GLFunction
from sympy import Tuple as SympyTuple


class Primitive2D(GLFunction):
    """Functions for declaring 2D primitives."""

    def pretty_print(self, tabs=0, tab_str="\t"):
        args = self.args
        n_tabs = tab_str * tabs
        replaced_args = [self.lookup_table.get(arg, arg) for arg in args]
        str_args = []
        for arg in replaced_args:
            if isinstance(arg, (GLExpr, GLFunction)):
                str_args.append(arg.pretty_print(tabs=tabs + 1, tab_str=tab_str))
            else:
                if isinstance(arg, SympyTuple):
                    item = [f"{x:.3f}" for x in arg]
                    item = ", ".join(item)
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
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_circle
    Read evaluator specific documentation for more.
    """


class RoundedBox2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_rounded_box
    Read evaluator specific documentation for more.
    """


class Box2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_box
    Read evaluator specific documentation for more.
    """


class Rectangle2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_box
    Read evaluator specific documentation for more.
    """


class OrientedBox2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_oriented_box
    Read evaluator specific documentation for more.
    """


class Rhombus2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_rhombus
    Read evaluator specific documentation for more.
    """


class Trapezoid2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_trapezoid
    Read evaluator specific documentation for more.
    """


class Parallelogram2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_parallelogram
    Read evaluator specific documentation for more.
    """


class EquilateralTriangle2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_equilateral_triangle
    Read evaluator specific documentation for more.
    """


class IsoscelesTriangle2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_isosceles_triangle
    Read evaluator specific documentation for more.
    """


class Triangle2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_circle
    Read evaluator specific documentation for more.
    """


class UnevenCapsule2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_uneven_capsule
    Read evaluator specific documentation for more.
    """


class RegularPentagon2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_regular_pentagon
    Read evaluator specific documentation for more.
    """


class RegularHexagon2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_regular_hexagon
    Read evaluator specific documentation for more.
    """


class Hexagram2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_hexagram
    Read evaluator specific documentation for more.
    """


class Star2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_star_5
    Read evaluator specific documentation for more.
    """


class RegularStar2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_regular_star
    Read evaluator specific documentation for more.
    """


class Pie2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_pie
    Read evaluator specific documentation for more.
    """


class CutDisk2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_cut_disk
    Read evaluator specific documentation for more.
    """


class Arc2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_arc
    Read evaluator specific documentation for more.
    """


class HorseShoe2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_horse_shoe
    Read evaluator specific documentation for more.
    """


class Vesica2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_vesica
    Read evaluator specific documentation for more.
    """


class OrientedVesica2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_oriented_vesica
    Read evaluator specific documentation for more.
    """


class Moon2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_moon
    Read evaluator specific documentation for more.
    """


class RoundedCross2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_rounded_cross
    Read evaluator specific documentation for more.
    """


class Egg2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_egg
    Read evaluator specific documentation for more.
    """


class Heart2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_heart
    Read evaluator specific documentation for more.
    """


class Cross2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_cross
    Read evaluator specific documentation for more.
    """


class RoundedX2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_rounded_x
    Read evaluator specific documentation for more.
    """


class Polygon2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_polygon
    Read evaluator specific documentation for more.
    """


class Ellipse2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_ellipse
    Read evaluator specific documentation for more.
    """


class Parabola2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_parabola
    Read evaluator specific documentation for more.
    """


class ParabolaSegment2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_parabola_segment
    Read evaluator specific documentation for more.
    """


class BlobbyCross2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_blobby_cross
    Read evaluator specific documentation for more.
    """


class Tunnel2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_tunnel
    Read evaluator specific documentation for more.
    """


class Stairs2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_stairs
    Read evaluator specific documentation for more.
    """


class QuadraticCircle2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_quadratic_circle
    Read evaluator specific documentation for more.
    """


class CoolS2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_cool_s
    Read evaluator specific documentation for more.
    """


class CircleWave2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_circle_wave
    Read evaluator specific documentation for more.
    """


class Hyperbola2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_hyperbola
    Read evaluator specific documentation for more.
    """


class QuadraticBezierCurve2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_quadratic_bezier_curve
    Read evaluator specific documentation for more.
    """


class Segment2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_segment
    Read evaluator specific documentation for more.
    """


class NoParamRectangle2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_no_param_rectangle
    Read evaluator specific documentation for more.
    """


class NoParamCircle2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_no_param_circle
    Read evaluator specific documentation for more.
    """


class NoParamTriangle2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_no_param_triangle
    Read evaluator specific documentation for more.
    """


class NullExpression2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf_null_op
    Read evaluator specific documentation for more.
    """

class TileUV2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_tile_primitive
    Read evaluator specific documentation for more.
    """

class SinRepeatX2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_sin_x
    Read evaluator specific documentation for more.
    """

class SinRepeatY2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_sin_y
    Read evaluator specific documentation for more.
    """ 
    
class SinOriginY2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_sin_y
    Read evaluator specific documentation for more.
    """ 
    