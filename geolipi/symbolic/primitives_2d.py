from .base import GLExpr, GLFunction
import sympy as sp
from sympy import Tuple as SympyTuple
from .registry import register_symbol

class Primitive2D(GLFunction):
    """Functions for declaring 2D primitives."""

    def pretty_print(self, tabs=0, tab_str="\t"):
        args = self.args
        n_tabs = tab_str * tabs
        replaced_args = []
        for arg in args:
            if isinstance(arg, sp.Symbol):
                replaced_args.append(self.lookup_table.get(arg, arg))
            else:
                replaced_args.append(arg)
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


@register_symbol
class Circle2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_circle
    Read evaluator specific documentation for more.
    """


@register_symbol
class RoundedBox2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_rounded_box
    Read evaluator specific documentation for more.
    """


@register_symbol
class Box2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_box
    Read evaluator specific documentation for more.
    """


@register_symbol
class Rectangle2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_box
    Read evaluator specific documentation for more.
    """


@register_symbol
class OrientedBox2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_oriented_box
    Read evaluator specific documentation for more.
    """


@register_symbol
class Rhombus2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_rhombus
    Read evaluator specific documentation for more.
    """


@register_symbol
class Trapezoid2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_trapezoid
    Read evaluator specific documentation for more.
    """


@register_symbol
class Parallelogram2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_parallelogram
    Read evaluator specific documentation for more.
    """


@register_symbol
class EquilateralTriangle2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_equilateral_triangle
    Read evaluator specific documentation for more.
    """


@register_symbol
class IsoscelesTriangle2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_isosceles_triangle
    Read evaluator specific documentation for more.
    """


@register_symbol
class Triangle2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_circle
    Read evaluator specific documentation for more.
    """


@register_symbol
class UnevenCapsule2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_uneven_capsule
    Read evaluator specific documentation for more.
    """


@register_symbol
class RegularPentagon2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_regular_pentagon
    Read evaluator specific documentation for more.
    """


@register_symbol
class RegularHexagon2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_regular_hexagon
    Read evaluator specific documentation for more.
    """


@register_symbol
class RegularOctagon2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_regular_octagon
    Read evaluator specific documentation for more.
    """


@register_symbol
class Hexagram2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_hexagram
    Read evaluator specific documentation for more.
    """


@register_symbol
class Star2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_star_5
    Read evaluator specific documentation for more.
    """


@register_symbol
class RegularStar2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_regular_star
    Read evaluator specific documentation for more.
    """


@register_symbol
class Pie2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_pie
    Read evaluator specific documentation for more.
    """


@register_symbol
class CutDisk2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_cut_disk
    Read evaluator specific documentation for more.
    """


@register_symbol
class Arc2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_arc
    Read evaluator specific documentation for more.
    """


@register_symbol
class HorseShoe2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_horse_shoe
    Read evaluator specific documentation for more.
    """


@register_symbol
class Vesica2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_vesica
    Read evaluator specific documentation for more.
    """


@register_symbol
class OrientedVesica2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_oriented_vesica
    Read evaluator specific documentation for more.
    """


@register_symbol
class Moon2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_moon
    Read evaluator specific documentation for more.
    """


@register_symbol
class RoundedCross2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_rounded_cross
    Read evaluator specific documentation for more.
    """


@register_symbol
class Egg2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_egg
    Read evaluator specific documentation for more.
    """


@register_symbol
class Heart2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_heart
    Read evaluator specific documentation for more.
    """


@register_symbol
class Cross2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_cross
    Read evaluator specific documentation for more.
    """


@register_symbol
class RoundedX2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_rounded_x
    Read evaluator specific documentation for more.
    """


@register_symbol
class Polygon2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_polygon
    Read evaluator specific documentation for more.
    """


@register_symbol
class Ellipse2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_ellipse
    Read evaluator specific documentation for more.
    """


@register_symbol
class Parabola2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_parabola
    Read evaluator specific documentation for more.
    """


@register_symbol
class ParabolaSegment2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_parabola_segment
    Read evaluator specific documentation for more.
    """


@register_symbol
class BlobbyCross2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_blobby_cross
    Read evaluator specific documentation for more.
    """


@register_symbol
class Tunnel2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_tunnel
    Read evaluator specific documentation for more.
    """


@register_symbol
class Stairs2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_stairs
    Read evaluator specific documentation for more.
    """


@register_symbol
class QuadraticCircle2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_quadratic_circle
    Read evaluator specific documentation for more.
    """


@register_symbol
class CoolS2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_cool_s
    Read evaluator specific documentation for more.
    """


@register_symbol
class CircleWave2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_circle_wave
    Read evaluator specific documentation for more.
    """


@register_symbol
class Hyperbola2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_hyperbola
    Read evaluator specific documentation for more.
    """


@register_symbol
class QuadraticBezierCurve2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_quadratic_bezier_curve
    Read evaluator specific documentation for more.
    """


@register_symbol
class Segment2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_segment
    Read evaluator specific documentation for more.
    """


@register_symbol
class NoParamRectangle2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_no_param_rectangle
    Read evaluator specific documentation for more.
    """


@register_symbol
class NoParamCircle2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_no_param_circle
    Read evaluator specific documentation for more.
    """


@register_symbol
class NoParamTriangle2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_no_param_triangle
    Read evaluator specific documentation for more.
    """


@register_symbol
class NullExpression2D(Primitive2D):

    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf_null_op
    Read evaluator specific documentation for more.
    """

@register_symbol
class TileUV2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_tile_primitive
    Read evaluator specific documentation for more.
    """

### THese are not really SDFS


@register_symbol
class SinRepeatX2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_sin_x
    Read evaluator specific documentation for more.
    """

@register_symbol
class SinRepeatY2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_sin_y
    Read evaluator specific documentation for more.
    """ 
    
@register_symbol
class SinAlongAxisY2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_sin_y
    Read evaluator specific documentation for more.
    """ 
    
@register_symbol
class SinDiagonal2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_sin_diagonal
    Read evaluator specific documentation for more.
    """ 

@register_symbol
class SinDiagonalFlip2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_flip_sin_diagonal
    Read evaluator specific documentation for more.
    """

@register_symbol
class SinRadial2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_sin_radial
    Read evaluator specific documentation for more.
    """

@register_symbol
class SquiggleX2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_squiggle
    Read evaluator specific documentation for more.
    """

@register_symbol
class SquiggleY2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_squiggle
    Read evaluator specific documentation for more.
    """

@register_symbol
class SquiggleDiagonal2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_squiggle_diagonal
    Read evaluator specific documentation for more.
    """

@register_symbol
class SquiggleDiagonalFlip2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_flip_squiggle_diagonal
    Read evaluator specific documentation for more.
    """

@register_symbol
class SquiggleRadial2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_squiggle_radial
    Read evaluator specific documentation for more.
    """
    
@register_symbol
class SquiggleDistortion2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_squiggle_distortion
    Read evaluator specific documentation for more.
    """

@register_symbol
class InstantiatedPrim2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_instantiated_primitive
    Read evaluator specific documentation for more.
    """

@register_symbol
class PolyLine2D(Primitive2D):
    """
    This class is mapped to the following evaluator function(s):
    - torch_compute.primitives_2d.sdf2d_polyline
    Read evaluator specific documentation for more.
    """