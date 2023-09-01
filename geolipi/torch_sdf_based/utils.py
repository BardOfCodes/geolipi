
from typing import Dict, List, Tuple, Union as type_union
from sympy import Symbol, Function
from geolipi.symbolic.base_symbolic import GLExpr, GLFunction
import inspect
import torch as th
import numpy as np
import rustworkx as rx
from collections import defaultdict
from geolipi.symbolic import Primitive3D, Primitive2D, Combinator, Modifier3D, Modifier2D
from geolipi.symbolic.primitives_3d import Cuboid3D, Sphere3D, Cylinder3D, Cone3D, NoParamSphere3D, NoParamCylinder3D, NoParamCuboid3D
from geolipi.symbolic.primitives_2d import Rectangle2D, Circle2D, NoParamRectangle2D, NoParamCircle2D, Triangle2D, TriangleEquilateral2D, NoParamTriangle2D
from geolipi.symbolic.combinators import Union, Intersection, Difference, Complement
from geolipi.symbolic.transforms_3d import Translate3D, EulerRotate3D, Scale3D, ReflectCoords3D, ColorTree3D
from geolipi.symbolic.transforms_2d import Translate2D, EulerRotate2D, Scale2D, ReflectCoords2D, ColorTree2D
from geolipi.symbolic.utils import resolve_macros, REFLECT_MACROS, SYM_MACROS, AXIAL_REFLECT_MACROS

from .sketcher import Sketcher
from .transforms import get_affine_translate_3D, get_affine_scale_3D, get_affine_rotate_euler_3D, get_affine_reflection_3D
from .transforms import get_affine_translate_2D, get_affine_scale_2D, get_affine_rotate_2D, get_affine_reflection_2D

from .sdf_functions import sdf3d_cuboid, sdf3d_sphere, sdf3d_cylinder, sdf3d_no_param_cuboid, sdf3d_no_param_cylinder, sdf3d_no_param_sphere
from .sdf_functions import sdf2d_rectangle, sdf2d_circle, sdf2d_no_param_rectangle, sdf2d_no_param_circle, sdf2d_triangle, sdf2d_equilateral_triangle, sdf2d_no_param_triangle
from .sdf_functions import sdf_union, sdf_intersection, sdf_difference, sdf_complement

# TODO: Clean the usage of primitives with params.

PARAM_TYPE = type_union[np.ndarray, th.Tensor]
MOD_TYPE = type_union[Modifier3D, Modifier2D]
TRANSLATE_TYPE = type_union[Translate3D, Translate2D]
SCALE_TYPE = type_union[Scale3D, Scale2D]
PRIM_TYPE = type_union[Primitive3D, Primitive2D]
MACRO_TYPE = type_union[REFLECT_MACROS, SYM_MACROS, AXIAL_REFLECT_MACROS]
COLOR_TYPE = type_union[ColorTree2D, ColorTree3D]
MODIFIER_MAP = {
    Translate3D: get_affine_translate_3D,
    EulerRotate3D: get_affine_rotate_euler_3D,
    Scale3D: get_affine_scale_3D,
    ReflectCoords3D: get_affine_reflection_3D,
    Translate2D: get_affine_translate_2D,
    EulerRotate2D: get_affine_rotate_2D,
    Scale2D: get_affine_scale_2D,
    ReflectCoords2D: get_affine_reflection_2D,

}
PRIMITIVE_MAP = {
    Cuboid3D: sdf3d_cuboid,
    Sphere3D: sdf3d_sphere,
    Cylinder3D: sdf3d_cylinder,
    NoParamCuboid3D: sdf3d_no_param_cuboid,
    NoParamSphere3D: sdf3d_no_param_sphere,
    NoParamCylinder3D: sdf3d_no_param_cylinder,
    Rectangle2D: sdf2d_rectangle,
    Circle2D: sdf2d_circle,
    NoParamRectangle2D: sdf2d_no_param_rectangle,
    NoParamCircle2D: sdf2d_no_param_circle,
    Triangle2D: sdf2d_triangle,
    TriangleEquilateral2D: sdf2d_equilateral_triangle,
    NoParamTriangle2D: sdf2d_no_param_triangle
}

COMBINATOR_MAP = {
    Union: sdf_union,
    Intersection: sdf_intersection,
    Difference: sdf_difference,
    Complement: sdf_complement,
}

# Fore resolution:

INVERTED_MAP = {
    Union: Intersection,
    Intersection: Union,
    Difference: Union,
}
NORMAL_MAP = {
    Union: Union,
    Intersection: Intersection,
    Difference: Intersection,
}
# ONLY_SIMPLIFY_RULES = set(["ii", "uu"])

ONLY_SIMPLIFY_RULES = set([(Intersection, Intersection), (Union, Union)])
ALL_RULES = set([(Intersection, Intersection),
                (Union, Union), (Intersection, Union)])


class PrimitiveSpec(GLFunction):
    @classmethod
    def eval(cls, prim_type: type, shift: int):
        return None


COLOR_MAP = {
    "RED": th.tensor([1.0, 0.0, 0.0]),
    "GREEN": th.tensor([0.0, 1.0, 0.0]),
    "BLUE": th.tensor([0.0, 0.0, 1.0]),
    "YELLOW": th.tensor([1.0, 1.0, 0.0]),
    "CYAN": th.tensor([0.0, 1.0, 1.0]),
    "MAGENTA": th.tensor([1.0, 0.0, 1.0]),
    "GRAY": th.tensor([0.5, 0.5, 0.5]),
    "WHITE": th.tensor([1.0, 1.0, 1.0]),
    "BLACK": th.tensor([0.0, 0.0, 0.0]),
    "ORANGE": th.tensor([1.0, 0.5, 0.0]),
    "PURPLE": th.tensor([0.5, 0.0, 0.5]),
    "PINK": th.tensor([1.0, 0.0, 0.5]),
    "BROWN": th.tensor([0.5, 0.25, 0.0]),
    "DARK_GREEN": th.tensor([0.0, 0.5, 0.0]),
    "DARK_BLUE": th.tensor([0.0, 0.0, 0.5]),
    "DARK_RED": th.tensor([0.5, 0.0, 0.0]),
}