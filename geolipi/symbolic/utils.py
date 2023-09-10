from .base_symbolic import GLExpr, GLFunction
from .transforms_3d import Reflect3D, ReflectCoords3D, ReflectX3D, ReflectY3D, ReflectZ3D
from .transforms_3d import AxialReflect3D, Translate3D, EulerRotate3D, Scale3D
from .transforms_3d import TranslationSymmetry3D, RotationSymmetry3D, ScaleSymmetry3D
from .transforms_2d import Reflect2D, ReflectCoords2D, ReflectX2D, ReflectY2D
from .transforms_2d import AxialReflect2D, Translate2D, EulerRotate2D, Scale2D
from .transforms_2d import TranslationSymmetry2D, RotationSymmetry2D, ScaleSymmetry2D
from .transforms_2d import Modifier2D, ColorTree2D
from .transforms_3d import Modifier3D, ColorTree3D
from .primitives_2d import Primitive2D
from .primitives_3d import Primitive3D

from .combinators import Union, Intersection, Difference, Complement
from sympy import Symbol, Tuple as SympyTuple, Integer as SympyInteger
import torch as th
from typing import Union as type_union, Tuple
from sympy import Function, Expr
from .base_symbolic import GLExpr, GLFunction
import numpy as np
import torch as th
import re

REFLECT_PARAM_MAP = {
    ReflectX3D: th.tensor([1, 0, 0], dtype=th.float32),
    ReflectY3D: th.tensor([0, 1, 0], dtype=th.float32),
    ReflectZ3D: th.tensor([0, 0, 1], dtype=th.float32),
    ReflectX2D: th.tensor([1, 0], dtype=th.float32),
    ReflectY2D: th.tensor([0, 1], dtype=th.float32),
}
AXIAL_REFLECT_PARAM_MAP = {
    Symbol("AX3D"): th.tensor([1, 0, 0], dtype=th.float32),
    Symbol("AY3D"): th.tensor([0, 1, 0], dtype=th.float32),
    Symbol("AZ3D"): th.tensor([0, 0, 1], dtype=th.float32),
    Symbol("AX2D"): th.tensor([1, 0], dtype=th.float32),
    Symbol("AY2D"): th.tensor([0, 1], dtype=th.float32),
}
REFLECT_OP_MAP = {
    ReflectX3D: ReflectCoords3D,
    ReflectY3D: ReflectCoords3D,
    ReflectZ3D: ReflectCoords3D,
    ReflectX2D: ReflectCoords2D,
    ReflectY2D: ReflectCoords2D,
    AxialReflect2D: ReflectCoords2D,
    AxialReflect3D: ReflectCoords3D,
}
SYM_OP_MAP = {
    TranslationSymmetry3D: Translate3D,
    TranslationSymmetry2D: Translate2D,
    RotationSymmetry3D: EulerRotate3D,
    RotationSymmetry2D: EulerRotate2D,
    ScaleSymmetry2D: Scale2D,
    ScaleSymmetry3D: Scale3D,
}

REFLECT_MACROS = type_union[ReflectX3D, ReflectY3D,
                           ReflectZ3D, ReflectX2D, ReflectY2D]

SYM_MACROS = type_union[TranslationSymmetry2D, TranslationSymmetry3D,
                       RotationSymmetry2D, RotationSymmetry3D, ScaleSymmetry2D, ScaleSymmetry3D]

AXIAL_REFLECT_MACROS = type_union[AxialReflect2D, AxialReflect3D]

PARAM_TYPE = type_union[np.ndarray, th.Tensor]
MOD_TYPE = type_union[Modifier3D, Modifier2D]
TRANSLATE_TYPE = type_union[Translate3D, Translate2D]
ROTATE_TYPE = type_union[EulerRotate3D, EulerRotate2D]
SCALE_TYPE = type_union[Scale3D, Scale2D]
PRIM_TYPE = type_union[Primitive3D, Primitive2D]
MACRO_TYPE = type_union[REFLECT_MACROS, SYM_MACROS, AXIAL_REFLECT_MACROS]
COLOR_TYPE = type_union[ColorTree2D, ColorTree3D]
COMBINATOR_TYPE = type_union[Union, Intersection, Difference, Complement]


# TODO: Fix the issue of tensor missing.

def resolve_macros(expr: GLFunction, device):
    """To resolve macros in GLExprs. Mostly useful in the SDF estimation process."""
    resolved_args = []
    for sub_expr in expr.args:
        if isinstance(sub_expr, (Tuple, SympyTuple, SympyInteger)):
            arg = sub_expr
        elif isinstance(sub_expr, Symbol):
            if sub_expr in expr.lookup_table.keys():
                arg = expr.lookup_table[sub_expr]
            else:
                arg = sub_expr
        else:
            arg = resolve_macros(sub_expr, device)
        resolved_args.append(arg)

    if isinstance(expr, REFLECT_MACROS):
        arg = resolved_args[0]
        param = REFLECT_PARAM_MAP[expr.__class__].clone().to(device)
        new_expr = Union(arg, REFLECT_OP_MAP[expr.__class__](arg, param))
    elif isinstance(expr, AXIAL_REFLECT_MACROS):
        arg = resolved_args[0]
        axis = resolved_args[1]
        param = AXIAL_REFLECT_PARAM_MAP[axis].clone().to(device)
        new_expr = Union(arg, REFLECT_OP_MAP[expr.__class__](arg, param))

    elif isinstance(expr, SYM_MACROS):
        subexpr = resolved_args[0]
        distance = resolved_args[1]
        n_count = resolved_args[2]
        all_subexprs = []
        trans_func = SYM_OP_MAP[expr.__class__]
        for ind in range(0, n_count-1):
            delta = distance * (ind + 1)
            all_subexprs.append(trans_func(subexpr, delta))
        all_subexprs.insert(0, subexpr)
        new_expr = Union(*all_subexprs)
    else:
        new_expr = type(expr)(*resolved_args)
    return new_expr

# TODO: Implement


def pretty_print(expr: GLExpr):
    """To pretty print GLExprs."""
    ...


def load_expr_from_str(expr_str: str, mapper, device):
    """To load GLExprs from string."""
    
    expr = eval(expr_str, mapper)
    expr = expr.to(device)
    return expr