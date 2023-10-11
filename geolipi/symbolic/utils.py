from .base_symbolic import GLExpr, GLFunction

from .transforms_3d import Reflect3D, ReflectCoords3D
from .transforms_3d import Translate3D, EulerRotate3D, Scale3D
from .transforms_3d import TranslationSymmetry3D, RotationSymmetry3D, ScaleSymmetry3D
from .transforms_3d import (ReflectX3D, ReflectY3D, ReflectZ3D, 
                            RotationSymmetryX3D, RotationSymmetryY3D, RotationSymmetryZ3D,
                            TranslationSymmetryX3D, TranslationSymmetryY3D, TranslationSymmetryZ3D)
from .transforms_3d import (AxialScaleSymmetry3D, AxialReflect3D, AxialRotationSymmetry3D, 
                            AxialTranslationSymmetry3D)

from .transforms_2d import Reflect2D, ReflectCoords2D
from .transforms_2d import Translate2D, EulerRotate2D, Scale2D
from .transforms_2d import TranslationSymmetry2D, RotationSymmetry2D, ScaleSymmetry2D
from .transforms_2d import ReflectX2D, ReflectY2D, TranslationSymmetryX2D, TranslationSymmetryY2D
from .transforms_2d import (AxialScaleSymmetry2D, AxialTranslationSymmetry2D, AxialReflect2D)

from .transforms_2d import Modifier2D, ColorTree2D
from .transforms_3d import Modifier3D, ColorTree3D
from .primitives_2d import Primitive2D
from .primitives_3d import Primitive3D

from .combinators import Union, Intersection, Difference, Complement, PseudoUnion
from sympy import Symbol, Tuple as SympyTuple, Integer as SympyInteger
import torch as th
from typing import Union as type_union, Tuple
from sympy import Function, Expr
from .base_symbolic import GLExpr, GLFunction
import numpy as np
import torch as th
import re

PARAM_TYPE = type_union[np.ndarray, th.Tensor]
MOD_TYPE = type_union[Modifier3D, Modifier2D]
TRANSLATE_TYPE = type_union[Translate3D, Translate2D]
ROTATE_TYPE = type_union[EulerRotate3D, EulerRotate2D]
SCALE_TYPE = type_union[Scale3D, Scale2D]
PRIM_TYPE = type_union[Primitive3D, Primitive2D]
COLOR_TYPE = type_union[ColorTree2D, ColorTree3D]
COMBINATOR_TYPE = type_union[Union, Intersection, Difference, Complement, PseudoUnion]

GENERAL_REFLECT = type_union[Reflect2D, Reflect3D]
GENERAL_PARAM_SYM = type_union[RotationSymmetry2D, RotationSymmetry3D, TranslationSymmetry2D, TranslationSymmetry3D, ScaleSymmetry2D, ScaleSymmetry3D]
GENERAL_MACROS = type_union[GENERAL_REFLECT, GENERAL_PARAM_SYM]

PREFIXED_AXIS_REFLECT = type_union[ReflectX2D, ReflectY2D, ReflectX3D, ReflectY3D, ReflectZ3D]
PREFIXED_AXIS_SYM = type_union[TranslationSymmetryX2D, TranslationSymmetryY2D, TranslationSymmetryX3D, TranslationSymmetryY3D, TranslationSymmetryZ3D,
                          RotationSymmetryX3D, RotationSymmetryY3D, RotationSymmetryZ3D]
PREFIXED_AXIS_MACROS = type_union[PREFIXED_AXIS_REFLECT, PREFIXED_AXIS_SYM]

AXIAL_REFLECT = type_union[AxialReflect2D, AxialReflect3D]
AXIAL_PARAM_SYM = type_union[AxialTranslationSymmetry2D, AxialTranslationSymmetry3D, 
                             AxialRotationSymmetry3D, AxialScaleSymmetry2D, AxialScaleSymmetry3D]
AXIAL_PARAM_MACROS = type_union[AXIAL_REFLECT, AXIAL_PARAM_SYM]

ALL_REFLECTS = type_union[GENERAL_REFLECT, PREFIXED_AXIS_REFLECT, AXIAL_REFLECT]
ALL_SYMS = type_union[GENERAL_PARAM_SYM, PREFIXED_AXIS_SYM, AXIAL_PARAM_SYM]


MACRO_TYPE = type_union[GENERAL_MACROS, PREFIXED_AXIS_MACROS, AXIAL_PARAM_MACROS]

AXIS_MAPPER = {
    Symbol("AX3D"): th.tensor([1, 0, 0], dtype=th.float32),
    Symbol("AY3D"): th.tensor([0, 1, 0], dtype=th.float32),
    Symbol("AZ3D"): th.tensor([0, 0, 1], dtype=th.float32),
    Symbol("AX2D"): th.tensor([1, 0], dtype=th.float32),
    Symbol("AY2D"): th.tensor([0, 1], dtype=th.float32),
    ReflectX3D: th.tensor([1, 0, 0], dtype=th.float32),
    ReflectY3D: th.tensor([0, 1, 0], dtype=th.float32),
    ReflectZ3D: th.tensor([0, 0, 1], dtype=th.float32),
    ReflectX2D: th.tensor([1, 0], dtype=th.float32),
    ReflectY2D: th.tensor([0, 1], dtype=th.float32),
    TranslationSymmetryX2D: th.tensor([1, 0], dtype=th.float32),
    TranslationSymmetryY2D: th.tensor([0, 1], dtype=th.float32),
    TranslationSymmetryX3D: th.tensor([1, 0, 0], dtype=th.float32),
    TranslationSymmetryY3D: th.tensor([0, 1, 0], dtype=th.float32),
    TranslationSymmetryZ3D: th.tensor([0, 0, 1], dtype=th.float32),
    RotationSymmetryX3D: th.tensor([1, 0, 0], dtype=th.float32),
    RotationSymmetryY3D: th.tensor([0, 1, 0], dtype=th.float32),
    RotationSymmetryZ3D: th.tensor([0, 0, 1], dtype=th.float32),
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
    TranslationSymmetryX2D: Translate2D,
    TranslationSymmetryY2D: Translate2D,
    AxialTranslationSymmetry2D: Translate2D,
    TranslationSymmetry2D: Translate2D,
    TranslationSymmetryX3D: Translate3D,
    TranslationSymmetryY3D: Translate3D,
    TranslationSymmetryZ3D: Translate3D,
    AxialTranslationSymmetry3D: Translate3D,
    TranslationSymmetry3D: Translate3D,
    
    RotationSymmetry2D: EulerRotate2D,
    RotationSymmetryX3D: EulerRotate3D,
    RotationSymmetryY3D: EulerRotate3D,
    RotationSymmetryZ3D: EulerRotate3D,
    RotationSymmetry3D: EulerRotate3D,
    AxialRotationSymmetry3D: EulerRotate3D,
    
    ScaleSymmetry2D: Scale2D,
    ScaleSymmetry3D: Scale3D,
}


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

    if isinstance(expr, ALL_REFLECTS):
        arg = resolved_args[0]
        if isinstance(expr, PREFIXED_AXIS_REFLECT):
            param = AXIS_MAPPER[expr.__class__].clone().to(device)
        elif isinstance(expr, AXIAL_REFLECT):
            axis = resolved_args[1]
            param = AXIS_MAPPER[axis].clone().to(device)
        else:
            param = resolved_args[1]
        new_expr = Union(arg, REFLECT_OP_MAP[expr.__class__](arg, param))
            
    elif isinstance(expr, ALL_SYMS):
        subexpr = resolved_args[0]
        subexpr = resolved_args[0]
        dist = resolved_args[1]
        n_count = resolved_args[2]
        if isinstance(expr, PREFIXED_AXIS_SYM):
            dir = AXIS_MAPPER[expr.__class__].clone().to(device)
        elif isinstance(expr, AXIAL_PARAM_SYM):
            axis = resolved_args[3]
            dir = AXIS_MAPPER[axis].clone().to(device)
        else:
            # This is one model where we get the direction and magnitude separately.
            # dir = resolved_args[3]
            # option 2: we get the direction and magnitude together.
            dir = 1
        all_subexprs = []
        trans_func = SYM_OP_MAP[expr.__class__]
        for ind in range(0, n_count-1):
            delta = dist * (ind + 1)* dir
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