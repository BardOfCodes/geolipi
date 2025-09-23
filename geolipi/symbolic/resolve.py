import torch as th
from typing import Union as type_union, Tuple
from .base import GLExpr, GLFunction
from sympy import Symbol, Tuple as SympyTuple, Integer as SympyInteger
from .symbol_types import (
    ALL_REFLECTS,
    PREFIXED_AXIS_REFLECT,
    AXIAL_REFLECT,
    ALL_SYMS,
    PREFIXED_AXIS_SYM,
    AXIAL_PARAM_SYM,
)

from .transforms_3d import (
    Reflect3D,
    ReflectCoords3D,
    Translate3D,
    EulerRotate3D,
    Scale3D,
    TranslationSymmetry3D,
    RotationSymmetry3D,
    ScaleSymmetry3D,
    ReflectX3D,
    ReflectY3D,
    ReflectZ3D,
    RotationSymmetryX3D,
    RotationSymmetryY3D,
    RotationSymmetryZ3D,
    TranslationSymmetryX3D,
    TranslationSymmetryY3D,
    TranslationSymmetryZ3D,
    AxialScaleSymmetry3D,
    AxialReflect3D,
    AxialRotationSymmetry3D,
    AxialTranslationSymmetry3D,
)
from .transforms_2d import (
    Reflect2D,
    ReflectCoords2D,
    Translate2D,
    EulerRotate2D,
    Scale2D,
    TranslationSymmetry2D,
    RotationSymmetry2D,
    ScaleSymmetry2D,
    ReflectX2D,
    ReflectY2D,
    TranslationSymmetryX2D,
    TranslationSymmetryY2D,
    AxialScaleSymmetry2D,
    AxialTranslationSymmetry2D,
    AxialReflect2D,
)
from .combinators import Union

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


def resolve_macros(expr: GLFunction, device) -> GLFunction:
    """Extends macros in GeoLIPI."""
    resolved_args = []
    for sub_expr in expr.args:
        if isinstance(sub_expr, (Tuple, SympyTuple, SympyInteger)):
            arg = sub_expr
        elif isinstance(sub_expr, Symbol):
            if sub_expr in expr.lookup_table:
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
        if isinstance(resolved_args[2], th.Tensor):
            n_count = resolved_args[2].long()
        else:
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
        for ind in range(0, n_count - 1):
            delta = dist * (ind + 1) * dir
            all_subexprs.append(trans_func(subexpr, delta))
        all_subexprs.insert(0, subexpr)
        new_expr = Union(*all_subexprs)

    else:
        new_expr = type(expr)(*resolved_args)
    return new_expr


def load_expr_from_str(expr_str: str, mapper, device):
    """To load GLExprs from string."""

    expr = eval(expr_str, mapper)
    expr = expr.to(device)
    return expr
