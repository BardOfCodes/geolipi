from .base_symbolic import GLExpr, GLFunction


import re
import numpy as np
import torch as th
from typing import Union as type_union, Tuple
from sympy import Function, Expr
from sympy import Symbol, Tuple as SympyTuple, Integer as SympyInteger

from .base_symbolic import GLExpr, GLFunction
from .combinators import *
from .primitives_higher import *
from .primitives_2d import *
from .primitives_3d import *
from .transforms_2d import *
from .transforms_3d import *
from .color import *


PARAM_TYPE = type_union[np.ndarray, th.Tensor]
# Combinators:
COMBINATOR_TYPE = Combinator
BASE_COMBINATORS = type_union[
    Union, Intersection, Difference, Complement, SwitchedDifference
]
PARAMETERIZED_COMBINATORS = type_union[
    SmoothUnion, SmoothIntersection, SmoothDifference
]

# Modifiers
MOD_TYPE = type_union[Modifier3D, Modifier2D]
TRANSFORM_TYPE = type_union[Transform2D, Transform3D]
POSITIONALMOD_TYPE = type_union[PositionalTransform2D, PositionalTransform3D]
SDFMOD_TYPE = type_union[SDFModifier2D, SDFModifier3D]
MACRO_TYPE = type_union[Macro2D, Macro3D]

TRANSLATE_TYPE = type_union[
    Translate3D, Translate2D
]  # also the translation syms will be affected.
ROTATE_TYPE = type_union[EulerRotate3D, EulerRotate2D]
SCALE_TYPE = type_union[Scale3D, Scale2D]
SHEAR_TYPE = type_union[Shear3D, Shear2D]

GENERAL_REFLECT = type_union[Reflect2D, Reflect3D]
GENERAL_PARAM_SYM = type_union[
    RotationSymmetry2D,
    RotationSymmetry3D,
    TranslationSymmetry2D,
    TranslationSymmetry3D,
    ScaleSymmetry2D,
    ScaleSymmetry3D,
]
GENERAL_MACROS = type_union[GENERAL_REFLECT, GENERAL_PARAM_SYM]

PREFIXED_AXIS_REFLECT = type_union[
    ReflectX2D, ReflectY2D, ReflectX3D, ReflectY3D, ReflectZ3D
]
PREFIXED_AXIS_SYM = type_union[
    TranslationSymmetryX2D,
    TranslationSymmetryY2D,
    TranslationSymmetryX3D,
    TranslationSymmetryY3D,
    TranslationSymmetryZ3D,
    RotationSymmetryX3D,
    RotationSymmetryY3D,
    RotationSymmetryZ3D,
]
PREFIXED_AXIS_MACROS = type_union[PREFIXED_AXIS_REFLECT, PREFIXED_AXIS_SYM]

AXIAL_REFLECT = type_union[AxialReflect2D, AxialReflect3D]
AXIAL_PARAM_SYM = type_union[
    AxialTranslationSymmetry2D,
    AxialTranslationSymmetry3D,
    AxialRotationSymmetry3D,
    AxialScaleSymmetry2D,
    AxialScaleSymmetry3D,
]
AXIAL_PARAM_MACROS = type_union[AXIAL_REFLECT, AXIAL_PARAM_SYM]

ALL_REFLECTS = type_union[GENERAL_REFLECT, PREFIXED_AXIS_REFLECT, AXIAL_REFLECT]
ALL_SYMS = type_union[GENERAL_PARAM_SYM, PREFIXED_AXIS_SYM, AXIAL_PARAM_SYM]

TRANSSYM_TYPE = type_union[
    TranslationSymmetryX2D,
    TranslationSymmetryY2D,
    TranslationSymmetryX3D,
    TranslationSymmetryY3D,
    TranslationSymmetryZ3D,
    AxialTranslationSymmetry2D,
    AxialTranslationSymmetry3D,
]
ROTSYM_TYPE = type_union[
    RotationSymmetryX3D,
    RotationSymmetryY3D,
    RotationSymmetryZ3D,
    RotationSymmetry2D,
    AxialRotationSymmetry3D,
    RotationSymmetry2D,
]

PRIM_TYPE = type_union[Primitive2D, Primitive3D, HigherOrderPrimitives3D]
HIGERPRIM_TYPE = HigherOrderPrimitives3D

NULL_EXPR_TYPE = type_union[NullExpression2D, NullExpression3D]

# COLOR_TYPE = type_union[ColorModifier2D, ColorModifier3D]
COLOR_MOD = type_union[ModifyColor2D, ModifyOpacity2D]
APPLY_COLOR_TYPE = ApplyColor2D
SVG_COMBINATORS = SVGCombinator
