from typing import List
import re
import numpy as np
import torch as th
from sympy import Symbol
import sympy
from geolipi.symbolic.primitives_2d import NoParamRectangle2D, NoParamCircle2D, NoParamTriangle2D
from geolipi.symbolic.combinators import Union, Intersection, Difference
from geolipi.symbolic.transforms_2d import Translate2D, Scale2D, ColorTree2D, TranslationSymmetry2D, RotationSymmetry2D, AxialReflect2D

BOOL_MAP = {
    'union': Union,
}
PRIM_MAP = {
    'circle': NoParamCircle2D,
    'square': NoParamRectangle2D,
    'triangle': NoParamTriangle2D,
}
MODIFIERS = {
    'scale': Scale2D,
    'move': Translate2D,
    'color': ColorTree2D,
    'symTranslate': TranslationSymmetry2D,
    'symRotate': RotationSymmetry2D,
    'symReflect': AxialReflect2D,
}

def str_to_expr(expression_string_list: List[str], to_cuda=False, dtype=th.float32):
    """Parse a list of expression strings into a geolipi expression.
    """
    cur_expr = expression_string_list[0]
    while(True):
        if cur_expr == "START":
            expr, remaining = str_to_expr(expression_string_list[1:], to_cuda=to_cuda, dtype=dtype)
            return expr
        elif cur_expr in MODIFIERS.keys():
            expr, remaining = str_to_expr(expression_string_list[1:], to_cuda=to_cuda, dtype=dtype)
            if cur_expr in ['scale', 'move']:
                if to_cuda:
                    params = th.tensor([float(x.strip()) for x in remaining[:2]],
                                    dtype=dtype, device='cuda')
                else:
                    params = np.array([float(x.strip()) for x in remaining[:2]])
                param_count = 2
                main_expr = MODIFIERS[cur_expr](expr, params)
            elif cur_expr in ['color']:
                params = Symbol(remaining[0].strip().upper())
                param_count = 1
                main_expr = MODIFIERS[cur_expr](expr, params)
            elif cur_expr == 'symTranslate':
                if to_cuda:
                    param_1 = th.tensor([float(x.strip()) for x in remaining[:2]],
                                    dtype=dtype, device='cuda')
                else:
                    params_1 = np.array([float(x.strip()) for x in remaining[:2]])
                param_2 = int(remaining[2].strip()) + 1
                param_count = 3
                main_expr = MODIFIERS[cur_expr](expr, param_1, param_2)
            elif cur_expr == 'symRotate':
                if to_cuda:
                    param_1 = th.tensor([float(x.strip()) for x in remaining[:1]],
                                    dtype=dtype, device='cuda')
                else:
                    param_1 = np.array([float(x.strip()) for x in remaining[:1]])
                param_2 = int(remaining[1].strip()) + 1
                param_count = 2
                main_expr = MODIFIERS[cur_expr](expr, param_1, param_2)
            elif cur_expr == 'symReflect':
                params = Symbol(remaining[0].strip().upper()+"2D")
                param_count = 1
                main_expr = MODIFIERS[cur_expr](expr, params)
                
            return main_expr, remaining[param_count:]
        elif cur_expr in PRIM_MAP.keys():
            main_expr = PRIM_MAP[cur_expr]()
            return main_expr, expression_string_list[1:]
        elif cur_expr in BOOL_MAP.keys():
            expr_1, remaining = str_to_expr(expression_string_list[1:], to_cuda=to_cuda, dtype=dtype)
            expr_2, remaining = str_to_expr(remaining, to_cuda=to_cuda, dtype=dtype)
            cmd = BOOL_MAP[cur_expr](expr_1, expr_2)
            return cmd, remaining
        else:
            raise ValueError(f'Unknown command {cur_expr}')
