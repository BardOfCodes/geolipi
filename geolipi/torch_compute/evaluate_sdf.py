
from sympy import Symbol, Function
from geolipi.symbolic.base_symbolic import GLExpr, GLFunction
import torch as th
import numpy as np
from geolipi.symbolic import Combinator
from geolipi.symbolic.utils import resolve_macros
from geolipi.symbolic.utils import MACRO_TYPE, MOD_TYPE, TRANSLATE_TYPE, SCALE_TYPE, PRIM_TYPE
from .sketcher import Sketcher
from .utils import MODIFIER_MAP, PRIMITIVE_MAP, COMBINATOR_MAP

# TODO: A recursiver version

def expr_to_sdf(expression: GLExpr, sketcher: Sketcher = None, rectify_transform=True):
    transforms_stack = [sketcher.get_affine_identity()]
    execution_stack = []
    operator_stack = []
    operator_nargs_stack = []
    execution_pointer_index = []
    if rectify_transform:
        scale_stack = [sketcher.get_scale_identity()]
    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, MACRO_TYPE):
            new_expr = resolve_macros(cur_expr, device=sketcher.device)
            parser_list.append(new_expr)
        elif isinstance(cur_expr, Combinator):
            operator_stack.append(type(cur_expr))
            n_args = len(cur_expr.args)
            operator_nargs_stack.append(n_args)
            transform = transforms_stack.pop()
            transform_chain = [transform.clone() for x in range(n_args)]
            transforms_stack.extend(transform_chain)
            if rectify_transform:
                scale = scale_stack.pop()
                scale_chain = [scale.clone() for x in range(n_args)]
                scale_stack.extend(scale_chain)
            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
            execution_pointer_index.append(len(execution_stack))
        elif isinstance(cur_expr, MOD_TYPE):
            params = cur_expr.args[1]
            if isinstance(params, Symbol):
                params = cur_expr.lookup_table[params]
            if rectify_transform:
                if isinstance(cur_expr, TRANSLATE_TYPE):
                    scale = scale_stack[-1]
                    params = params / scale
                elif isinstance(cur_expr, SCALE_TYPE):
                    scale_stack[-1] *= params
            transform = transforms_stack.pop()
            identity_mat = sketcher.get_affine_identity()
            new_transform = MODIFIER_MAP[type(cur_expr)](identity_mat, params)
            transform = th.matmul(new_transform, transform)
            transforms_stack.append(transform)
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
            transform = transforms_stack.pop()
            if rectify_transform:
                _ = scale_stack.pop()
            coords = sketcher.get_coords(transform)
            params = cur_expr.args
            if params:
                params = params[0]
                if isinstance(params, Symbol):
                    params = cur_expr.lookup_table[params]
            execution = PRIMITIVE_MAP[type(cur_expr)](coords, params)
            execution_stack.append(execution)
        else:
            raise ValueError(f'Unknown expression type {type(cur_expr)}')

        while (operator_stack and len(execution_stack) - execution_pointer_index[-1] >= operator_nargs_stack[-1]):
            n_args = operator_nargs_stack.pop()
            operator = operator_stack.pop()
            _ = execution_pointer_index.pop()
            args = execution_stack[-n_args:]
            new_canvas = COMBINATOR_MAP[operator](*args)
            execution_stack = execution_stack[:-n_args] + [new_canvas]

    assert len(execution_stack) == 1
    sdf = execution_stack[0]
    return sdf
