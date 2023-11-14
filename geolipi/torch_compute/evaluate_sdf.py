
from sympy import Symbol, Function
from geolipi.symbolic.base_symbolic import GLExpr, GLFunction
import torch as th
import numpy as np
from geolipi.symbolic import Combinator
from geolipi.symbolic.resolve import resolve_macros
from geolipi.symbolic.types import (MACRO_TYPE, MOD_TYPE, TRANSLATE_TYPE, SCALE_TYPE, PRIM_TYPE, TRANSSYM_TYPE,
                                    TRANSFORM_TYPE, POSITIONALMOD_TYPE, SDFMOD_TYPE)
from .sketcher import Sketcher
from .utils import MODIFIER_MAP, PRIMITIVE_MAP, COMBINATOR_MAP

# TODO: A recursiver version

def recursive_expr_to_sdf(expression, sketcher, initialize=True, rectify_transform=False,
                          affine_transform=None, tracked_scale=None):
    if initialize:
        affine_transform = sketcher.get_affine_identity()
        if rectify_transform:
            tracked_scale = sketcher.get_scale_identity()
    if isinstance(expression, MACRO_TYPE):
        resolved_expr = resolve_macros(expression, device=sketcher.device)
        return recursive_expr_to_sdf(resolved_expr, sketcher, initialize=False, rectify_transform=rectify_transform,
                                     affine_transform=affine_transform, tracked_scale=tracked_scale)
    elif isinstance(expression, Combinator):
        # what about parameterized combinators?
        tree_branches, cur_params = [], []
        for arg in expression.args:
            if arg in expression.lookup_table:
                cur_params.append(expression.lookup_table[arg])
            else:
                tree_branches.append(arg)
        n_args = len(tree_branches)
        sdf_list = []
        for child in tree_branches:
            cur_sdf = recursive_expr_to_sdf(child, sketcher, initialize=False,
                                                                    rectify_transform=rectify_transform,
                                                                    affine_transform=affine_transform.clone(),
                                                                    tracked_scale=tracked_scale.clone())
            sdf_list.append(cur_sdf)
        new_sdf = COMBINATOR_MAP[type(expression)](*sdf_list, *cur_params)
        return new_sdf
    elif isinstance(expression, MOD_TYPE):
        sub_expr = expression.args[0]

        params = expression.args[1:]
        if params:
            for ind, param in enumerate(params):
                if param in expression.lookup_table:
                    params[ind] = expression.lookup_table[param]
        # This is a hack unclear how to deal with other types)
        if isinstance(expression, TRANSFORM_TYPE):
            if rectify_transform:
                if isinstance(expression, (TRANSLATE_TYPE, TRANSSYM_TYPE)):
                    params[0] = params[0] / tracked_scale
                elif isinstance(expression, SCALE_TYPE):
                    tracked_scale *= params[0]
            identity_mat = sketcher.get_affine_identity()
            new_transform = MODIFIER_MAP[type(expression)](identity_mat, *params)
            affine_transform = th.matmul(new_transform, affine_transform)
            # Here we need to 
            return recursive_expr_to_sdf(sub_expr, sketcher, initialize=False, rectify_transform=rectify_transform,
                                            affine_transform=affine_transform, tracked_scale=tracked_scale)
        elif isinstance(expression, POSITIONALMOD_TYPE):
            # instantiate positions and send that as input with affine set to None
            ...
        elif isinstance(expression, SDFMOD_TYPE):
            # calculate sdf then create change before returning.
    elif isinstance(expression, PRIM_TYPE):
        # create sdf and return.
        ...
    
    

        

def expr_to_sdf(expression: GLExpr, sketcher: Sketcher = None, rectify_transform=True):
    transforms_stack = [sketcher.get_affine_identity()]
    execution_stack = []
    operator_stack = []
    operator_nargs_stack = []
    operator_params_stack = []
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
            # what about parameterized combinators?
            tree_branches, cur_params = [], []
            for arg in cur_expr.args:
                if arg in cur_expr.lookup_table:
                    cur_params.append(cur_expr.lookup_table[arg])
                else:
                    tree_branches.append(arg)
            n_args = len(tree_branches)
            operator_nargs_stack.append(n_args)
            operator_params_stack.append(cur_params)
            transform = transforms_stack.pop()
            transform_chain = [transform.clone() for x in range(n_args)]
            transforms_stack.extend(transform_chain)
            if rectify_transform:
                scale = scale_stack.pop()
                scale_chain = [scale.clone() for x in range(n_args)]
                scale_stack.extend(scale_chain)
            next_to_parse = tree_branches[::-1]
            parser_list.extend(next_to_parse)
            execution_pointer_index.append(len(execution_stack))
        elif isinstance(cur_expr, MOD_TYPE):
            params = cur_expr.args[1:]
            if params:
                for ind, param in enumerate(params):
                    if param in cur_expr.lookup_table:
                        params[ind] = cur_expr.lookup_table[param]
            # This is a hack unclear how to deal with other types)
            if rectify_transform:
                if isinstance(cur_expr, TRANSLATE_TYPE):
                    scale = scale_stack[-1]
                    params[0] = params[0] / scale
                elif isinstance(cur_expr, SCALE_TYPE):
                    scale_stack[-1] *= params[0]
                elif isinstance(cur_expr, TRANSSYM_TYPE):
                    scale = scale_stack[-1]
                    params[0] = params[0] / scale

            transform = transforms_stack.pop()
            identity_mat = sketcher.get_affine_identity()
            new_transform = MODIFIER_MAP[type(cur_expr)](identity_mat, *params)
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
                for ind, param in enumerate(params):
                    if param in cur_expr.lookup_table:
                        params[ind] = cur_expr.lookup_table[param]
            execution = PRIMITIVE_MAP[type(cur_expr)](coords, *params)
            execution_stack.append(execution)
        else:
            raise ValueError(f'Unknown expression type {type(cur_expr)}')

        while (operator_stack and len(execution_stack) - execution_pointer_index[-1] >= operator_nargs_stack[-1]):
            n_args = operator_nargs_stack.pop()
            operator = operator_stack.pop()
            _ = execution_pointer_index.pop()
            params = operator_params_stack.pop()
            args = execution_stack[-n_args:]
            new_canvas = COMBINATOR_MAP[operator](*args, *params)
            execution_stack = execution_stack[:-n_args] + [new_canvas]

    assert len(execution_stack) == 1
    sdf = execution_stack[0]
    return sdf

