"""
Deprecated evaluation functions.

These functions are kept for backward compatibility but are not recommended
for new code. Use recursive_evaluate from evaluate_expression instead.

Functions:
- expr_to_sdf: Stack-based SDF evaluation (limited feature support)
- expr_to_colored_canvas: Stack-based colored canvas evaluation (experimental)
"""

import torch as th
from .sketcher import Sketcher
from .maps import MODIFIER_MAP, PRIMITIVE_MAP, COMBINATOR_MAP, COLOR_MAP
from geolipi.symbolic.base import GLExpr, GLFunction


def _parse_param_from_expr_simple(expression, params):
    """Simplified parameter parsing for deprecated functions."""
    if params:
        param_list = []
        for ind, param in enumerate(params):
            if param in expression.lookup_table:
                cur_param = expression.lookup_table[param]
                param_list.append(cur_param)
            else:
                param_list.append(param)
        params = param_list
    return params


def expr_to_sdf(
    expression: GLFunction,
    sketcher: Sketcher,
    secondary_sketcher: Sketcher = None,
    rectify_transform: bool = False,
    coords: th.Tensor = None,
):
    """
    Converts a GeoLIPI SDF expression into a Signed Distance Field (SDF) using a sketcher.
    
    **DEPRECATED**: Use recursive_evaluate from evaluate_expression instead.
    
    This function is faster than `recursive_evaluate` as it evaluates the expression using a stack-based approach. 
    However, it does not support all GeoLIPI operations, notably higher-order primitives, and certain modifiers. 

    Parameters:
        expression (GLFunction): The GLFunction expression to be converted to an SDF.
        sketcher (Sketcher): The primary sketcher object used for generating SDFs.
        rectify_transform (bool): Flag to apply rectified transformations. Defaults to False.
        secondary_sketcher (Sketcher, optional): Secondary sketcher - Never used.
        coords (Tensor, optional): Custom coordinates to use for the SDF generation.

    Returns:
        Tensor: The generated SDF corresponding to the input expression.
    """
    from geolipi.symbolic.resolve import resolve_macros
    from geolipi.symbolic.symbol_types import (
        MACRO_TYPE, MOD_TYPE, TRANSLATE_TYPE, SCALE_TYPE, PRIM_TYPE, 
        TRANSSYM_TYPE, COMBINATOR_TYPE
    )
    
    transforms_stack = [sketcher.get_affine_identity()]
    execution_stack = []
    operator_stack = []
    operator_nargs_stack = []
    operator_params_stack = []
    execution_pointer_index = []
    if rectify_transform:
        scale_stack = [sketcher.get_scale_identity()]
    parser_list = [expression]
    while parser_list:
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, MACRO_TYPE):
            new_expr = resolve_macros(cur_expr, device=sketcher.device)
            parser_list.append(new_expr)
        elif isinstance(cur_expr, COMBINATOR_TYPE):
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
            params = _parse_param_from_expr_simple(cur_expr, params)
            # This is a hack unclear how to deal with other types)
            if rectify_transform:
                if isinstance(cur_expr, (TRANSLATE_TYPE, TRANSSYM_TYPE)):
                    scale = scale_stack[-1]
                    params[0] = params[0] / scale
                elif isinstance(cur_expr, SCALE_TYPE):
                    scale_stack[-1] *= params[0]

            transform = transforms_stack.pop()
            identity_mat = sketcher.get_affine_identity()
            new_transform = MODIFIER_MAP[type(cur_expr)](identity_mat, *params)
            transform = th.matmul(new_transform, transform)
            transforms_stack.append(transform)
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
            params = cur_expr.args
            params = _parse_param_from_expr_simple(cur_expr, params)
            transform = transforms_stack.pop()
            if rectify_transform:
                _ = scale_stack.pop()
            cur_coords = sketcher.get_coords(transform, points=coords)
            execution = PRIMITIVE_MAP[type(cur_expr)](cur_coords, *params)
            execution_stack.append(execution)
        else:
            raise ValueError(f"Unknown expression type {type(cur_expr)}")

        while (
            operator_stack
            and len(execution_stack) - execution_pointer_index[-1]
            >= operator_nargs_stack[-1]
        ):
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


def expr_to_colored_canvas(
    expression: GLExpr,
    sketcher: Sketcher,
    rectify_transform=False,
    relaxed_occupancy=False,
    relax_temperature=0.0,
    coords=None,
    canvas=None,
):
    """
    **DEPRECATED**: Use recursive_evaluate from evaluate_expression instead.
    
    TODO: This function is to be tested.
    """
    from sympy import Symbol
    from geolipi.symbolic.resolve import resolve_macros
    from geolipi.symbolic.symbol_types import (
        MACRO_TYPE, MOD_TYPE, TRANSLATE_TYPE, SCALE_TYPE, PRIM_TYPE, 
        TRANSSYM_TYPE, COMBINATOR_TYPE, APPLY_COLOR_TYPE
    )
    
    transforms_stack = [sketcher.get_affine_identity()]
    execution_stack = []
    execution_pointer_index = []
    if rectify_transform:
        scale_stack = [sketcher.get_scale_identity()]
    parser_list = [expression]
    color_stack = [Symbol("gray")]
    if canvas is None:
        colored_canvas = sketcher.get_color_canvas()
    else:
        colored_canvas = canvas
    while parser_list:
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, MACRO_TYPE):
            new_expr = resolve_macros(cur_expr, device=sketcher.device)
            parser_list.append(new_expr)
        elif isinstance(cur_expr, COMBINATOR_TYPE):
            n_args = len(cur_expr.args)
            # chain extensions
            transform = transforms_stack.pop()
            transform_chain = [transform.clone() for x in range(n_args)]
            transforms_stack.extend(transform_chain)
            color = color_stack.pop()
            if isinstance(color, th.Tensor):
                color_chain = [color.clone() for x in range(n_args)]
            else:
                color_chain = [color for x in range(n_args)]
            color_stack.extend(color_chain)
            if rectify_transform:
                scale = scale_stack.pop()
                scale_chain = [scale.clone() for x in range(n_args)]
                scale_stack.extend(scale_chain)

            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
            execution_pointer_index.append(len(execution_stack))
        elif isinstance(cur_expr, APPLY_COLOR_TYPE):
            color_stack.pop()
            color_stack.append(cur_expr.args[1])
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, MOD_TYPE):
            params = expression.args[1:]
            params = _parse_param_from_expr_simple(expression, params)
            if rectify_transform:
                if isinstance(expression, (TRANSLATE_TYPE, TRANSSYM_TYPE)):
                    params[0] = params[0] / scale_stack[-1]
                elif isinstance(expression, SCALE_TYPE):
                    scale_stack[-1] *= params[0]
            transform = transforms_stack.pop()
            identity_mat = sketcher.get_affine_identity()
            new_transform = MODIFIER_MAP[type(cur_expr)](identity_mat, params)
            transform = th.matmul(new_transform, transform)
            transforms_stack.append(transform)
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
            params = cur_expr.args
            params = _parse_param_from_expr_simple(cur_expr, params)
            transform = transforms_stack.pop()
            if rectify_transform:
                _ = scale_stack.pop()
            cur_coords = sketcher.get_coords(transform, points=coords)
            execution = PRIMITIVE_MAP[type(cur_expr)](cur_coords, *params)
            # At this point use color code to color the primitive
            color = color_stack.pop()
            if isinstance(color, Symbol):
                valid_color = COLOR_MAP[color.name].to(sketcher.device)
            else:
                valid_color = color
            # For differentiable relax, this also has to be relaxed.

            if relaxed_occupancy:
                # from the sdf execution compute occupancy
                occ = relaxed_occupancy(execution, temperature=relax_temperature)
            else:
                occ = execution <= 0
            # Amazing source: https://ciechanow.ski/alpha-compositing/
            alpha_a = occ[..., None] * valid_color[0, 3:4]
            color_a = occ.view(occ.shape[0], 1) * valid_color[..., :3]
            alpha_b = colored_canvas[..., 3:4]
            color_b = colored_canvas[..., :3]
            a_o = alpha_a + alpha_b * (1 - alpha_a)
            color_o = (color_a * alpha_a + color_b * alpha_b * (1 - alpha_a)) / a_o
            colored_canvas = th.cat([color_o, a_o], dim=-1)
        else:
            raise ValueError(f"Unknown expression type {type(cur_expr)}")

    return colored_canvas
