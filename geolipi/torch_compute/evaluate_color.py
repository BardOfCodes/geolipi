
from sympy import Symbol, Function
from geolipi.symbolic.base_symbolic import GLExpr, GLFunction
import torch as th
import numpy as np
from geolipi.symbolic import Union, Intersection, Difference
from geolipi.symbolic.utils import resolve_macros
from .sketcher import Sketcher
from geolipi.symbolic.utils import MACRO_TYPE, MOD_TYPE, TRANSLATE_TYPE, SCALE_TYPE, PRIM_TYPE, COLOR_TYPE
from .utils import MODIFIER_MAP, PRIMITIVE_MAP
from .utils import COLOR_MAP

# TODO: A recursiver version

def expr_to_colored_canvas(expression: GLExpr, sketcher: Sketcher = None, rectify_transform=False):
    
    transforms_stack = [sketcher.get_affine_identity()]
    execution_stack = []
    execution_pointer_index = []
    if rectify_transform:
        scale_stack = [sketcher.get_scale_identity()]
    parser_list = [expression]
    color_stack = [Symbol("gray")]
    colored_canvas = sketcher.get_color_canvas()
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, MACRO_TYPE):
            new_expr = resolve_macros(cur_expr, device=sketcher.device)
            parser_list.append(new_expr)
        elif isinstance(cur_expr, Union):
            n_args = len(cur_expr.args)
            transform = transforms_stack.pop()
            transform_chain = [transform.clone() for x in range(n_args)]
            transforms_stack.extend(transform_chain)
            color = color_stack.pop()
            color_chain = [Symbol(color.name) for x in range(n_args)]
            color_stack.extend(color_chain)
            if rectify_transform:
                scale = scale_stack.pop()
                scale_chain = [scale.clone() for x in range(n_args)]
                scale_stack.extend(scale_chain)
                
            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
            execution_pointer_index.append(len(execution_stack))
        elif isinstance(cur_expr, COLOR_TYPE):
            color_stack.pop()
            color_stack.append(cur_expr.args[1])
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, MOD_TYPE):
            params = cur_expr.args[1]
            if params in cur_expr.lookup_table:
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
            # At this point use color code to color the primitive
            color = color_stack.pop()
            valid_color = COLOR_MAP[color.name].to(sketcher.device)
            # For differentiable relax, this also has to be relaxed.
            occ = execution <= 0
            # make it alpha blending: https://en.wikipedia.org/wiki/Alpha_compositing
            alpha_a = occ[..., None] * valid_color[0, 3:4]
            color_a = occ.view(occ.shape[0], 1) * valid_color[..., :3]
            alpha_b = colored_canvas[..., 3:4]
            color_b = colored_canvas[..., :3]
            a_o = alpha_a + alpha_b * (1 - alpha_a)
            color_o = (color_a * alpha_a + color_b * alpha_b * (1 - alpha_a)) / a_o
            colored_canvas = th.cat([color_o, a_o], dim=-1)
        elif isinstance(cur_expr, (Intersection, Difference)):
            raise ValueError(f'Cannot use {type(cur_expr)} for coloring')
        else:
            raise ValueError(f'Unknown expression type {type(cur_expr)}')

    return colored_canvas

