from collections import defaultdict
from typing import List
import torch as th
import numpy as np
from geolipi.symbolic.base_symbolic import PrimitiveSpec
from geolipi.symbolic.types import COMBINATOR_TYPE
from .utils import PRIMITIVE_MAP, COMBINATOR_MAP

def create_evaluation_batches(compiled_expr_list: List[object], convert_to_cuda=True):
    batch_limiter = []
    all_expressions = []
    all_prim_transforms = defaultdict(list)
    all_prim_inversion = defaultdict(list)
    all_prim_param = defaultdict(list)

    for compiled_expr in compiled_expr_list:
        expression = compiled_expr[0]
        prim_transform = compiled_expr[1]
        prim_inversion = compiled_expr[2]
        prim_param = compiled_expr[3]
        batch_count = defaultdict(int)
        for prim_type, transforms in prim_transform.items():
            all_prim_transforms[prim_type].append(transforms)
            all_prim_inversion[prim_type].append(prim_inversion[prim_type])
            if prim_type in prim_param.keys():
                all_prim_param[prim_type].append(prim_param[prim_type])
            batch_count[prim_type] = transforms.shape[0]
        all_expressions.append(expression)
        batch_limiter.append(batch_count)
        
    for prim_type in all_prim_transforms.keys():
        if all_prim_transforms[prim_type]:
            all_prim_transforms[prim_type] =  th.cat(all_prim_transforms[prim_type], 0)
            all_prim_inversion[prim_type] = th.cat(all_prim_inversion[prim_type], 0)
            if prim_type in all_prim_param.keys():
                params = all_prim_param[prim_type]
                n_params = len(params[0])
                final_params = []
                for ind in range(n_params):
                    final_params.append(th.cat([x[ind] for x in params], 0))
                all_prim_param[prim_type] = final_params
    if convert_to_cuda:
        all_prim_transforms = {x:y.cuda() for x, y in all_prim_transforms.items()}
        all_prim_inversion = {x:y.cuda() for x, y in all_prim_inversion.items()}

        for prim_type, prim_param_list in all_prim_param.items():
            all_prim_param[prim_type] = [x.cuda() for x in prim_param_list]
        # all_prim_param = {x:y.cuda() for x, y in all_prim_param.items()}

    return all_expressions, all_prim_transforms, all_prim_inversion, all_prim_param, batch_limiter


def batch_evaluate(expr_set: List[object], sketcher):
    """Batch evaluate a set of expressions.

    Args:
        expr_set (List[object]): combine all the compiled exprs into a single list.
        batch_limiter (List[Dict[type, int]]): per expression counter of number of primitives of each type.
        sketcher (Sketcher): The sketcher object which creates the sdfs.

    Returns:
        all_sdfs (th.Tensor): The output sdf for each expression.
    """
    expressions = expr_set[0]
    prim_transforms = expr_set[1]
    prim_inversions = expr_set[2]
    prim_params = expr_set[3]
    batch_limiter = expr_set[4]

    points = sketcher.get_base_coords()

    M = points.shape[0]
    # Add a fourth column of ones to the point cloud to make it homogeneous
    points_hom = th.cat([points, th.ones(M, 1).to(sketcher.device)], dim=1)

    type_wise_primitives = dict()
    for draw_type, transforms in prim_transforms.items():
        if transforms == []:
            continue
        
        cur_points = points_hom.clone()
        # Apply the rotation matrices to the point cloud using einsum
        transformed_points_hom = th.einsum(
            'nij,mj->nmi', transforms, cur_points)
        # Extract the rotated points from the homogeneous coordinates
        rotated_points = transformed_points_hom[:, :, :sketcher.n_dims]

        draw_func = PRIMITIVE_MAP[draw_type]
        params = prim_params[draw_type]
        primitives = draw_func(rotated_points, *params)
        # For some primitives make this a expansion of fixed size.
        # inversion = th.stack(collapsed_inversions[draw_type], 0).unsqueeze(1)
        inversion = prim_inversions[draw_type].unsqueeze(1)
        sign_matrix = inversion * -2 + 1
        primitives = primitives * sign_matrix
        type_wise_primitives[draw_type] = primitives

    # next allot the primitive sequentially:
    # calculate offset for each graph
    type_wise_draw_count = defaultdict(int)
    all_sdfs = []
    for ind, expression in enumerate(expressions):
        if expression is None:
            sdf = sketcher.empty_sdf()
        else:
            sdf = execute_compiled_expression(
                expression, type_wise_primitives, type_wise_draw_count)
        all_sdfs.append(sdf)
        draw_specs = batch_limiter[ind]
        for draw_type in type_wise_draw_count.keys():
            type_wise_draw_count[draw_type] += draw_specs[draw_type]
        
    if all_sdfs:
        all_sdfs = th.stack(all_sdfs, 0)

    return all_sdfs



def execute_compiled_expression(expression, type_wise_primitives, type_wise_draw_count):
    execution_stack = []
    operator_stack = []
    operator_nargs_stack = []
    execution_pointer_index = []
    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, COMBINATOR_TYPE):
            operator_stack.append(type(cur_expr))
            n_args = len(cur_expr.args)
            operator_nargs_stack.append(n_args)
            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
            execution_pointer_index.append(len(execution_stack))
        elif isinstance(cur_expr, PrimitiveSpec):
            c_symbol = cur_expr.args[0]
            shift = cur_expr.args[1]
            idx = shift + type_wise_draw_count[c_symbol]
            execution = type_wise_primitives[c_symbol][idx]
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

