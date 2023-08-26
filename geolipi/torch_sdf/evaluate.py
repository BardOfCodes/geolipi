

from typing import Dict, List, Tuple, Union as type_union
from sympy import Expr, Symbol, Function
import inspect
import torch as th
import numpy as np
import rustworkx as rx
from collections import defaultdict
from geolipi.symbolic import Primitive3D, Combinator, Transform, Modifier
from geolipi.symbolic.primitives_3d import Cuboid, Sphere, Cylinder, Cone, NoParamSphere, NoParamCylinder, NoParamCuboid
from geolipi.symbolic.combinators import Union, Intersection, Difference, Complement
from geolipi.symbolic.transforms import Translate, EulerRotate, QuaternionRotate, Scale
from .sketcher import Sketcher
from .transforms import get_affine_translate, get_affine_scale, get_affine_rotate_euler
from .sdf_functions import sdf_cuboid, sdf_sphere, sdf_cylinder, no_param_cuboid, no_param_cylinder, no_param_sphere
from .sdf_functions import sdf_union, sdf_intersection, sdf_difference, sdf_complement

# TODO: Clean the usage of primitives with params.

PARAM_TYPE = type_union[np.ndarray, th.Tensor]
MODIFIER_MAP = {
    Translate: get_affine_translate,
    EulerRotate: get_affine_rotate_euler,
    Scale: get_affine_scale,
}
PRIMITIVE_MAP = {
    Cuboid: sdf_cuboid,
    Sphere: sdf_sphere,
    Cylinder: sdf_cylinder,
    NoParamCuboid: no_param_cuboid,
    NoParamSphere: no_param_sphere,
    NoParamCylinder: no_param_cylinder,
}

COMBINATOR_MAP = {
    Union: sdf_union,
    Intersection: sdf_intersection,
    Difference: sdf_difference,
    Complement: sdf_complement,
}

# Fore resolution:

INVERTED_MAP = {
    Union: Intersection,
    Intersection: Union,
    Difference: Union,
}
NORMAL_MAP = {
    Union: Union,
    Intersection: Intersection,
    Difference: Intersection,
}
# ONLY_SIMPLIFY_RULES = set(["ii", "uu"])

ONLY_SIMPLIFY_RULES = set([(Intersection, Intersection), (Union, Union)])
ALL_RULES = set([(Intersection, Intersection), (Union, Union), (Intersection, Union)])

class PrimitiveSpec(Function):
    @classmethod
    def eval(cls, prim_type: type, shift: int):
        return None

def expr_to_sdf(*args, mode: str = 'naive', **kawrgs):
    if mode == 'naive':
        return naive_expr_to_sdf(*args, **kawrgs)
    elif mode == 'fast':
        compiled_obj = fast_compile(*args, **kawrgs)
        return fast_execute(compiled_obj)
    else:
        raise ValueError(f'Unknown mode {mode}')

# TODO: Create a recursive version. Simpler to understand.

# polish notation stack based parser.
def naive_expr_to_sdf(expression: Expr, var_dict: Dict[Expr, PARAM_TYPE] = None,
                      sketcher: Sketcher = None, rectify_transform=False):
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
        if isinstance(cur_expr, Combinator):
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
        elif isinstance(cur_expr, Modifier):
            params = cur_expr.args[1]
            if isinstance(params, Symbol):
                params = var_dict[params.name]
            if rectify_transform:
                if isinstance(cur_expr, Translate):
                    scale = scale_stack[-1]
                    params = params / scale
                elif isinstance(cur_expr, Scale):
                    scale_stack[-1] *= params
            transform = transforms_stack.pop()
            identity_mat = sketcher.get_affine_identity()
            new_transform = MODIFIER_MAP[type(cur_expr)](identity_mat, params)
            transform = th.matmul(new_transform, transform)
            transforms_stack.append(transform)
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, Primitive3D):
            transform = transforms_stack.pop()
            if rectify_transform:
                _ = scale_stack.pop()
            coords = sketcher.get_coords(transform)
            params = cur_expr.args
            if params:
                if isinstance(params, Symbol):
                    params = var_dict[params.name]
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


def expr_prim_count(expression: Expr):
    """Get the number of primitives of each kind in the expression."""
    prim_count_dict = defaultdict(int)
    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, Combinator):
            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
        elif isinstance(cur_expr, Modifier):
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, Primitive3D):
            prim_count_dict[type(cur_expr)] += 1
        else:
            raise ValueError(f'Unknown expression type {type(cur_expr)}')
    return prim_count_dict


def compile_expr(expression: Expr, var_dict: Dict[Expr, PARAM_TYPE], prim_count_dict: Dict[str, int],
                                 sketcher: Sketcher=None, rectify_transform=False):
    """Gather & Compute the transforms, Gather the primitive_params, Remove Difference, and Resolve Complement,"""

    # trasforms
    transforms_stack = [sketcher.get_affine_identity()]
    # inversions
    inversion_mode = False
    inversion_stack = [inversion_mode]

    prim_transforms = dict()
    prim_inversions = dict()
    prim_params = dict()
    prim_counter = {x:0 for x in prim_count_dict.keys()}
    for prim_type, prim_count in prim_count_dict.items():
        prim_transforms[prim_type] = th.zeros((prim_count, 4, 4),
            dtype=sketcher.dtype, device=sketcher.device)
        prim_inversions[prim_type] = th.zeros((prim_count),
            dtype=th.bool, device=sketcher.device)
        n_params = len(inspect.signature(prim_type).parameters)
        if n_params > 0:
            prim_params[prim_type] = th.zeros((prim_count, n_params),
                dtype=sketcher.dtype, device=sketcher.device)

    execution_stack = []
    execution_pointer_index = []
    operator_stack = []
    operator_nargs_stack = []
    if rectify_transform:
        scale_stack = [sketcher.get_scale_identity()]

    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        inversion_mode = inversion_stack.pop()
        if isinstance(cur_expr, Combinator):
            n_args = len(cur_expr.args)
            transform = transforms_stack.pop()
            transform_chain = [transform.clone() for x in range(n_args)]
            transforms_stack.extend(transform_chain)
            if rectify_transform:
                scale = scale_stack.pop()
                scale_chain = [scale.clone() for x in range(n_args)]
                scale_stack.extend(scale_chain)

            if type(cur_expr) == Difference:
                inversion_stack.append(not inversion_mode)
            else:
                inversion_stack.append(inversion_mode)

            inversion_stack.extend([inversion_mode for x in range(n_args - 1)])

            if inversion_mode:
                current_symbol = INVERTED_MAP[type(cur_expr)]
            else:
                current_symbol = NORMAL_MAP[type(cur_expr)]
            operator_stack.append(current_symbol)
            operator_nargs_stack.append(n_args)

            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
            execution_pointer_index.append(len(execution_stack))
        elif isinstance(cur_expr, Modifier):
            params = cur_expr.args[1]
            if isinstance(params, Symbol):
                params = var_dict[params.name]
            if rectify_transform:
                if isinstance(cur_expr, Translate):
                    scale = scale_stack[-1]
                    params = params / scale
                elif isinstance(cur_expr, Scale):
                    scale_stack[-1] *= params
            transform = transforms_stack.pop()
            identity_mat = sketcher.get_affine_identity()
            new_transform = MODIFIER_MAP[type(cur_expr)](identity_mat, params)
            transform = th.matmul(new_transform, transform)
            transforms_stack.append(transform)
            inversion_stack.append(inversion_mode)
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, Primitive3D):
            transform = transforms_stack.pop()
            _ = scale_stack.pop()
            prim_type = type(cur_expr)
            prim_id = prim_counter[prim_type]
            prim_transforms[prim_type][prim_id] = transform
            if inversion_mode:
                prim_inversions[prim_type][prim_id] = True
            params = cur_expr.args
            if params:
                if isinstance(params, Symbol):
                    params = var_dict[params.name]
                prim_params[prim_type][prim_id] = params

            prim_spec = PrimitiveSpec(prim_type, prim_id)
            execution_stack.append(prim_spec)
            prim_counter[prim_type] += 1
            
        else:
            raise ValueError(f'Unknown expression type {type(cur_expr)}')

        while (operator_stack and len(execution_stack) - execution_pointer_index[-1] >= operator_nargs_stack[-1]):
            n_args = operator_nargs_stack.pop()
            operator = operator_stack.pop()
            _ = execution_pointer_index.pop()
            args = execution_stack[-n_args:]
            new_canvas = operator(*args)
            execution_stack = execution_stack[:-n_args] + [new_canvas]

    expression = execution_stack[0]
    return expression, prim_transforms, prim_inversions, prim_params

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
    # concatenate
    for prim_type in prim_transform.keys():
        if convert_to_cuda:
            np_array = np.concatenate(all_prim_transforms[prim_type], 0)
            all_prim_transforms[prim_type] = th.from_numpy(np_array).cuda()
            np_array = np.concatenate(all_prim_inversion[prim_type], 0)
            all_prim_inversion[prim_type] = th.from_numpy(np_array).cuda()
            if prim_type in prim_param.keys():
                np_array = np.concatenate(all_prim_param[prim_type], 0)
                all_prim_param[prim_type] = th.from_numpy(np_array).cuda()
            
    return all_expressions, all_prim_transforms, all_prim_inversion, all_prim_param, batch_limiter

def batch_evaluate(expr_set: List[object], sketcher: Sketcher):
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
        if draw_type in prim_params.keys():
            params = prim_params[draw_type]
            primitives = draw_func(rotated_points, params)
        else:
            primitives = draw_func(rotated_points, None)
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
        sdf = execute_compiled_expression(
            expression, type_wise_primitives, type_wise_draw_count)
        all_sdfs.append(sdf)
        draw_specs = batch_limiter[ind]
        for draw_type in type_wise_draw_count.keys():
            type_wise_draw_count[draw_type] += draw_specs[draw_type]
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
        if isinstance(cur_expr, Combinator):
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

def expr_to_graph(expression):
    """Convert sympy expression to a rustworkx graph.
    Note it is only valid for compiled graph."""

    graph = rx.PyDiGraph()
    parser_list = [expression]
    parent_ids = [None]
    while(parser_list):
        cur_expr = parser_list.pop()
        parent_id = parent_ids.pop()
        if isinstance(cur_expr, Combinator):
            node = dict(type=type(cur_expr))
            cur_id = graph.add_node(node)
            parser_list.extend(cur_expr.args[::-1])
            parent_ids.extend([cur_id for x in range(len(cur_expr.args))])
        elif isinstance(cur_expr, PrimitiveSpec):
            node = dict(type=type(cur_expr), expr=cur_expr)
            cur_id = graph.add_node(node)
        if not parent_id is None:
            graph.add_edge(parent_id, cur_id, None)
    return graph

def graph_to_expr(graph):
    """Convert a rustworkx graph to a sympy expression."""
    
    prim_stack = []
    prim_stack_pointer = []
    operator_stack = []
    operator_nargs_stack = []
    
    parser_list = [0]
    while(parser_list):
        cur_id = parser_list.pop()
        cur_node = graph[cur_id]
        c_type = cur_node['type']
        if c_type == PrimitiveSpec:
            prim_stack.append(cur_node['expr'])
        elif issubclass(c_type, Combinator):
            operator_stack.append(c_type)
            next_to_parse = list(graph.successor_indices(cur_id))
            n_args = len(next_to_parse)
            operator_nargs_stack.append(n_args)
            parser_list.extend(next_to_parse)
            prim_stack_pointer.append(len(prim_stack))
        
        while (operator_stack and len(prim_stack) - prim_stack_pointer[-1] >= operator_nargs_stack[-1]):
            n_args = operator_nargs_stack.pop()
            operator = operator_stack.pop()
            _ = prim_stack_pointer.pop()
            args = prim_stack[-n_args:]
            new_canvas = operator(*args)
            prim_stack = prim_stack[:-n_args] + [new_canvas]
    expression = prim_stack[0]
    return expression


def expr_to_dnf(expression):
    # convert to tree for easy usage:
    graph = expr_to_graph(expression)
    # RULES: 
    # Intersection-> Intersection = collapse
    # Union -> union = collapse
    # Intersection -> Union = invert
    # Union -> Intersection = retain
    rule_match = None
    while(True):
        rule_match = get_rule_match(graph, only_simplify=True)
        if rule_match is None:
            rule_match = get_rule_match(graph, only_simplify=False)
            if rule_match is None:
                break
            else:
                graph = resolve_rule(graph, rule_match)
        else:
            graph = resolve_rule(graph, rule_match)
    expression = graph_to_expr(graph)
    return expression

def get_rule_match(graph, only_simplify=False):
    if only_simplify:
        rule_set = ONLY_SIMPLIFY_RULES# ["i->i", "u->u"]
    else:
        rule_set = ALL_RULES#  ["i->i", "u->u", "i->u"]
    node_ids = [0]
    rule_match = None
    while node_ids:
        cur_id = node_ids.pop()
        cur_node = graph[cur_id]
        c_type = cur_node['type']
        if issubclass(c_type, Combinator):
            children = list(graph.successor_indices(cur_id))
            node_ids.extend(children[::-1])
            for ind, child_id in enumerate(children):
                child_node = graph[child_id]
                child_type = child_node['type']
                rel_sig = (c_type, child_type)
                if rel_sig in rule_set:
                    rule_match = (cur_id, child_id, rel_sig)
        if rule_match:
            break

    return rule_match

def resolve_rule(graph, resolve_rule):
    
    node_a_id = resolve_rule[0]
    node_b_id = resolve_rule[1]
    match_type = resolve_rule[2]
    node_a = graph[node_a_id]
        
    if match_type in ONLY_SIMPLIFY_RULES:
        # append the children of the child node to the parent node
        for child_id in graph.successor_indices(node_b_id):
            graph.add_edge(node_a_id, child_id, None)
        graph.remove_edge(node_a_id, node_b_id)
        # graph.remove_node(node_b_id)
    elif match_type == (Intersection, Union):
        node_a['type'] = Union
        children_a = list(graph.successor_indices(node_a_id))
        children_a_not_b = [ind for ind in children_a if ind != node_b_id]
        children_b = list(graph.successor_indices(node_b_id))
        for child_id in children_a:
            graph.remove_edge(node_a_id, child_id)
        for child_id in children_b:
            # create n new i nodes where n is the number of children of the u node
            node = dict(type=Intersection)
            new_id = graph.add_node(node)
            graph.add_edge(new_id, child_id, None)
            for not_b_child_id in children_a_not_b:
                graph.add_edge(new_id, not_b_child_id, None)
            graph.add_edge(node_a_id, new_id, None)
            # where each node has all the children of the i node <not U> and one child of the u node
            
    return graph
            
