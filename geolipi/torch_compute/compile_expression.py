

from typing import Dict, List, Tuple, Union as type_union
from collections import defaultdict
import numpy as np
import torch as th
import inspect
from sympy import Symbol
import rustworkx as rx
from geolipi.symbolic import Combinator, Difference, Intersection, Union
from geolipi.symbolic.base_symbolic import GLExpr
from geolipi.symbolic.utils import MACRO_TYPE, MOD_TYPE, TRANSLATE_TYPE, SCALE_TYPE, PRIM_TYPE, resolve_macros
from .sketcher import Sketcher
from .utils import MODIFIER_MAP
from .utils import INVERTED_MAP, NORMAL_MAP, PrimitiveSpec, ONLY_SIMPLIFY_RULES, ALL_RULES 

def expr_prim_count(expression: GLExpr):
    """Get the number of primitives of each kind in the expression."""
    prim_count_dict = defaultdict(int)
    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, Combinator):
            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
        elif isinstance(cur_expr, MOD_TYPE):
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
            prim_count_dict[type(cur_expr)] += 1
        else:
            raise ValueError(f'Unknown expression type {type(cur_expr)}')
    return prim_count_dict


def compile_expr(expression: GLExpr, prim_count_dict: Dict[str, int],
                 sketcher: Sketcher = None, rectify_transform=False):
    """Gather & Compute the transforms, Gather the primitive_params, Remove Difference, and Resolve Complement,"""

    # trasforms
    transforms_stack = [sketcher.get_affine_identity()]
    # inversions
    inversion_mode = False
    inversion_stack = [inversion_mode]

    prim_transforms = dict()
    prim_inversions = dict()
    prim_params = dict()
    prim_counter = {x: 0 for x in prim_count_dict.keys()}
    for prim_type, prim_count in prim_count_dict.items():
        prim_transforms[prim_type] = th.zeros((prim_count, sketcher.n_dims + 1, 
                                               sketcher.n_dims + 1),
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
        if isinstance(cur_expr, MACRO_TYPE):
            raise ValueError('Direct use of Macros not supported in compile_expr. \
                Resolve with resolve_macros first')
        elif isinstance(cur_expr, Combinator):
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
        elif isinstance(cur_expr, MOD_TYPE):
            params = cur_expr.args
            if params:
                params = cur_expr.args[1]
            if isinstance(params, Symbol):
                if params in cur_expr.lookup_table.keys():
                    params = cur_expr.lookup_table[params]
                else:
                    params = params
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
            inversion_stack.append(inversion_mode)
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
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
                    params = cur_expr.lookup_table[params]
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


def expr_to_graph(expression):
    """Convert sympy expression to a rustworkx graph.
    Note it is only valid for compiled graph."""

    graph = rx.PyDiGraph()
    parser_list = [expression]
    parent_ids = [None]
    while (parser_list):
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
    while (parser_list):
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
    while (True):
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
        rule_set = ONLY_SIMPLIFY_RULES  # ["i->i", "u->u"]
    else:
        rule_set = ALL_RULES  # ["i->i", "u->u", "i->u"]
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


def create_compiled_expr(expression, sketcher, resolve_to_dnf=False):
    prim_count = expr_prim_count(expression)
    expression = resolve_macros(expression, device=sketcher.device)
    compiled_expr = compile_expr(expression, prim_count, sketcher=sketcher, rectify_transform=True)
    expr = compiled_expr[0]
    if resolve_to_dnf:
        expr = expr_to_dnf(expr)
    transforms = {x:y.detach().cpu() for x, y in compiled_expr[1].items()}
    inversions = {x:y.detach().cpu() for x, y in compiled_expr[2].items()}
    params = {x:y.detach().cpu() for x, y in compiled_expr[3].items()}
    
    return expr, transforms, inversions, params