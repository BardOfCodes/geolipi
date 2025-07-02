from collections import defaultdict
import numpy as np
import torch as th
import rustworkx as rx

from geolipi.symbolic import Combinator, Difference, Intersection, Union
from geolipi.symbolic.base import GLExpr
from geolipi.symbolic.types import (
    MACRO_TYPE,
    MOD_TYPE,
    TRANSLATE_TYPE,
    SCALE_TYPE,
    PRIM_TYPE,
    TRANSSYM_TYPE,
)
from geolipi.symbolic.base import PrimitiveSpec
from geolipi.symbolic.resolve import resolve_macros

from .sketcher import Sketcher
from .maps import MODIFIER_MAP
from .maps import INVERTED_MAP, NORMAL_MAP, ONLY_SIMPLIFY_RULES, ALL_RULES, ONLY_SIMPLIFY_RULES_CNF, ALL_RULES_CNF

# Don't resolve when expression is crazy big.
MAX_EXPR_SIZE = 500


def expr_prim_count(expression: GLExpr):
    """Get the number of primitives of each kind in the expression."""
    prim_count_dict = defaultdict(int)
    parser_list = [expression]
    while parser_list:
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
            raise ValueError(f"Unknown expression type {type(cur_expr)}")
    return prim_count_dict


def compile_expr(
    expression: GLExpr, sketcher: Sketcher = None, rectify_transform=False
):
    """
    Compiles a GL expression into a format suitable for batch evaluation, gathering transformations,
    primitive parameters, and handling difference and complement operations.

    This function traverses the expression tree to extract and organize necessary data for
    rendering the expression. It accounts for transformations, inversion modes, and primitive
    parameters, and resolves any complex operations like Difference and Complement.

    Parameters:
        expression (GLExpr): The GL expression to be compiled.
        sketcher (Sketcher, optional): The sketcher object used for affine transformations.
            Defaults to None.
        rectify_transform (bool, optional): Flag to determine if transformations should be rectified.
            Defaults to False.

    Returns:
        tuple: A tuple containing the compiled expression, primitive transformations, primitive
        inversions, and primitive parameters. These are organized to facilitate batch evaluation
        of the expression.
    """

    prim_count_dict = expr_prim_count(expression)
    # trasforms
    transforms_stack = [sketcher.get_affine_identity()]
    # inversions
    inversion_mode = False
    inversion_stack = [inversion_mode]

    prim_transforms = dict()
    prim_inversions = dict()
    prim_params = defaultdict(list)
    prim_counter = {x: 0 for x in prim_count_dict.keys()}
    for prim_type, prim_count in prim_count_dict.items():
        prim_transforms[prim_type] = th.zeros(
            (prim_count, sketcher.n_dims + 1, sketcher.n_dims + 1),
            dtype=sketcher.dtype,
            device=sketcher.device,
        )
        prim_inversions[prim_type] = th.zeros(
            (prim_count), dtype=th.bool, device=sketcher.device
        )
        # Todo: Handle parameterized Primitives
        # Add type annotation to functions.
        n_params = 0  # len(inspect.signature(prim_type).parameters) # but each might not be singleton.
        if n_params > 0:
            prim_params[prim_type] = th.zeros(
                (prim_count, n_params), dtype=sketcher.dtype, device=sketcher.device
            )

    execution_stack = []
    execution_pointer_index = []
    operator_stack = []
    operator_nargs_stack = []
    operator_params_stack = []
    if rectify_transform:
        scale_stack = [sketcher.get_scale_identity()]

    parser_list = [expression]
    while parser_list:
        cur_expr = parser_list.pop()
        inversion_mode = inversion_stack.pop()
        if isinstance(cur_expr, MACRO_TYPE):
            raise ValueError(
                "Direct use of Macros not supported in compile_expr. \
                Resolve with resolve_macros first"
            )
        elif isinstance(cur_expr, Combinator):
            tree_branches, param_list = [], []
            for arg in cur_expr.args:
                if arg in cur_expr.lookup_table:
                    param_list.append(cur_expr.lookup_table[arg])
                else:
                    tree_branches.append(arg)
            n_args = len(tree_branches)
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
            operator_params_stack.append(param_list)
            next_to_parse = tree_branches[::-1]
            parser_list.extend(next_to_parse)
            execution_pointer_index.append(len(execution_stack))
        elif isinstance(cur_expr, MOD_TYPE):
            params = cur_expr.args[1:]
            next_to_parse = cur_expr.args[0]
            if params:
                param_list = []
                for ind, param in enumerate(params):
                    if param in cur_expr.lookup_table:
                        cur_param = cur_expr.lookup_table[param]
                        param_list.append(cur_param)
                    else:
                        param_list.append(param)
                params = param_list
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
            inversion_stack.append(inversion_mode)
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
            transform = transforms_stack.pop()
            if rectify_transform:
                _ = scale_stack.pop()
            prim_type = type(cur_expr)
            prim_id = prim_counter[prim_type]
            prim_transforms[prim_type][prim_id] = transform
            if inversion_mode:
                prim_inversions[prim_type][prim_id] = True
            param_list = []
            params = cur_expr.args
            if params:
                for ind, param in enumerate(params):
                    if param in cur_expr.lookup_table:
                        cur_param = cur_expr.lookup_table[param]
                        param_list.append(cur_param)
                    else:
                        param_list.append(param)
                params = param_list
            prim_params[prim_type].append(params)

            prim_spec = PrimitiveSpec(prim_type, prim_id)
            execution_stack.append(prim_spec)
            prim_counter[prim_type] += 1
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
            new_canvas = operator(*args, *params)
            execution_stack = execution_stack[:-n_args] + [new_canvas]

    for prim_type, prim_param_list in prim_params.items():
        n_params = len(prim_param_list[0])
        final_list = []
        for cur_id in range(n_params):
            cur_param_list = [x[cur_id] for x in prim_param_list]
            final_list.append(th.stack(cur_param_list, 0))
        prim_params[prim_type] = final_list

    expression = execution_stack[0]
    return expression, prim_transforms, prim_inversions, prim_params


def expr_to_graph(expression):
    """
    Converts a sympy expression to a rustworkx graph. This is valid only for compiled graphs.
    Used for simplifying the expression to DNF form.

    Parameters:
        expression: A sympy expression representing a mathematical or logical construct.

    Returns:
        A rustworkx PyDiGraph representing the structure of the expression.
    """
    graph = rx.PyDiGraph()
    parser_list = [expression]
    parent_ids = [None]
    while parser_list:
        cur_expr = parser_list.pop()
        parent_id = parent_ids.pop()
        if isinstance(cur_expr, Combinator):
            node = dict(type=type(cur_expr))
            cur_id = graph.add_node(node)
            parser_list.extend(cur_expr.args[::-1])
            parent_ids.extend([cur_id for x in range(len(cur_expr.args))])
        elif isinstance(cur_expr, (PrimitiveSpec, PRIM_TYPE)):
            node = dict(type=type(cur_expr), expr=cur_expr)
            cur_id = graph.add_node(node)
        if not parent_id is None:
            graph.add_edge(parent_id, cur_id, None)
    return graph


def graph_to_expr(graph):
    """
    Converts a rustworkx graph back to a sympy expression.
    Used to get the expression back after converting to the DNF form.

    Parameters:
        graph: A rustworkx PyDiGraph generated from a sympy expression.

    Returns:
        A sympy expression reconstructed from the graph.
    """
    prim_stack = []
    prim_stack_pointer = []
    operator_stack = []
    operator_nargs_stack = []

    parser_list = [0]
    while parser_list:
        cur_id = parser_list.pop()
        cur_node = graph[cur_id]
        c_type = cur_node["type"]
        if issubclass(c_type, (PrimitiveSpec, PRIM_TYPE)):
            prim_stack.append(cur_node["expr"])
        elif issubclass(c_type, Combinator):
            operator_stack.append(c_type)
            next_to_parse = list(graph.successor_indices(cur_id))
            n_args = len(next_to_parse)
            operator_nargs_stack.append(n_args)
            parser_list.extend(next_to_parse)
            prim_stack_pointer.append(len(prim_stack))

        while (
            operator_stack
            and len(prim_stack) - prim_stack_pointer[-1] >= operator_nargs_stack[-1]
        ):
            n_args = operator_nargs_stack.pop()
            operator = operator_stack.pop()
            _ = prim_stack_pointer.pop()
            args = prim_stack[-n_args:]
            new_canvas = operator(*args)
            prim_stack = prim_stack[:-n_args] + [new_canvas]
    expression = prim_stack[0]
    return expression


def expr_to_dnf(expression, max_expr_size=MAX_EXPR_SIZE):
    """
    Converts an expression to its Disjunctive Normal Form (DNF) using graph transformation rules.

    Parameters:
        expression: The expression to convert.
        max_expr_size (int): The maximum allowed size of the expression during conversion.

    Returns:
        A sympy expression in DNF.
    """
    graph = expr_to_graph(expression)
    # RULES:
    # Intersection-> Intersection = collapse
    # Union -> union = collapse
    # Intersection -> Union = invert
    # Union -> Intersection = retain
    rule_match = None
    while True:
        rule_match = get_rule_match(graph, only_simplify=True)
        if rule_match is None:
            rule_match = get_rule_match(graph, only_simplify=False)
            if rule_match is None:
                break
            else:
                graph = resolve_rule(graph, rule_match)
        else:
            graph = resolve_rule(graph, rule_match)
        if graph.num_nodes() > max_expr_size:
            return expression
    expression = graph_to_expr(graph)
    return expression


def get_rule_match(graph, only_simplify=False):
    """
    Identifies a rule match in a graph that represents an expression.

    Parameters:
        graph: The graph representation of the expression.
        only_simplify (bool): If True, only simplification rules are applied.

    Returns:
        A match if a rule can be applied, else None.
    """
    if only_simplify:
        rule_set = ONLY_SIMPLIFY_RULES  # ["i->i", "u->u"]
    else:
        rule_set = ALL_RULES  # ["i->i", "u->u", "i->u"]
    node_ids = [0]
    rule_match = None
    while node_ids:
        cur_id = node_ids.pop()
        cur_node = graph[cur_id]
        c_type = cur_node["type"]
        if issubclass(c_type, Combinator):
            children = list(graph.successor_indices(cur_id))
            node_ids.extend(children[::-1])
            for ind, child_id in enumerate(children):
                child_node = graph[child_id]
                child_type = child_node["type"]
                rel_sig = (c_type, child_type)
                if rel_sig in rule_set:
                    rule_match = (cur_id, child_id, rel_sig)
        if rule_match:
            break

    return rule_match


def resolve_rule(graph, resolve_rule):
    """
    Applies a rule to transform a graph representing an expression.

    Parameters:
        graph: The graph representation of the expression.
        resolve_rule: The rule to be applied to the graph.

    Returns:
        The graph after applying the transformation rule.
    """
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
        node_a["type"] = Union
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


#############################################################################
def expr_to_cnf(expression, max_expr_size=MAX_EXPR_SIZE):
    """
    Converts an expression to its Conjunctive Normal Form (CNF) using graph transformation rules.

    Parameters:
        expression: The expression to convert.
        max_expr_size (int): The maximum allowed size of the expression during conversion.

    Returns:
        A sympy expression in CNF.
    """
    graph = expr_to_graph(expression)

    while True:
        rule_match = get_rule_match_cnf(graph, only_simplify=True)
        if rule_match is None:
            rule_match = get_rule_match_cnf(graph, only_simplify=False)
            if rule_match is None:
                break
            else:
                graph = resolve_rule_cnf(graph, rule_match)
        else:
            graph = resolve_rule_cnf(graph, rule_match)

        if graph.num_nodes() > max_expr_size:
            return expression

    expression = graph_to_expr(graph)
    return expression


def get_rule_match_cnf(graph, only_simplify=False):
    """
    Identifies a rule match for CNF conversion.

    Parameters:
        graph: The graph representation of the expression.
        only_simplify (bool): If True, only simplification rules are applied.

    Returns:
        A match if a rule can be applied, else None.
    """
    rule_set = ONLY_SIMPLIFY_RULES_CNF if only_simplify else ALL_RULES_CNF
    node_ids = [0]

    while node_ids:
        cur_id = node_ids.pop()
        cur_node = graph[cur_id]
        c_type = cur_node["type"]

        if issubclass(c_type, Combinator):
            children = list(graph.successor_indices(cur_id))
            node_ids.extend(children[::-1])

            for child_id in children:
                child_node = graph[child_id]
                child_type = child_node["type"]
                rel_sig = (c_type, child_type)

                if rel_sig in rule_set:
                    return (cur_id, child_id, rel_sig)

    return None


def resolve_rule_cnf(graph, rule):
    """
    Applies CNF transformation rules to the graph.

    Parameters:
        graph: The graph representing the expression.
        rule: The matched rule to apply.

    Returns:
        The updated graph after applying the rule.
    """
    node_a_id, node_b_id, match_type = rule
    node_a = graph[node_a_id]

    if match_type in ONLY_SIMPLIFY_RULES_CNF:
        # Merge nested Unions/Intersections
        for child_id in graph.successor_indices(node_b_id):
            graph.add_edge(node_a_id, child_id, None)
        graph.remove_edge(node_a_id, node_b_id)

    elif match_type == (Union, Intersection):
        # Distribute Union over Intersection
        node_a["type"] = Intersection
        children_a = list(graph.successor_indices(node_a_id))
        children_a_not_b = [ind for ind in children_a if ind != node_b_id]
        children_b = list(graph.successor_indices(node_b_id))

        # Remove current edges
        for child_id in children_a:
            graph.remove_edge(node_a_id, child_id)

        # Apply distribution
        for child_id in children_b:
            new_node = dict(type=Union)
            new_id = graph.add_node(new_node)
            graph.add_edge(new_id, child_id, None)
            for not_b_child_id in children_a_not_b:
                graph.add_edge(new_id, not_b_child_id, None)
            graph.add_edge(node_a_id, new_id, None)

    return graph
#############################################################################

def create_compiled_expr(
    expression,
    sketcher,
    resolve_to_dnf=False,
    convert_to_cpu=True,
    rectify_transform=False,
):
    """
    Compiles an expression and optionally converts it to Disjunctive Normal Form (DNF) and to CPU.

    Parameters:
        expression: The expression to compile.
        sketcher: The sketcher object used in compilation.
        resolve_to_dnf (bool): If True, converts the expression to DNF.
        convert_to_cpu (bool): If True, converts tensors to CPU tensors.
        rectify_transform (bool): If True, rectifies transformations during compilation.

    Returns:
        A tuple containing the compiled expression, transforms, inversions, and parameters.
    """
    expression = resolve_macros(expression, device=sketcher.device)
    compiled_expr = compile_expr(
        expression, sketcher=sketcher, rectify_transform=rectify_transform
    )
    expr, transforms, inversions, params = compiled_expr

    if resolve_to_dnf:
        expr = expr_to_dnf(expr)

    if convert_to_cpu:
        transforms = {x: y.cpu() for x, y in transforms.items()}
        inversions = {x: y.cpu() for x, y in inversions.items()}
        for prim_type, prim_param_list in params.items():
            params[prim_type] = [x.cpu() for x in prim_param_list]
        # params = {x: y.cpu() for x, y in params.items()}

    return expr, transforms, inversions, params
