
from sympy import Symbol, Function
from geolipi.symbolic.base_symbolic import GLExpr, GLFunction
import torch as th
import numpy as np
from geolipi.symbolic import Combinator
from geolipi.symbolic.utils import resolve_macros
from geolipi.symbolic.utils import MACRO_TYPE, MOD_TYPE, TRANSLATE_TYPE, SCALE_TYPE, PRIM_TYPE
from geolipi.symbolic.combinators import Difference, PseudoUnion
from .geonodes import MODIFIER_MAP, PRIMITIVE_MAP, COMBINATOR_MAP
from .geonodes import import_bpy, create_geonode_tree
from .utils import BASE_COLORS
from .materials import create_material_tree, create_simple_material_tree

# TODO: Note - its always in the rectify transform mode.

def expr_to_geonode_graph(expression: GLExpr, dummy_obj, 
                          device, create_material=True,
                          colors=None, simple_material=False):
    
    node_group = create_geonode_tree(dummy_obj)
    mat_id = 0
    if colors is None:
        colors = BASE_COLORS
    output_node = node_group.nodes['Group Output']
    link_stack = [output_node.inputs['Geometry']]
    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, MACRO_TYPE):
            # TODO: Update this to allow sym functions to be used directly.
            new_expr = resolve_macros(cur_expr, device=device)
            parser_list.append(new_expr)
        elif isinstance(cur_expr, Combinator):
            node_seq = COMBINATOR_MAP[type(cur_expr)](node_group)
            bool_node = node_seq[0]
            # Linking:
            outer_input = link_stack.pop()
            if type(cur_expr) == PseudoUnion:
                node_output = bool_node.outputs['Geometry']
            else:
                node_output = bool_node.outputs['Mesh']
            node_group.links.new(node_output, outer_input)
            if type(cur_expr) == Difference:
                link_stack.append(bool_node.inputs['Mesh 2'])
                link_stack.append(bool_node.inputs['Mesh 1'])
            elif type(cur_expr) == PseudoUnion:
                for _ in range(len(cur_expr.args)):
                    link_stack.append(bool_node.inputs[0])
            else:
                for _ in range(len(cur_expr.args)):
                    link_stack.append(bool_node.inputs[1])
            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
        elif isinstance(cur_expr, MOD_TYPE):
            params = cur_expr.args[1]
            if isinstance(params, Symbol):
                params = cur_expr.lookup_table[params]
            if isinstance(params, th.Tensor):
                params = list(params.detach().cpu().numpy())
            node_func, param_name = MODIFIER_MAP[type(cur_expr)]
            node_seq = node_func(node_group)
            transform_node = node_seq[0]
            transform_node.inputs[param_name].default_value = params
            outer_input = link_stack.pop()
            node_output = transform_node.outputs['Geometry']
            node_group.links.new(node_output, outer_input)
            link_stack.append(transform_node.inputs[0])
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
            params = cur_expr.args
            if params:
                params = params[0]
                if isinstance(params, Symbol):
                    if params in cur_expr.lookup_table.keys():
                        params = cur_expr.lookup_table[params]
                    else:
                        params = params.name
                    
            if params:
                print(params)
                node_seq = PRIMITIVE_MAP[type(cur_expr)](node_group, params)
            else:
                node_seq = PRIMITIVE_MAP[type(cur_expr)](node_group)
            prim_node, material_node = node_seq
            if create_material:
                mat_name = f'Material_{mat_id}'
                color = colors[mat_id % len(colors)]
                if simple_material:
                    material = create_simple_material_tree(color, mat_name)
                else:
                    material = create_material_tree(color, mat_name)
                material_node.inputs['Material'].default_value = material
                mat_id += 1
            outer_input = link_stack.pop()
            node_output = material_node.outputs['Geometry']
            node_group.links.new(node_output, outer_input)
        else:
            raise ValueError(f'Unknown expression type {type(cur_expr)}')

    return node_group
