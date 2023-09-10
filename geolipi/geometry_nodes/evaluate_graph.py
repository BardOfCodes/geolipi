
from sympy import Symbol, Function
from geolipi.symbolic.base_symbolic import GLExpr, GLFunction
import torch as th
import numpy as np
from geolipi.symbolic import Combinator
from geolipi.symbolic.utils import resolve_macros
from geolipi.symbolic.utils import MACRO_TYPE, MOD_TYPE, TRANSLATE_TYPE, SCALE_TYPE, PRIM_TYPE
from .utils import MODIFIER_MAP, PRIMITIVE_MAP, COMBINATOR_MAP
from .geonodes import import_bpy

# TODO: Note - its always in the rectify transform mode.

def expr_to_geonode_graph(expression: GLExpr, device, dummy_obj):
    
    bpy = import_bpy()
    mod = dummy_obj.modifiers.new("CSG-GN", 'NODES')
    
    node_group = bpy.data.node_groups.new("CSG-Tree", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    output_node.is_active_output = True
    output_node.select = False
    mod.node_group = node_group
    
    output_node = node_group.nodes['Group Output']
    link_stack = [output_node.inputs['Geometry']]
        
    execution_stack = []
    operator_stack = []
    operator_nargs_stack = []
    execution_pointer_index = []
    parser_list = [expression]
    while (parser_list):
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, MACRO_TYPE):
            new_expr = resolve_macros(cur_expr, device=device)
            parser_list.append(new_expr)
        elif isinstance(cur_expr, Combinator):
            bool_node = node_group.nodes.new(type="GeometryNodeGroup")
            bool_node.node_tree = bpy.data.node_groups[COMBINATOR_MAP[type(cur_expr)]]
                
            # Linking:
            outer_input = link_stack.pop()
            node_output = bool_node.outputs['Geometry']
            node_group.links.new(node_output, outer_input)
            n_args = len(cur_expr.args)
            for arg in range(n_args):
                link_stack.append(bool_node.inputs[1])
                
            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
        elif isinstance(cur_expr, MOD_TYPE):
            params = cur_expr.args[1]
            if isinstance(params, Symbol):
                params = cur_expr.lookup_table[params]
            new_transform_node = MODIFIER_MAP[type(cur_expr)](params)
            outer_input = link_stack.pop()
            node_output = new_transform_node.outputs['Geometry']
            node_group.links.new(node_output, outer_input)
            link_stack.append(new_transform_node.inputs[0])
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
            params = cur_expr.args
            if params:
                params = params[0]
                if isinstance(params, Symbol):
                    params = cur_expr.lookup_table[params]
            
            primitive_node = PRIMITIVE_MAP[type(cur_expr)](params)
            outer_input = link_stack.pop()
            node_output = new_transform_node.outputs['Geometry']
            node_group.links.new(node_output, outer_input)
        else:
            raise ValueError(f'Unknown expression type {type(cur_expr)}')

    assert len(execution_stack) == 1
    sdf = execution_stack[0]
    return sdf

