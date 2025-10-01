import sympy as sp
from geolipi.symbolic.base import GLFunction
import torch as th
from geolipi.symbolic import Combinator
import geolipi.symbolic as gls
from geolipi.symbolic.resolve import resolve_macros
from geolipi.symbolic.symbol_types import MACRO_TYPE, MOD_TYPE, PRIM_TYPE
from .utils import MODIFIER_MAP, PRIMITIVE_MAP, COMBINATOR_MAP, BASE_COLORS
from .geonodes import create_geonode_tree
from .materials import (
    create_material_tree,
    create_simple_material_tree,
    create_edge_material_tree,
    create_monotone_material,
)

# TODO: Note - its always in the rectify transform mode.
def expr_to_geonode_graph(
    expression: GLFunction,
    dummy_obj,
    device,
    create_material=True,
    colors=None,
    material_type="base",
):
    """
    Converts a geometric expression into a Blender geometry node graph.
    Currently, this function only supports the NoParam 2D/3D Primitives, and basic boolean/transform operations.
    
    Parameters:
        expression (GLFunction): The geometric expression to be converted into a node graph.
        dummy_obj (bpy.types.Object): The Blender object to which the node graph will be attached.
        device: The device used for any necessary computations (e.g., resolving macros).
        create_material (bool, optional): Whether to create materials for the geometry. Defaults to True.
        colors (list, optional): A list of colors to be used for the materials. If None, BASE_COLORS is used. Defaults to None.
        material_type (str, optional): Specifies the type of material to be created. Options are "base", "simple", "with_edge", "monotone". Defaults to "base".

    Returns:
        bpy.types.NodeTree: The newly created geometry node tree.

    Raises:
        ValueError: If an unknown expression type is encountered.

    """
    node_group = create_geonode_tree(dummy_obj)
    mat_id = 0
    if colors is None:
        colors = BASE_COLORS
    output_node = node_group.nodes["Group Output"]
    link_stack = [output_node.inputs["Geometry"]]
    parser_list = [expression]
    while parser_list:
        cur_expr = parser_list.pop()
        if isinstance(cur_expr, MACRO_TYPE):
            # TODO: Update this to allow sym functions to be used directly.
            new_expr = resolve_macros(cur_expr, device=device)
            parser_list.append(new_expr)
        elif isinstance(cur_expr, gls.Combinator):
            node_seq = COMBINATOR_MAP[type(cur_expr)](node_group)
            bool_node = node_seq[0]
            # Linking:
            outer_input = link_stack.pop()
            if type(cur_expr) == gls.JoinUnion:
                node_output = bool_node.outputs["Geometry"]
            else:
                node_output = bool_node.outputs["Mesh"]
            node_group.links.new(node_output, outer_input)
            if type(cur_expr) == gls.Difference:
                link_stack.append(bool_node.inputs["Mesh 2"])
                link_stack.append(bool_node.inputs["Mesh 1"])
            elif type(cur_expr) == gls.JoinUnion:
                for _ in range(len(cur_expr.args)):
                    link_stack.append(bool_node.inputs[0])
            elif type(cur_expr) == gls.Complement:
                link_stack.append(bool_node.inputs["Mesh 2"])
            else:
                for _ in range(len(cur_expr.args)):
                    link_stack.append(bool_node.inputs[1])
            next_to_parse = cur_expr.args[::-1]
            parser_list.extend(next_to_parse)
        elif isinstance(cur_expr, MOD_TYPE):

            params = cur_expr.args[1:]
            params = _parse_param_from_expr(cur_expr, params)
            node_func, param_name = MODIFIER_MAP[type(cur_expr)]
            node_seq = node_func(node_group)
            transform_node = node_seq[0]
            if isinstance(cur_expr, gls.EulerRotate3D):
                params = ([-x for x in params[0]],)
            transform_node.inputs[param_name].default_value = params[0]
            outer_input = link_stack.pop()
            node_output = transform_node.outputs["Geometry"]
            node_group.links.new(node_output, outer_input)
            link_stack.append(transform_node.inputs[0])
            next_to_parse = cur_expr.args[0]
            parser_list.append(next_to_parse)
        elif isinstance(cur_expr, PRIM_TYPE):
            params = cur_expr.args
            params = _parse_param_from_expr(cur_expr, params)
            node_seq = PRIMITIVE_MAP[type(cur_expr)](node_group, *params)
            material_node = node_seq[1]
            if create_material:
                # TODO: Better color management.
                mat_name = f"Material_{mat_id}"
                color = colors[mat_id % len(colors)]
                if material_type == "base":
                    material = create_material_tree(mat_name, color)
                elif material_type == "simple":
                    material = create_simple_material_tree(mat_name, color)
                elif material_type == "with_edge":
                    material = create_edge_material_tree(mat_name, color)
                elif material_type == "monotone":
                    color_list = [
                        colors[(mat_id + i * 2) % len(colors)] for i in range(3)
                    ]
                    silhuette_color = colors[(mat_id - 1) % len(colors)]
                    material = create_monotone_material(
                        mat_name, color, color_list, silhuette_color
                    )

                material_node.inputs["Material"].default_value = material
                mat_id += 1
            outer_input = link_stack.pop()
            node_output = material_node.outputs["Geometry"]
            node_group.links.new(node_output, outer_input)
        else:
            raise ValueError(f"Unknown expression type {type(cur_expr)}")

    return node_group

def _parse_param_from_expr(expression, params):
    if params:
        param_list = []
        for ind, param in enumerate(params):
            if param in expression.lookup_table:
                cur_param = expression.lookup_table[param]
            else: 
                cur_param = param
            if isinstance(cur_param, th.Tensor):
                cur_param = list(params.detach().cpu().numpy())
            if isinstance(cur_param, sp.Tuple):
                cur_param = list(float(x) for x in cur_param)
            param_list.append(cur_param)
        params = param_list
    return params
