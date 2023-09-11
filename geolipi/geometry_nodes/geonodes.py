import sys
from geolipi.symbolic.combinators import Union, Intersection, Difference
from geolipi.symbolic.transforms_3d import Translate3D, Scale3D, EulerRotate3D
from geolipi.symbolic.primitives_3d import NoParamCuboid3D, NoParamCylinder3D, NoParamSphere3D
from .utils import import_bpy
# TBD others.


def create_geonode_tree(dummy_obj, mod_name="CSG-GN", group_name="CSG-GN"):
    bpy = import_bpy()
    mod = dummy_obj.modifiers.new(mod_name, 'NODES')
    node_group = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    output_node.is_active_output = True
    output_node.select = False
    mod.node_group = node_group
    return node_group


def create_cuboid_node_seq(node_group):
    draw_node = node_group.nodes.new(type="GeometryNodeMeshCube")
    draw_node.inputs['Size'].default_value = [1, 1, 1]
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(
        draw_node.outputs['Mesh'], material_node.inputs['Geometry'])
    node_seq = [draw_node, material_node]
    return node_seq


def create_sphere_node_seq(node_group):
    draw_node = node_group.nodes.new(type="GeometryNodeMeshUVSphere")
    draw_node.inputs['Radius'].default_value = 0.5
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(
        draw_node.outputs['Mesh'], material_node.inputs['Geometry'])
    node_seq = [draw_node, material_node]
    return node_seq


def create_cylinder_node_seq(node_group):
    draw_node = node_group.nodes.new(type="GeometryNodeMeshCylinder")
    draw_node.inputs['Radius'].default_value = 0.5
    draw_node.inputs['Depth'].default_value = 1
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(
        draw_node.outputs['Mesh'], material_node.inputs['Geometry'])
    node_seq = [draw_node, material_node]
    return node_seq


def create_boolean_union_node_seq(node_group):
    bool_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    bool_node.operation = "UNION"
    node_seq = [bool_node]
    return node_seq


def create_boolean_intersection_node_seq(node_group):
    bool_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    bool_node.operation = "INTERSECT"
    node_seq = [bool_node]
    return node_seq


def create_boolean_difference_node_seq(node_group):
    bool_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    bool_node.operation = "DIFFERENCE"
    node_seq = [bool_node]
    return node_seq


def create_transform_node_seq(node_group):
    bpy = import_bpy()
    transform_node = node_group.nodes.new(type="GeometryNodeTransform")
    node_seq = [transform_node]
    return node_seq


def create_reflect_node_seq(node_group):
    # join_node = node_group.nodes.new(type="GeometryNodeJoinGeometry")
    join_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    join_node.operation = "UNION"
    set_position_node = node_group.nodes.new(type="GeometryNodeSetPosition")
    position_node = node_group.nodes.new(type="GeometryNodeInputPosition")
    vector_math_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    vector_math_node.operation = "REFLECT"

    node_group.links.new(
        set_position_node.outputs['Geometry'], join_node.inputs[0])
    node_group.links.new(
        vector_math_node.outputs['Vector'], set_position_node.inputs['Position'])
    node_group.links.new(
        position_node.outputs['Position'], vector_math_node.inputs[0])
    return [join_node, set_position_node, position_node, vector_math_node]


COMBINATOR_MAP = {
    Union: create_boolean_union_node_seq,
    Intersection: create_boolean_intersection_node_seq,
    Difference: create_boolean_difference_node_seq,
}
PRIMITIVE_MAP = {
    NoParamCuboid3D: create_cuboid_node_seq,
    NoParamSphere3D: create_sphere_node_seq,
    NoParamCylinder3D: create_cylinder_node_seq,
}

MODIFIER_MAP = {
    Translate3D: (create_transform_node_seq, "Translation"),
    Scale3D: (create_transform_node_seq, "Scale"),
    EulerRotate3D: (create_transform_node_seq, "Rotation")

}
