import sys
from .utils import import_bpy


def create_geonode_tree(dummy_obj, mod_name="CSG-GN", group_name="CSG-GN"):
    """
    Creates a new geometry node tree and attaches it to a given object.

    Parameters:
        dummy_obj (bpy.types.Object): The Blender object to attach the node tree to.
        mod_name (str, optional): Name of the modifier to be used. Defaults to 'CSG-GN'.
        group_name (str, optional): Name of the new node group. Defaults to 'CSG-GN'.

    Returns:
        bpy.types.NodeTree: The newly created node tree.
    """
    bpy = import_bpy()
    if mod_name in dummy_obj.modifiers:
        dummy_obj.modifiers.remove(dummy_obj.modifiers[mod_name])
    mod = dummy_obj.modifiers.new(mod_name, "NODES")
    node_group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    node_group.outputs.new("NodeSocketGeometry", "Geometry")
    output_node = node_group.nodes.new("NodeGroupOutput")
    output_node.is_active_output = True
    output_node.select = False
    mod.node_group = node_group
    return node_group


def create_cuboid_node_seq(node_group):
    """
    Creates a sequence of nodes for generating a cuboid geometry.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.

    Returns:
        list: A list of created nodes in the sequence.
    """
    draw_node = node_group.nodes.new(type="GeometryNodeMeshCube")
    draw_node.inputs["Size"].default_value = [1, 1, 1]
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(draw_node.outputs["Mesh"], material_node.inputs["Geometry"])
    node_seq = [draw_node, material_node]
    return node_seq


def create_sphere_node_seq(node_group):
    """
    Creates a sequence of nodes for generating a sphere geometry.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.

    Returns:
        list: A list of created nodes in the sequence.
    """
    draw_node = node_group.nodes.new(type="GeometryNodeMeshUVSphere")
    draw_node.inputs["Radius"].default_value = 0.5
    draw_node.inputs["Rings"].default_value = 32
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(draw_node.outputs["Mesh"], material_node.inputs["Geometry"])
    node_seq = [draw_node, material_node]
    return node_seq


def create_cylinder_node_seq(node_group):
    """
    Creates a sequence of nodes for generating a cylinder geometry.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.

    Returns:
        list: A list of created nodes in the sequence.
    """
    draw_node = node_group.nodes.new(type="GeometryNodeMeshCylinder")
    draw_node.inputs["Radius"].default_value = 0.5
    draw_node.inputs["Depth"].default_value = 1
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(draw_node.outputs["Mesh"], material_node.inputs["Geometry"])
    node_seq = [draw_node, material_node]
    return node_seq


def create_prebaked_primitive_node_seq(node_group, filepath):
    """
    Creates a sequence of nodes for using a prebaked mesh from a file.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.
        filepath (str): File path to the mesh file.

    Returns:
        list: A list of created nodes in the sequence.
    """
    bpy = import_bpy()
    obj_name = filepath.split("/")[-1].split(".")[0]
    bpy.ops.import_mesh.ply(filepath=filepath)
    cur_obj = bpy.data.objects[obj_name]
    cur_obj.hide_viewport = True
    cur_obj.hide_render = True
    info_node = node_group.nodes.new(type="GeometryNodeObjectInfo")
    info_node.inputs[0].default_value = cur_obj

    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(
        info_node.outputs["Geometry"], material_node.inputs["Geometry"]
    )
    node_seq = [info_node, material_node]
    return node_seq


def create_boolean_union_node_seq(node_group):
    """
    Creates a sequence of nodes for performing a boolean union operation.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.

    Returns:
        list: A list containing the boolean union node.
    """
    bool_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    bool_node.operation = "UNION"
    node_seq = [bool_node]
    return node_seq


def create_boolean_intersection_node_seq(node_group):
    """
    Creates a sequence of nodes for performing a boolean intersection operation.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.

    Returns:
        list: A list containing the boolean intersection node.
    """
    bool_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    bool_node.operation = "INTERSECT"
    node_seq = [bool_node]
    return node_seq


def create_boolean_difference_node_seq(node_group):
    """
    Creates a sequence of nodes for performing a boolean difference operation.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.

    Returns:
        list: A list containing the boolean difference node.
    """
    bool_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    bool_node.operation = "DIFFERENCE"
    node_seq = [bool_node]
    return node_seq


def create_boolean_join_node_seq(node_group):
    """
    Creates a sequence of nodes for joining multiple geometries.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.

    Returns:
        list: A list containing the join geometry node.
    """
    join_node = node_group.nodes.new(type="GeometryNodeJoinGeometry")
    node_seq = [join_node]
    return node_seq


def create_transform_node_seq(node_group):
    """
    Creates a sequence of nodes for applying transformations to a geometry.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.

    Returns:
        list: A list containing the transform node.
    """
    transform_node = node_group.nodes.new(type="GeometryNodeTransform")
    node_seq = [transform_node]
    return node_seq


def create_reflect_node_seq(node_group):
    # join_node = node_group.nodes.new(type="GeometryNodeJoinGeometry")
    """
    Creates a sequence of nodes for reflecting a geometry.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.

    Returns:
        list: A list of created nodes in the sequence including the reflect operation.
    """
    join_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    join_node.operation = "UNION"
    set_position_node = node_group.nodes.new(type="GeometryNodeSetPosition")
    position_node = node_group.nodes.new(type="GeometryNodeInputPosition")
    vector_math_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    vector_math_node.operation = "REFLECT"

    node_group.links.new(set_position_node.outputs["Geometry"], join_node.inputs[0])
    node_group.links.new(
        vector_math_node.outputs["Vector"], set_position_node.inputs["Position"]
    )
    node_group.links.new(position_node.outputs["Position"], vector_math_node.inputs[0])
    return [join_node, set_position_node, position_node, vector_math_node]
