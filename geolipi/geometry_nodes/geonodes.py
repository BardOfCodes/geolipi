import sys
import numpy as np
from .bl_utils import import_bpy

SDF_RESOLUTION = 64
EPSILON = 1e-6


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
    node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    # node_group.outputs.new("NodeSocketGeometry", "Geometry")
    output_node = node_group.nodes.new("NodeGroupOutput")
    output_node.is_active_output = True
    output_node.select = False
    mod.node_group = node_group
    return node_group


def create_cuboid_node_seq(node_group, size=None):
    """
    Creates a sequence of nodes for generating a cuboid geometry.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.
        size (tuple, optional): The size of the cuboid in the form (x, y, z). Defaults to (1, 1, 1).

    Returns:
        list: A list of created nodes in the sequence.
    """
    if size is None:
        size = (1, 1, 1)
    draw_node = node_group.nodes.new(type="GeometryNodeMeshCube")
    draw_node.inputs["Size"].default_value = size
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(draw_node.outputs["Mesh"], material_node.inputs["Geometry"])
    node_seq = [draw_node, material_node]
    return node_seq


def create_sphere_node_seq(node_group, r=None):
    """
    Creates a sequence of nodes for generating a sphere geometry.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.
        r (tuple, optional): The radius of the sphere. Defaults to (0.5,).

    Returns:
        list: A list of created nodes in the sequence.
    """
    if r is None:
        r = (0.5,)
    draw_node = node_group.nodes.new(type="GeometryNodeMeshUVSphere")
    draw_node.inputs["Radius"].default_value = r[0]
    draw_node.inputs["Rings"].default_value = 32
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(draw_node.outputs["Mesh"], material_node.inputs["Geometry"])
    node_seq = [draw_node, material_node]
    return node_seq


def create_cylinder_node_seq(node_group, h=None, r=None):
    """
    Creates a sequence of nodes for generating a cylinder geometry.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.
        h (tuple, optional): The height of the cylinder. Defaults to (1,).
        r (tuple, optional): The radius of the cylinder. Defaults to (0.5,).

    Returns:
        list: A list of created nodes in the sequence.
    """
    if h is None:
        h = (1,)
    if r is None:
        r = (0.5,)
    draw_node = node_group.nodes.new(type="GeometryNodeMeshCylinder")
    draw_node.inputs["Depth"].default_value = h[0]
    draw_node.inputs["Radius"].default_value = r[0]
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(draw_node.outputs["Mesh"], material_node.inputs["Geometry"])
    node_seq = [draw_node, material_node]
    return node_seq

def create_cone_node_seq(node_group, angle, h):
    """
    Creates a sequence of nodes for generating a cylinder geometry.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.
        angle (tuple): The angle the cone makes about the horizontal plane.
        h (tuple): The height of the cone.

    Returns:
        list: A list of created nodes in the sequence.
    """
    # based on height and angle find radius
    r = h[0] * np.tan(np.pi/2 - angle[0])
    r = (r,)
    draw_node = node_group.nodes.new(type="GeometryNodeMeshCone")
    draw_node.inputs["Depth"].default_value = h[0]
    draw_node.inputs["Radius Bottom"].default_value = r[0]
    # translate
    transform_node = node_group.nodes.new(type="GeometryNodeTransform")
    transform_node.inputs['Translation'].default_value = (0, 0, -h[0])
    node_group.links.new(draw_node.outputs["Mesh"], transform_node.inputs["Geometry"])

    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(transform_node.outputs["Geometry"], material_node.inputs["Geometry"])
    node_seq = [draw_node, material_node]
    return node_seq

def create_inexact_super_quadrics_node_seq(node_group, skew_vec, epsilon_1, epsilon_2):
    """
    Creates a sequence of nodes for generating a cylinder geometry.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.
        skew_vec (tuple): The skew vector for the superquadric.
        epsilon_1 (tuple): The epsilon_1 value for the superquadric.
        epsilon_2 (tuple): The epsilon_2 value for the superquadric.

    Returns:
        list: A list of created nodes in the sequence.
    """
    vol_to_mesh_node = node_group.nodes.new(type="GeometryNodeVolumeToMesh")
    vol_to_mesh_node.inputs[3].default_value = 0.001
    vol_cube = node_group.nodes.new(type="GeometryNodeVolumeCube")
    vol_cube.inputs['Resolution X'].default_value = SDF_RESOLUTION
    vol_cube.inputs['Resolution Y'].default_value = SDF_RESOLUTION
    vol_cube.inputs['Resolution Z'].default_value = SDF_RESOLUTION
    # Entire maths here
    epsilon_2 = epsilon_2[0]
    epsilon_1 = epsilon_1[0]
    position_node = node_group.nodes.new(type="GeometryNodeInputPosition")
    abs_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    abs_node.operation = "ABSOLUTE"
    node_group.links.new(position_node.outputs["Position"], abs_node.inputs[0])
    div_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    div_node.operation = "DIVIDE"
    div_node.inputs[1].default_value = skew_vec
    node_group.links.new(abs_node.outputs["Vector"], div_node.inputs[0])
    sep_node = node_group.nodes.new(type="ShaderNodeSeparateXYZ")
    node_group.links.new(div_node.outputs["Vector"], sep_node.inputs[0])
    pow_node_x = node_group.nodes.new(type="ShaderNodeMath")
    pow_node_x.operation = "POWER"
    expo_1 = 2 /(epsilon_1 + EPSILON)
    expo_2 = 2 /(epsilon_2 + EPSILON)
    pow_node_x.inputs[1].default_value = expo_1
    node_group.links.new(sep_node.outputs["X"], pow_node_x.inputs[0])
    # Same for Y
    pow_node_y = node_group.nodes.new(type="ShaderNodeMath")
    pow_node_y.operation = "POWER"
    pow_node_y.inputs[1].default_value = expo_1
    node_group.links.new(sep_node.outputs["Y"], pow_node_y.inputs[0])
    # Same for Z
    pow_node_z = node_group.nodes.new(type="ShaderNodeMath")
    pow_node_z.operation = "POWER"
    pow_node_z.inputs[1].default_value = expo_2
    node_group.links.new(sep_node.outputs["Z"], pow_node_z.inputs[0])
    add_x_y = node_group.nodes.new(type="ShaderNodeMath")
    add_x_y.operation = "ADD"
    node_group.links.new(pow_node_x.outputs["Value"], add_x_y.inputs[0])
    node_group.links.new(pow_node_y.outputs["Value"], add_x_y.inputs[1])
    abs_x_y = node_group.nodes.new(type="ShaderNodeMath")
    abs_x_y.operation = "ABSOLUTE"
    node_group.links.new(add_x_y.outputs["Value"], abs_x_y.inputs[0])
    # add epsilon
    add_eps = node_group.nodes.new(type="ShaderNodeMath")
    add_eps.operation = "ADD"
    add_eps.inputs["Value"].default_value = EPSILON
    node_group.links.new(abs_x_y.outputs["Value"], add_eps.inputs[0])
    # take power
    pow_node = node_group.nodes.new(type="ShaderNodeMath")
    pow_node.operation = "POWER"
    pow_node.inputs[1].default_value = (epsilon_2 / (epsilon_1 + EPSILON))
    node_group.links.new(add_eps.outputs["Value"], pow_node.inputs[0])
    # add to z
    all_sum = node_group.nodes.new(type="ShaderNodeMath")
    all_sum.operation = "ADD"
    node_group.links.new(pow_node_z.outputs["Value"], all_sum.inputs[0])
    node_group.links.new(pow_node.outputs["Value"], all_sum.inputs[1])
    # abs
    abs_all = node_group.nodes.new(type="ShaderNodeMath")
    abs_all.operation = "ABSOLUTE"
    node_group.links.new(all_sum.outputs["Value"], abs_all.inputs[0])
    # pow
    pow_all = node_group.nodes.new(type="ShaderNodeMath")
    pow_all.operation = "POWER"
    pow_all.inputs[1].default_value = (-epsilon_1 / 2.0)
    node_group.links.new(abs_all.outputs["Value"], pow_all.inputs[0])
    # greater than 1
    greater_than = node_group.nodes.new(type="ShaderNodeMath")
    greater_than.operation = "GREATER_THAN"
    greater_than.inputs[1].default_value = 1
    node_group.links.new(pow_all.outputs["Value"], greater_than.inputs[0])
    # link to vol cube
    node_group.links.new(greater_than.outputs["Value"], vol_cube.inputs["Density"])
    # vol cube to vol_to_mesh
    node_group.links.new(vol_cube.outputs["Volume"], vol_to_mesh_node.inputs["Volume"])

    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(vol_to_mesh_node.outputs["Mesh"], material_node.inputs["Geometry"])
    node_seq = [vol_to_mesh_node, material_node]
    return node_seq

def create_plane_node_seq(node_group, n, h):
    """
    Creates a sequence of nodes for generating a half space based on a plane normal and distance from origin.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.
        n (tuple): The normal vector of the plane.
        h (tuple): The distance of the plane from the origin.
    
    Returns:
        list: A list of created nodes in the sequence.
    """

    vol_to_mesh_node = node_group.nodes.new(type="GeometryNodeVolumeToMesh")
    vol_to_mesh_node.inputs[3].default_value = 0.001
    vol_cube = node_group.nodes.new(type="GeometryNodeVolumeCube")
    vol_cube.inputs['Resolution X'].default_value = SDF_RESOLUTION
    vol_cube.inputs['Resolution Y'].default_value = SDF_RESOLUTION
    vol_cube.inputs['Resolution Z'].default_value = SDF_RESOLUTION
    # Maths
    # normal vector
    vector_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    vector_node.operation = "NORMALIZE"
    print("n here", n)
    vector_node.inputs[0].default_value = n
    # position
    position_node = node_group.nodes.new(type="GeometryNodeInputPosition")
    # dot product
    dot_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    dot_node.operation = "DOT_PRODUCT"
    node_group.links.new(position_node.outputs["Position"], dot_node.inputs[0])
    node_group.links.new(vector_node.outputs["Vector"], dot_node.inputs[1])
    # add
    add_node = node_group.nodes.new(type="ShaderNodeMath")
    add_node.operation = "ADD"
    add_node.inputs[1].default_value = h[0]
    node_group.links.new(dot_node.outputs["Value"], add_node.inputs[0])
    # less than
    less_than_node = node_group.nodes.new(type="ShaderNodeMath")
    less_than_node.operation = "LESS_THAN"
    less_than_node.inputs[1].default_value = EPSILON
    node_group.links.new(add_node.outputs["Value"], less_than_node.inputs[0])
    # link to vol cube
    node_group.links.new(less_than_node.outputs["Value"], vol_cube.inputs["Density"])
    # vol cube to vol_to_mesh
    node_group.links.new(vol_cube.outputs["Volume"], vol_to_mesh_node.inputs["Volume"])

    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(vol_to_mesh_node.outputs["Mesh"], material_node.inputs["Geometry"])
    node_seq = [vol_to_mesh_node, material_node]
    return node_seq


def create_inf_cylinder_node_seq(node_group, c):
    """
    Creates a sequence of nodes for generating an infinite cylinder geometry.
    
    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.
        c (tuple):  A tuple of shape [3] representing the cylinder's center and radius (last component).

    Returns:
        list: A list of created nodes in the sequence.
    """

    vol_to_mesh_node = node_group.nodes.new(type="GeometryNodeVolumeToMesh")
    vol_to_mesh_node.inputs[3].default_value = 0.001
    vol_cube = node_group.nodes.new(type="GeometryNodeVolumeCube")
    vol_cube.inputs['Resolution X'].default_value = SDF_RESOLUTION
    vol_cube.inputs['Resolution Y'].default_value = SDF_RESOLUTION
    vol_cube.inputs['Resolution Z'].default_value = SDF_RESOLUTION
    # Maths
    position_node = node_group.nodes.new(type="GeometryNodeInputPosition")
    # sub_node
    sub_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    sub_node.operation = "SUBTRACT"
    sub_node.inputs[1].default_value = c
    node_group.links.new(position_node.outputs["Position"], sub_node.inputs[0])
    # separate XYZ
    sep_node = node_group.nodes.new(type="ShaderNodeSeparateXYZ")
    node_group.links.new(sub_node.outputs["Vector"], sep_node.inputs[0])
    # combine XYZ
    comb_node = node_group.nodes.new(type="ShaderNodeCombineXYZ")
    node_group.links.new(sep_node.outputs["X"], comb_node.inputs[1])
    node_group.links.new(sep_node.outputs["Y"], comb_node.inputs[2])
    # length
    len_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    len_node.operation = "LENGTH"
    node_group.links.new(comb_node.outputs["Vector"], len_node.inputs[0])
    # sub math
    sub_math_node = node_group.nodes.new(type="ShaderNodeMath")
    sub_math_node.operation = "SUBTRACT"
    sub_math_node.inputs[1].default_value = c[2]
    node_group.links.new(len_node.outputs["Value"], sub_math_node.inputs[0])
    # less than
    less_than_node = node_group.nodes.new(type="ShaderNodeMath")
    less_than_node.operation = "LESS_THAN"
    less_than_node.inputs[1].default_value = EPSILON
    node_group.links.new(sub_math_node.outputs["Value"], less_than_node.inputs[0])
    # link to vol cube
    node_group.links.new(less_than_node.outputs["Value"], vol_cube.inputs["Density"])
    # vol cube to vol_to_mesh
    node_group.links.new(vol_cube.outputs["Volume"], vol_to_mesh_node.inputs["Volume"])

    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(vol_to_mesh_node.outputs["Mesh"], material_node.inputs["Geometry"])
    node_seq = [vol_to_mesh_node, material_node]
    return node_seq

def create_inf_cone_node_seq(node_group, angle):

    vol_to_mesh_node = node_group.nodes.new(type="GeometryNodeVolumeToMesh")
    vol_to_mesh_node.inputs[3].default_value = 0.001
    vol_cube = node_group.nodes.new(type="GeometryNodeVolumeCube")
    vol_cube.inputs['Resolution X'].default_value = SDF_RESOLUTION
    vol_cube.inputs['Resolution Y'].default_value = SDF_RESOLUTION
    vol_cube.inputs['Resolution Z'].default_value = SDF_RESOLUTION
    # Maths
    position_node = node_group.nodes.new(type="GeometryNodeInputPosition")
    # separate XYZ
    sep_node = node_group.nodes.new(type="ShaderNodeSeparateXYZ")
    node_group.links.new(position_node.outputs["Position"], sep_node.inputs[0])
    # combine x y
    comb_node = node_group.nodes.new(type="ShaderNodeCombineXYZ")
    node_group.links.new(sep_node.outputs["X"], comb_node.inputs[0])
    node_group.links.new(sep_node.outputs["Y"], comb_node.inputs[1])
    # length
    len_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    len_node.operation = "LENGTH"
    node_group.links.new(comb_node.outputs["Vector"], len_node.inputs[0])
    # combine this length with origin z in com xyz
    q = node_group.nodes.new(type="ShaderNodeCombineXYZ")
    node_group.links.new(len_node.outputs["Value"], q.inputs[0])
    node_group.links.new(sep_node.outputs["Z"], q.inputs[1])

    # cos node
    cos_node = node_group.nodes.new(type="ShaderNodeMath")
    cos_node.operation = "COSINE"
    cos_node.inputs[0].default_value = angle[0]
    # sine node
    sin_node = node_group.nodes.new(type="ShaderNodeMath")
    sin_node.operation = "SINE"
    sin_node.inputs[0].default_value = angle[0]
    # create c by combining cos and sin
    c = node_group.nodes.new(type="ShaderNodeCombineXYZ")
    node_group.links.new(cos_node.outputs["Value"], c.inputs[0])
    node_group.links.new(sin_node.outputs["Value"], c.inputs[1])
    # multi q and c
    multi_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    multi_node.operation = "MULTIPLY"
    node_group.links.new(q.outputs["Vector"], multi_node.inputs[0])
    node_group.links.new(c.outputs["Vector"], multi_node.inputs[1])
    # separate into x y z
    sep_node_2 = node_group.nodes.new(type="ShaderNodeSeparateXYZ")
    node_group.links.new(multi_node.outputs["Vector"], sep_node_2.inputs[0])
    # add x and y
    sum_node = node_group.nodes.new(type="ShaderNodeMath")
    sum_node.operation = "ADD"
    node_group.links.new(sep_node_2.outputs["X"], sum_node.inputs[0])
    node_group.links.new(sep_node_2.outputs["Y"], sum_node.inputs[1])
    # apply as max of 0 and sum_node output
    max_node = node_group.nodes.new(type="ShaderNodeMath")
    max_node.operation = "MAXIMUM"
    max_node.inputs[1].default_value = 0
    node_group.links.new(sum_node.outputs["Value"], max_node.inputs[0])
    # create comXYZ and put max_node output into x and y
    com_node = node_group.nodes.new(type="ShaderNodeCombineXYZ")
    node_group.links.new(max_node.outputs["Value"], com_node.inputs[0])
    node_group.links.new(max_node.outputs["Value"], com_node.inputs[1])
    # multiply com_node to c
    multi_node_2 = node_group.nodes.new(type="ShaderNodeVectorMath")
    multi_node_2.operation = "MULTIPLY"
    node_group.links.new(com_node.outputs["Vector"], multi_node_2.inputs[0])
    node_group.links.new(c.outputs["Vector"], multi_node_2.inputs[1])
    # subtract this from q
    sub_node_2 = node_group.nodes.new(type="ShaderNodeVectorMath")
    sub_node_2.operation = "SUBTRACT"
    node_group.links.new(q.outputs["Vector"], sub_node_2.inputs[0])
    node_group.links.new(multi_node_2.outputs["Vector"], sub_node_2.inputs[1])
    # take length
    len_node_2 = node_group.nodes.new(type="ShaderNodeVectorMath")
    len_node_2.operation = "LENGTH"
    node_group.links.new(sub_node_2.outputs["Vector"], len_node_2.inputs[0])
    # q[0] * c[1] < q[1] * c[0] ? len_node_2 * -1, else len_node_2
    multi_node_lhs = node_group.nodes.new(type="ShaderNodeMath")
    multi_node_lhs.operation = "MULTIPLY"
    # link to len_node
    node_group.links.new(len_node.outputs["Value"], multi_node_lhs.inputs[0])
    # link to c[1]
    node_group.links.new(sin_node.outputs["Value"], multi_node_lhs.inputs[1])
    multi_node_rhs = node_group.nodes.new(type="ShaderNodeMath")
    multi_node_rhs.operation = "MULTIPLY"
    # link to sep Z
    node_group.links.new(sep_node.outputs["Z"], multi_node_rhs.inputs[0])
    # link to c[0]
    node_group.links.new(cos_node.outputs["Value"], multi_node_rhs.inputs[1])
    # sub lhs - rhs
    sub_math_node = node_group.nodes.new(type="ShaderNodeMath")
    sub_math_node.operation = "SUBTRACT"
    node_group.links.new(multi_node_lhs.outputs["Value"], sub_math_node.inputs[0])
    node_group.links.new(multi_node_rhs.outputs["Value"], sub_math_node.inputs[1])
    # get sign
    sign_node = node_group.nodes.new(type="ShaderNodeMath")
    sign_node.operation = "SIGN"
    node_group.links.new(sub_math_node.outputs["Value"], sign_node.inputs[0])

    multi_node_3 = node_group.nodes.new(type="ShaderNodeMath")
    multi_node_3.operation = "MULTIPLY"
    # link to len_node_2
    node_group.links.new(len_node_2.outputs["Value"], multi_node_3.inputs[0])
    # link to sign
    node_group.links.new(sign_node.outputs["Value"], multi_node_3.inputs[1])

    # less than
    less_than_node = node_group.nodes.new(type="ShaderNodeMath")
    less_than_node.operation = "LESS_THAN"
    less_than_node.inputs[1].default_value = EPSILON
    node_group.links.new(multi_node_3.outputs["Value"], less_than_node.inputs[0])
    # link to vol cube
    node_group.links.new(less_than_node.outputs["Value"], vol_cube.inputs["Density"])
    # vol cube to vol_to_mesh
    node_group.links.new(vol_cube.outputs["Volume"], vol_to_mesh_node.inputs["Volume"])

    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(vol_to_mesh_node.outputs["Mesh"], material_node.inputs["Geometry"])
    node_seq = [vol_to_mesh_node, material_node]
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


def create_boolean_complement_node_seq(node_group):
    """
    Creates a sequence of nodes for performing a boolean union operation.

    Parameters:
        node_group (bpy.types.NodeTree): The node tree to which the nodes will be added.

    Returns:
        list: A list containing the boolean union node.
    """
    print("creating complement node seq")
    bool_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    bool_node.operation = "DIFFERENCE"
    draw_node = node_group.nodes.new(type="GeometryNodeMeshCube")
    draw_node.inputs["Size"].default_value = [1, 1, 1]
    node_group.links.new(draw_node.outputs["Mesh"], bool_node.inputs["Mesh 1"])
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
