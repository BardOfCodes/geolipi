import sys

def import_bpy():
    module_name = 'bpy'
    if module_name not in sys.modules:  
        print("Importing bpy...")
        # import bpy and assign to a variable
        import bpy
        print("Imported bpy.")
    else:
        print("bpy already imported.")
    return sys.modules[module_name]
        

def create_all_node_groups(dummy_obj):
    bpy = import_bpy()
    mod = dummy_obj.modifiers.new("CSG-GN", 'NODES')
    # Draw Commands:
    node_group = bpy.data.node_groups.new("draw_cuboid", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    input_node = node_group.nodes.new('NodeGroupInput')
    node_group.inputs.new('NodeSocketMaterial', "Material")
    draw_node = node_group.nodes.new(type="GeometryNodeMeshCube")
    draw_node.inputs['Size'].default_value = [0.5, 0.5, 0.5]
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(material_node.outputs['Geometry'], output_node.inputs["Geometry"])
    node_group.links.new(draw_node.outputs['Mesh'], material_node.inputs['Geometry'])
    node_group.links.new(input_node.outputs['Material'], material_node.inputs['Material'])

    node_group = bpy.data.node_groups.new("draw_sphere", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    input_node = node_group.nodes.new('NodeGroupInput')
    node_group.inputs.new('NodeSocketMaterial', "Material")
    draw_node = node_group.nodes.new(type="GeometryNodeMeshUVSphere")
    draw_node.inputs['Radius'].default_value = 0.5
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(material_node.outputs['Geometry'], output_node.inputs["Geometry"])
    node_group.links.new(draw_node.outputs['Mesh'], material_node.inputs['Geometry'])
    node_group.links.new(input_node.outputs['Material'], material_node.inputs['Material'])

    node_group = bpy.data.node_groups.new("draw_cylinder", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    input_node = node_group.nodes.new('NodeGroupInput')
    node_group.inputs.new('NodeSocketMaterial', "Material")
    draw_node = node_group.nodes.new(type="GeometryNodeMeshCylinder")
    draw_node.inputs['Radius'].default_value = 0.5
    draw_node.inputs['Depth'].default_value = 1
    material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
    node_group.links.new(material_node.outputs['Geometry'], output_node.inputs["Geometry"])
    node_group.links.new(draw_node.outputs['Mesh'], material_node.inputs['Geometry'])
    node_group.links.new(input_node.outputs['Material'], material_node.inputs['Material'])
    
    # booleans
    node_group = bpy.data.node_groups.new("boolean_union", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    input_node = node_group.nodes.new('NodeGroupInput')
    node_group.inputs.new('NodeSocketGeometry', "Geometry 1")
    node_group.inputs.new('NodeSocketGeometry', "Geometry 2")
    bool_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    bool_node.operation = "UNION"
    node_group.links.new(input_node.outputs['Geometry 1'], bool_node.inputs[1])
    node_group.links.new(input_node.outputs['Geometry 2'], bool_node.inputs[1])
    node_group.links.new(bool_node.outputs['Mesh'], output_node.inputs['Geometry'])

    node_group = bpy.data.node_groups.new("boolean_intersection", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    input_node = node_group.nodes.new('NodeGroupInput')
    node_group.inputs.new('NodeSocketGeometry', "Geometry 1")
    node_group.inputs.new('NodeSocketGeometry', "Geometry 2")
    bool_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    bool_node.operation = "INTERSECT"
    node_group.links.new(input_node.outputs['Geometry 1'], bool_node.inputs[1])
    node_group.links.new(input_node.outputs['Geometry 2'], bool_node.inputs[1])
    node_group.links.new(bool_node.outputs['Mesh'], output_node.inputs['Geometry'])

    node_group = bpy.data.node_groups.new("boolean_difference", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    input_node = node_group.nodes.new('NodeGroupInput')
    node_group.inputs.new('NodeSocketGeometry', "Geometry 1")
    node_group.inputs.new('NodeSocketGeometry', "Geometry 2")
    bool_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    bool_node.operation = "DIFFERENCE"
    node_group.links.new(input_node.outputs['Geometry 1'], bool_node.inputs["Mesh 2"])
    node_group.links.new(input_node.outputs['Geometry 2'], bool_node.inputs["Mesh 1"])
    node_group.links.new(bool_node.outputs['Mesh'], output_node.inputs['Geometry'])

    # transforms:
    node_group = bpy.data.node_groups.new("transform_translate", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    input_node = node_group.nodes.new('NodeGroupInput')
    node_group.inputs.new('NodeSocketGeometry', "Geometry")
    node_group.inputs.new('NodeSocketVector', "Parameters")
    transform_node = node_group.nodes.new(type="GeometryNodeTransform")
    node_group.links.new(input_node.outputs['Geometry'], transform_node.inputs["Geometry"])
    node_group.links.new(input_node.outputs['Parameters'], transform_node.inputs["Translation"])
    node_group.links.new(transform_node.outputs['Geometry'], output_node.inputs['Geometry'])

    node_group = bpy.data.node_groups.new("transform_scale", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    input_node = node_group.nodes.new('NodeGroupInput')
    node_group.inputs.new('NodeSocketGeometry', "Geometry")
    node_group.inputs.new('NodeSocketVector', "Parameters")
    transform_node = node_group.nodes.new(type="GeometryNodeTransform")
    node_group.links.new(input_node.outputs['Geometry'], transform_node.inputs["Geometry"])
    node_group.links.new(input_node.outputs['Parameters'], transform_node.inputs["Scale"])
    node_group.links.new(transform_node.outputs['Geometry'], output_node.inputs['Geometry'])

    node_group = bpy.data.node_groups.new("transform_rotate", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    input_node = node_group.nodes.new('NodeGroupInput')
    node_group.inputs.new('NodeSocketGeometry', "Geometry")
    node_group.inputs.new('NodeSocketVector', "Parameters")
    transform_node = node_group.nodes.new(type="GeometryNodeTransform")
    node_group.links.new(input_node.outputs['Geometry'], transform_node.inputs["Geometry"])
    math_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    math_node.operation = "MULTIPLY"
    math_node.inputs[1].default_value = [-1., -1., -1.]
    node_group.links.new(math_node.outputs[0], transform_node.inputs["Rotation"])
    node_group.links.new(input_node.outputs['Parameters'], math_node.inputs[0])
    # node_group.links.new(input_node.outputs['Parameters'], transform_node.inputs["Rotation"])
    node_group.links.new(transform_node.outputs['Geometry'], output_node.inputs['Geometry'])


def preview_node_groups(dummy_obj):
    # Mirror:
    bpy = import_bpy()
    node_group = bpy.data.node_groups.new("mirror", 'GeometryNodeTree')
    node_group.outputs.new('NodeSocketGeometry', "Geometry")
    output_node = node_group.nodes.new('NodeGroupOutput')
    input_node = node_group.nodes.new('NodeGroupInput')
    node_group.inputs.new('NodeSocketGeometry', "Geometry")
    node_group.inputs.new('NodeSocketVector', "Parameters")

    #join_node = node_group.nodes.new(type="GeometryNodeJoinGeometry")
    join_node = node_group.nodes.new(type="GeometryNodeMeshBoolean")
    join_node.operation = "UNION"
    set_position_node = node_group.nodes.new(type="GeometryNodeSetPosition")
    position_node = node_group.nodes.new(type="GeometryNodeInputPosition")
    vector_math_node = node_group.nodes.new(type="ShaderNodeVectorMath")
    vector_math_node.operation = "REFLECT"

    node_group.links.new(join_node.outputs['Mesh'], output_node.inputs['Geometry'])
    node_group.links.new(input_node.outputs['Geometry'], join_node.inputs[0])
    node_group.links.new(set_position_node.outputs['Geometry'], join_node.inputs[0])
    node_group.links.new(input_node.outputs['Geometry'], set_position_node.inputs['Geometry'])
    node_group.links.new(vector_math_node.outputs['Vector'], set_position_node.inputs['Position'])
    node_group.links.new(position_node.outputs['Position'], vector_math_node.inputs[0])
    node_group.links.new(input_node.outputs['Parameters'],  vector_math_node.inputs[1])

