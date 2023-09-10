import bpy
import numpy as np
import math
from iccv_render.bt_utils import create_material



class CSGVisualizer:
    
    def __init__(self, dummy_obj, colors):
        
        # Create all the node groups:
        self.create_material_per_primitive = True
        self.load_language_specific_details()
        
        self.create_all_node_groups(dummy_obj)
        self.colors = colors
    
    
    def create_all_node_groups(self, dummy_obj):

        mod = dummy_obj.modifiers.new("CSG-GN", 'NODES')
        # Draw Commands:
        node_group = bpy.data.node_groups.new("draw_cuboid", 'GeometryNodeTree')
        node_group.outputs.new('NodeSocketGeometry', "Geometry")
        output_node = node_group.nodes.new('NodeGroupOutput')
        input_node = node_group.nodes.new('NodeGroupInput')
        node_group.inputs.new('NodeSocketMaterial', "Material")
        draw_node = node_group.nodes.new(type="GeometryNodeMeshCube")
        draw_node.inputs['Size'].default_value = [1, 1, 1]
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
        math_node.inputs[1].default_value = [-math.pi / 180., -math.pi / 180., -math.pi / 180.]
        node_group.links.new(math_node.outputs[0], transform_node.inputs["Rotation"])
        node_group.links.new(input_node.outputs['Parameters'], math_node.inputs[0])
        # node_group.links.new(input_node.outputs['Parameters'], transform_node.inputs["Rotation"])
        node_group.links.new(transform_node.outputs['Geometry'], output_node.inputs['Geometry'])

        # Mirror:
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

    def load_language_specific_details(self):
        self.command_n_param = {
            "sphere": 0,
            "cylinder": 0,
            "cuboid": 0,
            "translate": 3,
            "rotate": 3,
            "scale": 3,
            "union": 0,
            "intersection": 0,
            "difference": 0,
            "mirror": 3,
            # For FCSG
            "rotate_sphere": 3,
            "rotate_cylinder": 3,
            "rotate_cuboid": 3,
        }
        self.command_symbol_to_type = {
            "sphere": "D",
            "cylinder": "D",
            "cuboid": "D",
            "translate": "T",
            "rotate": "T",
            "scale": "T",
            "union": "B",
            "intersection": "B",
            "difference": "B",
            "mirror": "M",
            "rotate_sphere": "RD",
            "rotate_cylinder": "RD",
            "rotate_cuboid": "RD",
        }
        self.mirror_params = {
            "MIRROR_X": [1., 0., 0.],
            "MIRROR_Y": [0., 1., 0.],
            "MIRROR_Z": [0., 0., 1.],
        }
        self.invalid_commands = []

        self.trivial_expression = ["sphere", "$"]
        self.has_transform_commands = True
        
        self.fixed_macros = dict()
        # Fixed mirror commands
        command_type = "M"
        command_symbol = "mirror"
        for key, value in self.mirror_params.items():
            command = [{'type': command_type, "symbol": command_symbol, 'param': value, 'macro_mode': "macro(%s)" % key}]
            self.fixed_macros[key] = command
            
    def parse(self, expression_list):
        command_list = []
        for expr in expression_list:

            command_symbol = expr.split("(")[0]
            if "#" in command_symbol:
                continue
            elif command_symbol == "$":
                # END OF PROGRAM
                break
            elif command_symbol == "macro":
                macro_name = expr.split("(")[1][:-1]
                command_list.extend(self.fixed_macros[macro_name])
            else:
                command_type = self.command_symbol_to_type[command_symbol]
                command_dict = {'type': command_type, "symbol": command_symbol}
                n_param = self.command_n_param[command_symbol]
                if n_param > 0:
                    param_str = expr.split("(")[1][:-1]
                    param = [float(x.strip()) for x in param_str.split(",")]
                    command_dict['param'] = param
                command_list.append(command_dict)
        return command_list
    
    
    # convert all expressions to MCSG format
    def compile(self, command_list, dummy_obj):
        
        mod = dummy_obj.modifiers.new("CSG-GN", 'NODES')
        
        node_group = bpy.data.node_groups.new("CSG-Tree", 'GeometryNodeTree')
        node_group.outputs.new('NodeSocketGeometry', "Geometry")
        output_node = node_group.nodes.new('NodeGroupOutput')
        output_node.is_active_output = True
        output_node.select = False
        mod.node_group = node_group
        
        output_node = node_group.nodes['Group Output']
        link_stack = [output_node.inputs['Geometry']]
        node_list = []
        mat_id = 0
        
        for cmd in command_list:
            if cmd['type'] == "B":
                bool_node = node_group.nodes.new(type="GeometryNodeGroup")
                bool_node.node_tree = bpy.data.node_groups['boolean_%s' % cmd['symbol']]
                
                node_list.append(bool_node)
                # Linking:
                outer_input = link_stack.pop()
                node_output = bool_node.outputs['Geometry']
                node_group.links.new(node_output, outer_input)
                
                link_stack.append(bool_node.inputs['Geometry 1'])
                link_stack.append(bool_node.inputs['Geometry 2'])
            elif cmd['type'] == "D":
                draw_node = node_group.nodes.new(type="GeometryNodeGroup")
                draw_node.node_tree = bpy.data.node_groups['draw_%s' % cmd['symbol']]
                node_list.append(draw_node)
                outer_input = link_stack.pop()
                node_output = draw_node.outputs['Geometry']
                
                if self.create_material_per_primitive:
                    self.create_new_material(mat_id)
                    mat = bpy.data.materials["material_%d" % (mat_id)]
                    draw_node.inputs['Material'].default_value = mat
                    mat_id += 1
                node_group.links.new(node_output, outer_input)
                
            elif cmd['type'] == "T":
                transform_node = node_group.nodes.new(type="GeometryNodeGroup")
                transform_node.node_tree = bpy.data.node_groups['transform_%s' % cmd['symbol']]
                transform_node.inputs['Parameters'].default_value = cmd['param']
                # Linking:
                outer_input = link_stack.pop()
                node_output = transform_node.outputs['Geometry']
                node_group.links.new(node_output, outer_input)
                
                link_stack.append(transform_node.inputs['Geometry'])
                
            elif cmd['type'] == "M":
                transform_node = node_group.nodes.new(type="GeometryNodeGroup")
                transform_node.node_tree = bpy.data.node_groups['mirror']
                transform_node.inputs['Parameters'].default_value = cmd['param']
                # Linking:
                outer_input = link_stack.pop()
                node_output = transform_node.outputs['Geometry']
                node_group.links.new(node_output, outer_input)
                
                link_stack.append(transform_node.inputs['Geometry'])
    
    def create_new_material(self, mat_id):
        
        color = self.colors[-mat_id]
        name="material_%d" % (mat_id)
        create_material(color, name)
        
        # mat = bpy.data.materials.new(name="material_%d" % (mat_id))
        # mat.use_nodes = True
        # mat_node_tree = mat.node_tree
        # mat_nodes = mat_node_tree.nodes
        # bsdf = mat_nodes.get("Principled BSDF") 
        # color = self.colors[-mat_id]
        # bsdf.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1)