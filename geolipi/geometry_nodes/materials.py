from .utils import import_bpy
# ref: Derek

class colorObj(object):

    def __init__(self, RGBA, H=0.5, S=1.0, V=1.0,
                 B=0.0, C=0.0):
        self.H = H  # hue
        self.S = S  # saturation
        self.V = V  # value
        self.RGBA = RGBA
        self.B = B  # birghtness
        self.C = C  # contrast

# For .gltf exports
def create_simple_material_tree(color, material_name):
    bpy = import_bpy()
    RGBA = (color[0], color[1],  color[2], 1)
    mesh_color = colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 1.0)

    mat = bpy.data.materials.new(material_name)

    mat.use_nodes = True
    tree = mat.node_tree
    # set principled BSDF
    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.3
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
    tree.nodes["Principled BSDF"].inputs['Specular'].default_value = 0.5
    tree.nodes["Principled BSDF"].inputs['IOR'].default_value = 1.45
    tree.nodes["Principled BSDF"].inputs['Transmission'].default_value = 0
    tree.nodes["Principled BSDF"].inputs['Clearcoat Roughness'].default_value = 0
    tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = mesh_color.RGBA

    return mat

def create_material_tree(color, material_name):
    bpy = import_bpy()
    RGBA = (color[0], color[1],  color[2], 1)
    mesh_color = colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)

    mat = bpy.data.materials.new(material_name)

    mat.use_nodes = True
    tree = mat.node_tree
    ao_strength = 0.0
    # set principled BSDF
    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.3
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
    tree.nodes["Principled BSDF"].inputs['Specular'].default_value = 0.5
    tree.nodes["Principled BSDF"].inputs['IOR'].default_value = 1.45
    tree.nodes["Principled BSDF"].inputs['Transmission'].default_value = 0
    tree.nodes["Principled BSDF"].inputs['Clearcoat Roughness'].default_value = 0

    # add Ambient Occlusion
    tree.nodes.new('ShaderNodeAmbientOcclusion')
    tree.nodes.new('ShaderNodeGamma')
    mix_rgb_node = tree.nodes.new('ShaderNodeMixRGB')
    mix_rgb_node.blend_type = 'MULTIPLY'
    tree.nodes["Gamma"].inputs["Gamma"].default_value = ao_strength
    tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
    tree.nodes["Gamma"].location.x -= 600

    # set color using Hue/Saturation node
    hsv_node = tree.nodes.new('ShaderNodeHueSaturation')
    hsv_node.inputs['Color'].default_value = mesh_color.RGBA
    hsv_node.inputs['Saturation'].default_value = mesh_color.S
    hsv_node.inputs['Value'].default_value = mesh_color.V
    hsv_node.inputs['Hue'].default_value = mesh_color.H
    hsv_node.location.x -= 200

    # set color brightness/contrast
    bc_node = tree.nodes.new('ShaderNodeBrightContrast')
    bc_node.inputs['Bright'].default_value = mesh_color.B
    bc_node.inputs['Contrast'].default_value = mesh_color.C
    bc_node.location.x -= 400

    # link all the nodes
    tree.links.new(hsv_node.outputs['Color'], bc_node.inputs['Color'])
    tree.links.new(bc_node.outputs['Color'], tree.nodes['Ambient Occlusion'].inputs['Color'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], mix_rgb_node.inputs['Color1'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
    tree.links.new(tree.nodes["Gamma"].outputs['Color'], mix_rgb_node.inputs['Color2'])
    tree.links.new(mix_rgb_node.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

    return mat


def create_edge_material_tree(material_name, mesh_color, edge_thickness, 
                              edge_color=(0, 0, 0), 
                              ao_strength=1.0):
    rgba = (mesh_color[0], mesh_color[1],  mesh_color[2], 1)
    mesh_color = colorObj(rgba, 0.5, 1.0, 1.0, 0.0, 1.0)
    rgba = (edge_color[0], edge_color[1],  edge_color[2], 1)
    edge_color = colorObj(rgba, 0.5, 1.0, 1.0, 0.0, 1.0)
    
    bpy = import_bpy()
    mat = bpy.data.materials.new(material_name)
    mat.use_nodes = True
    tree = mat.node_tree
    # set principled BSDF
    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.7
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
    # add Ambient Occlusion
    tree.nodes.new('ShaderNodeAmbientOcclusion')
    tree.nodes.new('ShaderNodeGamma')
    mix_rgb = tree.nodes.new('ShaderNodeMixRGB')
    mix_rgb.blend_type = 'MULTIPLY'
    tree.nodes["Gamma"].inputs["Gamma"].default_value = ao_strength
    tree.nodes["Gamma"].location.x -= 600
    tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
    tree.nodes["Ambient Occlusion"].inputs["Color"].default_value = mesh_color
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], mix_rgb.inputs['Color1'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
    tree.links.new(tree.nodes["Gamma"].outputs['Color'], mix_rgb.inputs['Color2'])
    tree.links.new(mix_rgb.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
    # add edge wireframe
    tree.nodes.new(type="ShaderNodeWireframe")
    wire = tree.nodes[-1]
    wire.inputs[0].default_value = edge_thickness
    wire.location.x -= 200
    wire.location.y -= 200
    tree.nodes.new(type="ShaderNodeBsdfDiffuse")
    mat_wire = tree.nodes[-1]
    hsv_node = tree.nodes.new('ShaderNodeHueSaturation')
    hsv_node.inputs['Color'].default_value = edge_color.RGBA
    hsv_node.inputs['Saturation'].default_value = edge_color.S
    hsv_node.inputs['Value'].default_value = edge_color.V
    hsv_node.inputs['Hue'].default_value = edge_color.H
    hsv_node.location.x -= 200
    # set color brightness/contrast
    bc_node = tree.nodes.new('ShaderNodeBrightContrast')
    bc_node.inputs['Bright'].default_value = edge_color.B
    bc_node.inputs['Contrast'].default_value = edge_color.C
    bc_node.location.x -= 400
    tree.links.new(hsv_node.outputs['Color'],bc_node.inputs['Color'])
    tree.links.new(bc_node.outputs['Color'],mat_wire.inputs['Color'])
    tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(wire.outputs[0], tree.nodes['Mix Shader'].inputs[0])
    tree.links.new(mat_wire.outputs['BSDF'], tree.nodes['Mix Shader'].inputs[2])
    tree.links.new(tree.nodes["Principled BSDF"].outputs['BSDF'], tree.nodes['Mix Shader'].inputs[1])
    tree.links.new(tree.nodes["Mix Shader"].outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])
    
    return mat
    

def create_monotone_material(material_name, mesh_color, color_list, silhouette_color, shadow_size):
    bpy = import_bpy()
    mat = bpy.data.materials.new(material_name)
    mat.use_nodes = True
    tree = mat.node_tree
    rgba = (mesh_color[0], mesh_color[1],  mesh_color[2], 1)
    mesh_color = colorObj(rgba, 0.5, 1.0, 1.0, 0.0, 1.0)
    rgba = (silhouette_color[0], silhouette_color[1],  silhouette_color[2], 1)
    silhouette_color = colorObj(rgba, 0.5, 1.0, 1.0, 0.0, 1.0)
    new_color_list = []
    for color in color_list:
        rgba = (color[0], color[1],  color[2], 1)
        new_color_list.append(colorObj(rgba, 0.5, 1.0, 1.0, 0.0, 1.0))
    color_list = new_color_list
    
    num_color = len(color_list) # num_color >= 3

    # set principled	
    principle_node = tree.nodes["Principled BSDF"]
    principle_node.inputs['Roughness'].default_value = 1.0
    principle_node.inputs['Sheen Tint'].default_value = 0
    principle_node.inputs['Specular'].default_value = 0.0

    # init level
    init_rgb = tree.nodes.new('ShaderNodeHueSaturation')
    init_rgb.inputs['Color'].default_value = (.5,.5,.5,1)
    init_rgb.inputs['Hue'].default_value = 0
    init_rgb.inputs['Saturation'].default_value = 0
    init_rgb.inputs['Value'].default_value = color_list[0].brightness
    init_diff = tree.nodes.new('ShaderNodeBsdfDiffuse')
    tree.links.new(init_rgb.outputs['Color'], init_diff.inputs['Color'])

    # init array
    RGB_list = [None] * (num_color - 1)
    ramp_list = [None] * (num_color - 1)
    mix_list = [None] * (num_color - 1)
    diff_list = [None] * (num_color - 1)
    for ii in range(num_color-1):
        # RGB node
        RGB_list[ii] = tree.nodes.new('ShaderNodeHueSaturation')
        RGB_list[ii].inputs['Color'].default_value = (.5,.5,.5,1)
        RGB_list[ii].inputs['Hue'].default_value = 0
        RGB_list[ii].inputs['Saturation'].default_value = 0
        RGB_list[ii].inputs['Value'].default_value = color_list[ii+1].brightness
        # Diffuse after RGB
        diff_list[ii] = tree.nodes.new('ShaderNodeBsdfDiffuse')
        # Color Ramp
        ramp_list[ii] = tree.nodes.new('ShaderNodeValToRGB')
        ramp_list[ii].color_ramp.interpolation = 'EASE'
        ramp_list[ii].color_ramp.elements.new(0.5)
        # ramp_list[ii].color_ramp.elements[1].position = color_list[ii+1].rampElement1_pos
        ramp_list[ii].color_ramp.elements[1].color = (0,0,0,1)
        # ramp_list[ii].color_ramp.elements[2].position = color_list[ii+1].rampElement2_pos
        # Mix shader
        mix_list[ii] = tree.nodes.new('ShaderNodeMixShader')
        # Link shaders
        if ii > 0 and ii < (num_color-1):
            tree.links.new(mix_list[ii-1].outputs['Shader'], mix_list[ii].inputs[1])
        tree.links.new(ramp_list[ii].outputs['Color'], mix_list[ii].inputs[0])
        tree.links.new(RGB_list[ii].outputs['Color'], diff_list[ii].inputs['Color'])
        tree.links.new(diff_list[ii].outputs['BSDF'], mix_list[ii].inputs[2])
        # set node location

    # color of the mesh
    mainColor_HSV = tree.nodes.new('ShaderNodeHueSaturation')
    mainColor_HSV.inputs['Color'].default_value = mesh_color.RGBA
    mainColor_HSV.inputs['Hue'].default_value = mesh_color.H
    mainColor_HSV.inputs['Saturation'].default_value = mesh_color.S
    mainColor_HSV.inputs['Value'].default_value = mesh_color.V

    # main color BC
    mainColor = tree.nodes.new('ShaderNodeBrightContrast')
    mainColor.inputs['Bright'].default_value = mesh_color.B
    mainColor.inputs['Contrast'].default_value = mesh_color.C
    tree.links.new(mainColor_HSV.outputs['Color'], mainColor.inputs['Color'])

    # initial and end links
    addShader = tree.nodes.new('ShaderNodeAddShader')
    tree.links.new(init_diff.outputs['BSDF'], mix_list[0].inputs[1])
    tree.links.new(mix_list[-1].outputs['Shader'], addShader.inputs[0])
    tree.links.new(mainColor.outputs['Color'], principle_node.inputs['Base Color'])
    tree.links.new(principle_node.outputs['BSDF'], addShader.inputs[1])

    # add silhouette 
    mixEnd = tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(mixEnd.outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])

    edgeShadow_HSV = tree.nodes.new('ShaderNodeHueSaturation')
    edgeShadow_HSV.inputs['Color'].default_value = silhouette_color.RGBA
    edgeShadow_HSV.inputs['Hue'].default_value = silhouette_color.H
    edgeShadow_HSV.inputs['Saturation'].default_value = silhouette_color.S
    edgeShadow_HSV.inputs['Value'].default_value = silhouette_color.V

    edgeShadow = tree.nodes.new('ShaderNodeBrightContrast')
    edgeShadow.inputs['Bright'].default_value = silhouette_color.B
    edgeShadow.inputs['Contrast'].default_value = silhouette_color.C
    
    tree.links.new(edgeShadow_HSV.outputs['Color'], edgeShadow.inputs['Color'])

    diffEnd = tree.nodes.new('ShaderNodeBsdfDiffuse')
    tree.links.new(edgeShadow.outputs['Color'], diffEnd.inputs['Color'])
    tree.links.new(diffEnd.outputs['BSDF'], mixEnd.inputs[2])

    fresnelEnd = tree.nodes.new('ShaderNodeFresnel')
    RampEnd = tree.nodes.new('ShaderNodeValToRGB')
    RampEnd.color_ramp.elements[1].position = shadow_size
    tree.links.new(fresnelEnd.outputs[0], RampEnd.inputs['Fac'])
    tree.links.new(RampEnd.outputs['Color'], mixEnd.inputs[0])
    tree.links.new(addShader.outputs[0], mixEnd.inputs[1])

    # add normal to the color
    fresnelNode = tree.nodes.new('ShaderNodeFresnel')
    textureNode = tree.nodes.new('ShaderNodeTexCoord')
    tree.links.new(textureNode.outputs['Normal'], fresnelNode.inputs['Normal'])
    for ii in range(len(ramp_list)):
        tree.links.new(fresnelNode.outputs[0], ramp_list[ii].inputs['Fac'])

    # set node location
    for node in tree.nodes:
        node.location.x = 0
        node.location.y = 0
    yLoc = 0
    xLoc = -400
    RampEnd.location.x = xLoc
    RampEnd.location.y = -300
    for node in ramp_list:
        node.location.x = xLoc
        node.location.y = yLoc
        yLoc += 300
    yLoc = 0
    xLoc = -600
    init_rgb.location.x = xLoc
    init_rgb.location.y = -200
    for node in RGB_list:
        node.location.x = xLoc
        node.location.y = yLoc
        yLoc += 200

    mainColor.location.x = -800
    mainColor.location.y = 0
    mainColor_HSV.location.x = -1000
    mainColor_HSV.location.y = 0
    edgeShadow.location.x = -800
    edgeShadow.location.y = 200
    edgeShadow_HSV.location.x = -1000
    edgeShadow_HSV.location.y = 200
    return mat