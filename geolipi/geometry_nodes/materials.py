from .utils import import_bpy
# Credits: Extracted from Derek Hsu's code: https://github.com/HTDerekLiu/BlenderToolbox

class colorObj(object):
    """
    Represents a color object with various color properties.

    Parameters:
        RGBA (tuple): The RGBA color values.
        H (float, optional): Hue component of the color. Defaults to 0.5.
        S (float, optional): Saturation component of the color. Defaults to 1.0.
        V (float, optional): Value (brightness) component of the color. Defaults to 1.0.
        B (float, optional): Additional brightness factor. Defaults to 0.0.
        C (float, optional): Contrast factor of the color. Defaults to 0.0.
    """

    def __init__(self, RGBA, H=0.5, S=1.0, V=1.0,
                 B=0.0, C=0.0):
        self.H = H  # hue
        self.S = S  # saturation
        self.V = V  # value
        self.RGBA = RGBA
        self.B = B  # birghtness
        self.C = C  # contrast

def create_simple_material_tree(material_name, color):
    """
    Creates a simple material node tree suitable for .gltf exports.

    Parameters:
        material_name (str): Name of the new material.
        color (tuple): RGB color values for the material.

    Returns:
        bpy.types.Material: The newly created Blender material.
    """
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

def create_material_tree(material_name, color):
    """
    Creates a complex material node tree for the renders in CoReF paper (ICCV 2023).

    Parameters:
        material_name (str): Name of the new material.
        color (tuple): RGB color values for the material.

    Returns:
        bpy.types.Material: The newly created Blender material.
    """
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
    # set color using Hue/Saturation node
    hsv_node = tree.nodes.new('ShaderNodeHueSaturation')
    hsv_node.inputs['Color'].default_value = mesh_color.RGBA
    hsv_node.inputs['Saturation'].default_value = mesh_color.S
    hsv_node.inputs['Value'].default_value = mesh_color.V
    hsv_node.inputs['Hue'].default_value = mesh_color.H

    # set color brightness/contrast
    bc_node = tree.nodes.new('ShaderNodeBrightContrast')
    bc_node.inputs['Bright'].default_value = mesh_color.B
    bc_node.inputs['Contrast'].default_value = mesh_color.C
    # link all the nodes
    tree.links.new(hsv_node.outputs['Color'], bc_node.inputs['Color'])
    tree.links.new(bc_node.outputs['Color'], tree.nodes['Ambient Occlusion'].inputs['Color'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], mix_rgb_node.inputs['Color1'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
    tree.links.new(tree.nodes["Gamma"].outputs['Color'], mix_rgb_node.inputs['Color2'])
    tree.links.new(mix_rgb_node.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

    return mat


def create_edge_material_tree(material_name, mesh_color, edge_thickness=0.001, 
                              edge_color=(0, 0, 0), 
                              ao_strength=1.0):
    """
    Creates a material node tree with edge highlighting and ambient occlusion.

    Parameters:
        material_name (str): Name of the new material.
        mesh_color (tuple): RGB color values for the mesh.
        edge_thickness (float, optional): Thickness of the edges. Defaults to 0.001.
        edge_color (tuple, optional): RGB color values for the edges. Defaults to black.
        ao_strength (float, optional): Strength of the ambient occlusion effect. Defaults to 1.0.

    Returns:
        bpy.types.Material: The newly created Blender material.
    """
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
    tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
    tree.nodes["Ambient Occlusion"].inputs["Color"].default_value = mesh_color.RGBA
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], mix_rgb.inputs['Color1'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
    tree.links.new(tree.nodes["Gamma"].outputs['Color'], mix_rgb.inputs['Color2'])
    tree.links.new(mix_rgb.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
    # add edge wireframe
    tree.nodes.new(type="ShaderNodeWireframe")
    wire = tree.nodes[-1]
    wire.inputs[0].default_value = edge_thickness
    tree.nodes.new(type="ShaderNodeBsdfDiffuse")
    mat_wire = tree.nodes[-1]
    hsv_node = tree.nodes.new('ShaderNodeHueSaturation')
    hsv_node.inputs['Color'].default_value = edge_color.RGBA
    hsv_node.inputs['Saturation'].default_value = edge_color.S
    hsv_node.inputs['Value'].default_value = edge_color.V
    hsv_node.inputs['Hue'].default_value = edge_color.H
    # set color brightness/contrast
    bc_node = tree.nodes.new('ShaderNodeBrightContrast')
    bc_node.inputs['Bright'].default_value = edge_color.B
    bc_node.inputs['Contrast'].default_value = edge_color.C
    tree.links.new(hsv_node.outputs['Color'],bc_node.inputs['Color'])
    tree.links.new(bc_node.outputs['Color'],mat_wire.inputs['Color'])
    tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(wire.outputs[0], tree.nodes['Mix Shader'].inputs[0])
    tree.links.new(mat_wire.outputs['BSDF'], tree.nodes['Mix Shader'].inputs[2])
    tree.links.new(tree.nodes["Principled BSDF"].outputs['BSDF'], tree.nodes['Mix Shader'].inputs[1])
    tree.links.new(tree.nodes["Mix Shader"].outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])
    
    return mat
    

def create_monotone_material(material_name, mesh_color, color_list, silhouette_color, shadow_size=0.1):
    """
    Creates a monotone material with multiple color levels, silhouette, and shadow effects.

    Parameters:
        material_name (str): Name of the new material.
        mesh_color (tuple): RGB color values for the mesh.
        color_list (list): List of RGB color tuples for different color levels.
        silhouette_color (tuple): RGB color values for the silhouette effect.
        shadow_size (float, optional): Size of the shadow effect. Defaults to 0.1.

    Returns:
        bpy.types.Material: The newly created Blender material.
    """
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
    init_rgb.inputs['Value'].default_value = color_list[0].B
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
        RGB_list[ii].inputs['Value'].default_value = color_list[ii+1].B
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
    main_color_HSV = tree.nodes.new('ShaderNodeHueSaturation')
    main_color_HSV.inputs['Color'].default_value = mesh_color.RGBA
    main_color_HSV.inputs['Hue'].default_value = mesh_color.H
    main_color_HSV.inputs['Saturation'].default_value = mesh_color.S
    main_color_HSV.inputs['Value'].default_value = mesh_color.V

    # main color BC
    main_color = tree.nodes.new('ShaderNodeBrightContrast')
    main_color.inputs['Bright'].default_value = mesh_color.B
    main_color.inputs['Contrast'].default_value = mesh_color.C
    tree.links.new(main_color_HSV.outputs['Color'], main_color.inputs['Color'])

    # initial and end links
    add_shader = tree.nodes.new('ShaderNodeAddShader')
    tree.links.new(init_diff.outputs['BSDF'], mix_list[0].inputs[1])
    tree.links.new(mix_list[-1].outputs['Shader'], add_shader.inputs[0])
    tree.links.new(main_color.outputs['Color'], principle_node.inputs['Base Color'])
    tree.links.new(principle_node.outputs['BSDF'], add_shader.inputs[1])

    # add silhouette 
    mix_end = tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(mix_end.outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])

    edge_shadow_HSV = tree.nodes.new('ShaderNodeHueSaturation')
    edge_shadow_HSV.inputs['Color'].default_value = silhouette_color.RGBA
    edge_shadow_HSV.inputs['Hue'].default_value = silhouette_color.H
    edge_shadow_HSV.inputs['Saturation'].default_value = silhouette_color.S
    edge_shadow_HSV.inputs['Value'].default_value = silhouette_color.V

    edge_shadow = tree.nodes.new('ShaderNodeBrightContrast')
    edge_shadow.inputs['Bright'].default_value = silhouette_color.B
    edge_shadow.inputs['Contrast'].default_value = silhouette_color.C
    
    tree.links.new(edge_shadow_HSV.outputs['Color'], edge_shadow.inputs['Color'])

    diff_end = tree.nodes.new('ShaderNodeBsdfDiffuse')
    tree.links.new(edge_shadow.outputs['Color'], diff_end.inputs['Color'])
    tree.links.new(diff_end.outputs['BSDF'], mix_end.inputs[2])

    fresnel_end = tree.nodes.new('ShaderNodeFresnel')
    ramp_end = tree.nodes.new('ShaderNodeValToRGB')
    ramp_end.color_ramp.elements[1].position = shadow_size
    tree.links.new(fresnel_end.outputs[0], ramp_end.inputs['Fac'])
    tree.links.new(ramp_end.outputs['Color'], mix_end.inputs[0])
    tree.links.new(add_shader.outputs[0], mix_end.inputs[1])

    # add normal to the color
    fresnel_node = tree.nodes.new('ShaderNodeFresnel')
    texture_node = tree.nodes.new('ShaderNodeTexCoord')
    tree.links.new(texture_node.outputs['Normal'], fresnel_node.inputs['Normal'])
    for ii in range(len(ramp_list)):
        tree.links.new(fresnel_node.outputs[0], ramp_list[ii].inputs['Fac'])

    return mat