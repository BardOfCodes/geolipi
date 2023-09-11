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


def create_material_tree(color, material_name):
    bpy = import_bpy()
    RGBA = (color[0], color[1],  color[2], 1)
    meshColor = colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)

    mat = bpy.data.materials.new(material_name)

    mat.use_nodes = True
    tree = mat.node_tree
    AOStrength = 0.0
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
    MIXRGB = tree.nodes.new('ShaderNodeMixRGB')
    MIXRGB.blend_type = 'MULTIPLY'
    tree.nodes["Gamma"].inputs["Gamma"].default_value = AOStrength
    tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
    tree.nodes["Gamma"].location.x -= 600

    # set color using Hue/Saturation node
    HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
    HSVNode.inputs['Color'].default_value = meshColor.RGBA
    HSVNode.inputs['Saturation'].default_value = meshColor.S
    HSVNode.inputs['Value'].default_value = meshColor.V
    HSVNode.inputs['Hue'].default_value = meshColor.H
    HSVNode.location.x -= 200

    # set color brightness/contrast
    BCNode = tree.nodes.new('ShaderNodeBrightContrast')
    BCNode.inputs['Bright'].default_value = meshColor.B
    BCNode.inputs['Contrast'].default_value = meshColor.C
    BCNode.location.x -= 400

    # link all the nodes
    tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
    tree.links.new(BCNode.outputs['Color'],
                   tree.nodes['Ambient Occlusion'].inputs['Color'])
    tree.links.new(
        tree.nodes["Ambient Occlusion"].outputs['Color'], MIXRGB.inputs['Color1'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'],
                   tree.nodes['Gamma'].inputs['Color'])
    tree.links.new(tree.nodes["Gamma"].outputs['Color'],
                   MIXRGB.inputs['Color2'])
    tree.links.new(
        MIXRGB.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

    mat = bpy.data.materials[material_name]
    return mat
