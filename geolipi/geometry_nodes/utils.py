import sys
import math
import numpy as np

# Unsafe!!!
def blender_importer(module_name):
    if module_name not in sys.modules:
        eval(f"import {module_name}")
    return sys.modules[module_name]


def import_bpy():
    module_name = 'bpy'
    module = blender_importer(module_name)
    return module


BASE_COLORS = np.array([[0., 0., 0.06666667],
                        [0., 0.01568627, 1.],
                        [0., 0.94901961, 0.],
                        [0., 1., 0.87058824],
                        [1., 0., 0.05882353],
                        [1., 0., 1.],
                        [0.95294118, 1., 0.],
                        [0.9372549, 1., 1.],
                        [0., 0.61176471, 0.20392157],
                        [0., 0.63921569, 1.],
                        [0.75686275, 0., 0.56470588],
                        [0.56078431, 1., 0.51764706],
                        [1., 0.62745098, 0.10588235],
                        [1., 0.61960784, 1.],
                        [0.2627451, 0., 0.60392157],
                        [0., 0.2627451, 0.38039216],
                        [0., 0.97647059, 0.43529412],
                        [0.50980392, 0., 0.10588235],
                        [0.5254902, 0., 1.],
                        [0.25882353, 0.32156863, 0.],
                        [0.20392157, 0.33333333, 0.80784314],
                        [0.49803922, 1., 0.03529412],
                        [0.48235294, 0.96862745, 0.99215686],
                        [1., 0.29803922, 0.41960784],
                        [0.74117647, 0.32156863, 0.],
                        [1., 0.89019608, 0.50196078],
                        [0.76862745, 0.30588235, 0.8627451],
                        [0.50196078, 0.63137255, 0.13333333],
                        [0.49803922, 0.29411765, 0.41568627],
                        [0.49803922, 0.59215686, 1.],
                        [0.23529412, 0.67058824, 0.60392157],
                        [0.70980392, 0.60784314, 0.56862745]])
BASE_COLORS = BASE_COLORS[::-1, :]


def clean_up(avoid_keys=[]):
    print("cleaning up")
    bpy = import_bpy()
    for key in bpy.data.node_groups.keys():
        if not key in avoid_keys:
            group = bpy.data.node_groups[key]
            bpy.data.node_groups.remove(group, do_unlink=True)
    for obj in bpy.data.objects:
        if not obj.name in avoid_keys:
            bpy.data.objects.remove(obj, do_unlink=True)
    for mat in bpy.data.materials:
        if not mat.name in avoid_keys:
            bpy.data.materials.remove(mat, do_unlink=True)
    for obj in bpy.data.meshes:
        if not obj.name in avoid_keys:
            bpy.data.meshes.remove(obj, do_unlink=True)
    for obj in bpy.data.cameras:
        if not obj.name in avoid_keys:
            bpy.data.cameras.remove(obj, do_unlink=True)
    if bpy.context.scene.use_nodes:
        tree = bpy.context.scene.node_tree
        for name, cur_node in tree.nodes.items():
            if not name in ("Render Layers", "Composite"):
                tree.nodes.remove(cur_node)
        inp = tree.nodes['Render Layers']
        composite = tree.nodes['Composite']
        tree.links.new(inp.outputs['Image'], composite.inputs["Image"])
        tree.links.new(inp.outputs['Alpha'], composite.inputs["Alpha"])
        


def init_camera():
    bpy = import_bpy()
    mathutils = blender_importer('mathutils')
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW',
                              location=(2.0, 2.0, 1.5),
                              rotation=(0, 0, 0),
                              scale=(1, 1, 1))
    point = mathutils.Vector((0, 0, 0))

    camera = bpy.data.objects['Camera']
    direction = point - camera.location
    rotQuat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rotQuat.to_euler()

    bpy.context.scene.camera = camera


def init_lights(sun_energy=3.5):
    bpy = import_bpy()
    bpy.ops.object.light_add(
        type='SUN', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    light = bpy.data.objects['Sun']
    light.data.angle = 25/180. * math.pi
    light.data.energy = sun_energy

    bpy.data.scenes[0].world.use_nodes = True
    bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Color'].default_value = (
        0.1, 0.1, 0.1, 1)
    
# https://blender.stackexchange.com/questions/55702/getting-global-axis-value-of-the-vertex-with-the-lowest-value
def get_obj_min_z(obj):
    mw = obj.matrix_world      # Active object's world matrix
    glob_vertex_coordinates = [mw @ v.co for v in obj.data.vertices]  # Global
    # Find the lowest Z value amongst the object's verts
    minZ = min([co.z for co in glob_vertex_coordinates])
    return minZ


def set_render_settings(resolution=1024, shadow_catch=False, shadow_catcher_z=-1, 
                        transparent_mode=True, line_thickness=0.25):
    # Shadow catcher
    bpy = import_bpy()
    if shadow_catch:
        bpy.ops.mesh.primitive_plane_add(
            size=1, enter_editmode=False, align='WORLD', location=(0, 0, shadow_catcher_z), scale=(1, 1, 1))
        plane_obj = bpy.context.active_object
        plane_obj.name = "shadow_catcher"
        plane_obj.scale[0] = 100
        plane_obj.scale[1] = 100
        plane_obj.scale[2] = 100
        plane_obj.is_shadow_catcher = True

    # Render Settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    if transparent_mode:
        bpy.context.scene.render.film_transparent = True
    else:
        bpy.context.scene.render.film_transparent = True
        # set background color to white:
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        inp = tree.nodes['Render Layers']
        composite = tree.nodes['Composite']
        alpha_node = tree.nodes.new(type="CompositorNodeAlphaOver")
        alpha_node.inputs[1].default_value = (16.19, 16.19, 16.19, 1)
        # denoise_node = tree.nodes.new(type="CompositorNodeDenoise")
        # tree.links.new(inp.outputs['Image'], denoise_node.inputs['Image'])
        tree.links.new(inp.outputs['Image'], alpha_node.inputs[2])
        tree.links.new(alpha_node.outputs['Image'], composite.inputs["Image"])
        #Remove link to alpha channel
        for link in inp.outputs['Alpha'].links:
            tree.links.remove(link)
        

    bpy.context.scene.cycles.samples = 512# 32
    bpy.context.scene.render.use_freestyle = True
    bpy.context.scene.render.line_thickness = line_thickness#0.25
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    freestyle_settings = bpy.context.scene.view_layers["ViewLayer"].freestyle_settings
    lineset = freestyle_settings.linesets.active
    lineset.select_silhouette = True
    lineset.select_crease = True
    lineset.select_border = True
    lineset.select_material_boundary = True
    lineset.select_external_contour = True
    lineset.select_edge_mark = True


def render_with_rotation(xy_size, number_frames, save_loc):
    bpy = import_bpy()
    mathutils = blender_importer('mathutils')
    origin_point = mathutils.Vector((0, 0, 0))
    for i in range(number_frames):
        theta = (45 + 360/number_frames * i) * math.pi/180.
        camera_x = xy_size * math.cos(theta)
        camera_y = xy_size * math.sin(theta)
        camera = bpy.data.objects['Camera']
        camera.location = (camera_x, camera_y, camera.location[2])
        direction = origin_point - camera.location
        rotQuat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rotQuat.to_euler()
        save_file_name = f"{save_loc}/{i}.png"
        bpy.context.scene.render.filepath = save_file_name
        bpy.ops.render.render(write_still=True)
